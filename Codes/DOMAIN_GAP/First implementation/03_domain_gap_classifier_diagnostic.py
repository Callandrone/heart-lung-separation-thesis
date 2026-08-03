#!/usr/bin/env python3
"""
Domain classifier for heart/lung domain-gap audit outputs.

Goal
----
Use already-computed acoustic features from `segment_level_features_sampled.csv`
and test whether two domains are automatically distinguishable.

Recommended use:
    LS + segment-level + RMS features only.

The script performs source-disjoint cross-validation: segments from the same
source_id are never split across train/test within the same fold.

Outputs
-------
- domain_classifier_metrics.csv
- domain_classifier_predictions.csv
- domain_classifier_feature_importance.csv
- domain_classifier_config.json
- README_domain_classifier.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KEY_FEATURES = [
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "crest_factor",
    "energy_20_50_ratio",
    "lung_band_50_1800_ratio",
    "ls_band_50_150_ratio",
    "ls_band_150_500_ratio",
    "ls_band_500_1000_ratio",
    "ls_band_1000_1800_ratio",
    "hf_noise_ratio_1000_2000",
]

METADATA_COLS = {
    "domain",
    "source_type",
    "mode",
    "item_id",
    "audio_path",
    "sr",
    "n_samples",
    "duration_s",
    "base_id",
    "segment_index",
    "snr_label",
    "effective_snr_db",
    "source_id",
    "class_name",
    "macro_class",
    "split",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train source-disjoint domain classifiers on LS segment-level RMS audit features."
    )
    p.add_argument(
        "--features-csv",
        required=True,
        type=Path,
        help="Path to segment_level_features_sampled.csv from a domain-gap audit output directory.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory.",
    )
    p.add_argument(
        "--source-type",
        default="LS",
        choices=["LS", "HS"],
        help="Source type to classify. Default: LS.",
    )
    p.add_argument(
        "--mode",
        default="rms",
        choices=["raw", "rms"],
        help="Feature mode. Default: rms.",
    )
    p.add_argument(
        "--domains",
        default=None,
        help="Optional comma-separated pair of domain names. If omitted, the two domains in the CSV are used.",
    )
    p.add_argument(
        "--feature-set",
        default="key",
        choices=["key", "all_numeric"],
        help="Features to use. `key` is recommended for thesis interpretability. Default: key.",
    )
    p.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated explicit feature names. Overrides --feature-set.",
    )
    p.add_argument(
        "--macro-class",
        default=None,
        help="Optional class filter, e.g. normal, crackles, wheezes, other_mixed. Requires macro_class or class_name column.",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of source-disjoint CV folds. Default: 5.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fallback GroupShuffleSplit test size if StratifiedGroupKFold is not usable. Default: 0.25.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    p.add_argument(
        "--max-segments-per-domain",
        type=int,
        default=0,
        help="Optional cap after filtering, sampled approximately source-balanced per domain. 0 = no cap.",
    )
    p.add_argument(
        "--compute-permutation-importance",
        action="store_true",
        help="Compute permutation importance on each test fold. Slower but useful for feature ranking.",
    )
    return p.parse_args()


def infer_features(df: pd.DataFrame, feature_set: str, explicit: Optional[str]) -> List[str]:
    if explicit:
        feats = [x.strip() for x in explicit.split(",") if x.strip()]
    elif feature_set == "key":
        feats = [f for f in KEY_FEATURES if f in df.columns]
    else:
        feats = []
        for c in df.columns:
            if c in METADATA_COLS:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                # Avoid raw amplitude/gain columns by default for a morphology-focused domain classifier.
                # Keep this conservative. Users can pass --features explicitly if needed.
                if c.startswith("raw_") or c in {"rms", "rms_db", "peak_abs", "total_psd_power", "total_psd_power_db"}:
                    continue
                feats.append(c)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"Missing requested feature columns: {missing}")
    if not feats:
        raise ValueError("No usable features selected. Check --feature-set or --features.")
    return feats


def source_balanced_cap(df: pd.DataFrame, cap_per_domain: int, seed: int) -> pd.DataFrame:
    if cap_per_domain <= 0:
        return df

    rng = np.random.default_rng(seed)
    chunks = []
    for domain, dfd in df.groupby("domain", sort=False):
        if len(dfd) <= cap_per_domain:
            chunks.append(dfd)
            continue

        # Sample approximately equally across source_id so one source cannot dominate.
        sources = list(dfd["source_id"].dropna().unique())
        if not sources:
            chunks.append(dfd.sample(n=cap_per_domain, random_state=seed))
            continue

        per_source = max(1, math.ceil(cap_per_domain / len(sources)))
        sampled = []
        for s, dfs in dfd.groupby("source_id", sort=False):
            n = min(per_source, len(dfs))
            sampled.append(dfs.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
        out = pd.concat(sampled, ignore_index=False)
        if len(out) > cap_per_domain:
            out = out.sample(n=cap_per_domain, random_state=seed)
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True)


def build_cv_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    test_size: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # StratifiedGroupKFold requires enough groups per class.
    labels = pd.Series(y)
    group_df = pd.DataFrame({"y": y, "group": groups}).drop_duplicates("group")
    min_groups_per_class = group_df.groupby("y")["group"].nunique().min()
    usable_splits = min(n_splits, int(min_groups_per_class))

    if usable_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=usable_splits, shuffle=True, random_state=seed)
        return list(cv.split(X, y, groups))

    # Fallback for very small class-conditioned runs.
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    return list(gss.split(X, y, groups))


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def get_positive_scores(model, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return model.predict(X_test)


def model_specs(seed: int) -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=2000,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def extract_model_importance(model_name: str, fitted_model, features: Sequence[str]) -> pd.DataFrame:
    if model_name == "logistic_regression":
        clf = fitted_model.named_steps["clf"]
        vals = clf.coef_.ravel()
        return pd.DataFrame(
            {
                "model": model_name,
                "feature": features,
                "importance": vals,
                "abs_importance": np.abs(vals),
                "importance_type": "standardized_logistic_coefficient",
            }
        )
    if model_name == "random_forest":
        clf = fitted_model.named_steps["clf"]
        vals = clf.feature_importances_
        return pd.DataFrame(
            {
                "model": model_name,
                "feature": features,
                "importance": vals,
                "abs_importance": np.abs(vals),
                "importance_type": "gini_importance",
            }
        )
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    required = {"domain", "source_type", "mode", "source_id"}
    missing_req = sorted(required - set(df.columns))
    if missing_req:
        raise ValueError(f"Input CSV is missing required columns: {missing_req}")

    df = df[(df["source_type"] == args.source_type) & (df["mode"] == args.mode)].copy()

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        if len(domains) != 2:
            raise ValueError("--domains must contain exactly two comma-separated domain names.")
        df = df[df["domain"].isin(domains)].copy()
    else:
        domains = sorted(df["domain"].dropna().unique().tolist())
        if len(domains) != 2:
            raise ValueError(f"Expected exactly two domains after filtering, found {domains}. Use --domains.")

    if args.macro_class:
        if "macro_class" in df.columns:
            df = df[df["macro_class"].astype(str) == args.macro_class].copy()
        elif "class_name" in df.columns:
            df = df[df["class_name"].astype(str) == args.macro_class].copy()
        else:
            raise ValueError("--macro-class requested, but neither macro_class nor class_name exists in CSV.")

    df = source_balanced_cap(df, args.max_segments_per_domain, args.seed)

    features = infer_features(df, args.feature_set, args.features)
    # Drop rows where all features are NaN/inf after replacement.
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["domain", "source_id"])
    df = df.dropna(subset=features, how="all")

    if len(df) == 0:
        raise ValueError("No rows left after filtering.")

    # Label convention: domain[0] = 0, domain[1] = 1. Direction is saved in config.
    domains = [d for d in domains if d in set(df["domain"])]
    if len(domains) != 2:
        raise ValueError(f"Need two domains with rows after filtering. Found {domains}.")
    domain_to_label = {domains[0]: 0, domains[1]: 1}
    label_to_domain = {0: domains[0], 1: domains[1]}

    df["y"] = df["domain"].map(domain_to_label).astype(int)
    df["group"] = df["domain"].astype(str) + "::" + df["source_id"].astype(str)

    X = df[features].copy()
    y = df["y"].to_numpy()
    groups = df["group"].to_numpy()

    n_by_domain = df.groupby("domain").size().rename("n_segments")
    n_sources_by_domain = df.groupby("domain")["source_id"].nunique().rename("n_sources")
    domain_counts = pd.concat([n_by_domain, n_sources_by_domain], axis=1).reset_index()
    domain_counts.to_csv(args.out_dir / "domain_classifier_input_counts.csv", index=False)

    splits = build_cv_splits(X, y, groups, args.n_splits, args.seed, args.test_size)
    models = model_specs(args.seed)

    metric_rows = []
    pred_rows = []
    importance_rows = []
    perm_rows = []

    for model_name, model in models.items():
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]

            # Guard against accidental leakage.
            overlap = set(groups_train).intersection(set(groups_test))
            if overlap:
                raise RuntimeError(f"Group leakage detected in fold {fold_idx}: {list(overlap)[:5]}")

            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)
            y_score = get_positive_scores(fitted, X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            metric_rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "n_train_segments": len(train_idx),
                    "n_test_segments": len(test_idx),
                    "n_train_sources": len(set(groups_train)),
                    "n_test_sources": len(set(groups_test)),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "f1_positive_domain": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc_positive_domain": safe_auc(y_test, y_score),
                    "tn_domain0": tn,
                    "fp_domain0_as_domain1": fp,
                    "fn_domain1_as_domain0": fn,
                    "tp_domain1": tp,
                    "negative_domain_0": label_to_domain[0],
                    "positive_domain_1": label_to_domain[1],
                }
            )

            fold_df = df.iloc[test_idx][["domain", "source_id", "item_id", "class_name"] if "class_name" in df.columns else ["domain", "source_id", "item_id"]].copy()
            if "macro_class" in df.columns:
                fold_df["macro_class"] = df.iloc[test_idx]["macro_class"].to_numpy()
            fold_df["model"] = model_name
            fold_df["fold"] = fold_idx
            fold_df["true_label"] = y_test
            fold_df["pred_label"] = y_pred
            fold_df["score_positive_domain"] = y_score
            fold_df["true_domain"] = [label_to_domain[int(v)] for v in y_test]
            fold_df["pred_domain"] = [label_to_domain[int(v)] for v in y_pred]
            pred_rows.append(fold_df)

            imp = extract_model_importance(model_name, fitted, features)
            if not imp.empty:
                imp["fold"] = fold_idx
                importance_rows.append(imp)

            if args.compute_permutation_importance:
                try:
                    pi = permutation_importance(
                        fitted,
                        X_test,
                        y_test,
                        n_repeats=10,
                        random_state=args.seed + fold_idx,
                        scoring="balanced_accuracy",
                        n_jobs=-1,
                    )
                    perm_rows.append(
                        pd.DataFrame(
                            {
                                "model": model_name,
                                "fold": fold_idx,
                                "feature": features,
                                "importance": pi.importances_mean,
                                "importance_std": pi.importances_std,
                                "abs_importance": np.abs(pi.importances_mean),
                                "importance_type": "permutation_balanced_accuracy_drop",
                            }
                        )
                    )
                except Exception as e:
                    print(f"[WARN] permutation importance failed for {model_name} fold {fold_idx}: {e}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_dir / "domain_classifier_metrics.csv", index=False)

    # Aggregate metric summary.
    metric_cols = ["accuracy", "balanced_accuracy", "f1_positive_domain", "roc_auc_positive_domain"]
    summary = (
        metrics.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns.to_flat_index()]
    summary.to_csv(args.out_dir / "domain_classifier_metrics_summary.csv", index=False)

    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    preds.to_csv(args.out_dir / "domain_classifier_predictions.csv", index=False)

    if importance_rows:
        imp_all = pd.concat(importance_rows, ignore_index=True)
        imp_all.to_csv(args.out_dir / "domain_classifier_feature_importance_per_fold.csv", index=False)
        imp_summary = (
            imp_all.groupby(["model", "feature", "importance_type"], as_index=False)
            .agg(
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
                abs_importance_mean=("abs_importance", "mean"),
                abs_importance_std=("abs_importance", "std"),
            )
            .sort_values(["model", "abs_importance_mean"], ascending=[True, False])
        )
        imp_summary.to_csv(args.out_dir / "domain_classifier_feature_importance.csv", index=False)

    if perm_rows:
        perm_all = pd.concat(perm_rows, ignore_index=True)
        perm_all.to_csv(args.out_dir / "domain_classifier_permutation_importance_per_fold.csv", index=False)
        perm_summary = (
            perm_all.groupby(["model", "feature", "importance_type"], as_index=False)
            .agg(
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
                abs_importance_mean=("abs_importance", "mean"),
                abs_importance_std=("abs_importance", "std"),
            )
            .sort_values(["model", "abs_importance_mean"], ascending=[True, False])
        )
        perm_summary.to_csv(args.out_dir / "domain_classifier_permutation_importance.csv", index=False)

    config = {
        "features_csv": str(args.features_csv),
        "out_dir": str(args.out_dir),
        "source_type": args.source_type,
        "mode": args.mode,
        "domains": domains,
        "domain_to_label": domain_to_label,
        "positive_domain_label_1": label_to_domain[1],
        "feature_set": args.feature_set,
        "features": features,
        "macro_class": args.macro_class,
        "n_splits_requested": args.n_splits,
        "n_splits_used": len(splits),
        "seed": args.seed,
        "max_segments_per_domain": args.max_segments_per_domain,
        "compute_permutation_importance": bool(args.compute_permutation_importance),
        "n_rows_after_filtering": int(len(df)),
        "n_sources_after_filtering": int(df["group"].nunique()),
    }
    with open(args.out_dir / "domain_classifier_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Markdown README.
    best_lines = []
    for _, row in summary.iterrows():
        best_lines.append(
            f"| {row['model']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | "
            f"{row['roc_auc_positive_domain_mean']:.4f} ± {row['roc_auc_positive_domain_std']:.4f} |"
        )

    readme = f"""# Domain classifier report

Input CSV: `{args.features_csv}`

Filtered rows: `{len(df)}`  
Source type: `{args.source_type}`  
Mode: `{args.mode}`  
Domains: `{domains[0]}` vs `{domains[1]}`  
Positive class / label 1: `{label_to_domain[1]}`  
Source-disjoint folds used: `{len(splits)}`  
Feature set: `{args.feature_set}`  
Features: `{', '.join(features)}`

## Input counts

{domain_counts.to_markdown(index=False)}

## Cross-validation metrics

| Model | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|
{chr(10).join(best_lines)}

## Interpretation guide

- Accuracy/AUC close to 0.50: domains are not separable from these features.
- 0.65–0.75: moderate domain separation.
- 0.80–0.90: strong domain separation.
- >0.90: very strong domain separation.

This classifier is diagnostic: it does not prove causality and does not directly measure separation quality. It tests whether LS segment-level RMS acoustic features contain enough information to identify the dataset/domain while avoiding source leakage.
"""
    with open(args.out_dir / "README_domain_classifier.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print("Saved outputs to:", args.out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

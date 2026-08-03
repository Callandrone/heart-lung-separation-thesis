#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical domain-gap analysis: EXP_H / ICBHI vs Torabi LS.

Input expected: segment_level_features_sampled.csv from the existing DOMAIN_GAP audit.
Main analysis:
  1) filter LS + rms + EXP_H_ALL/TORABI_FOLD2_ALL;
  2) aggregate segment-level features at source level;
  3) compute observed SMD at segment and source level;
  4) source-level bootstrap confidence intervals for SMD;
  5) source-level permutation test on mean absolute SMD;
  6) source-level logistic domain classifier before/after train-fold mean/std matching;
  7) CORAL covariance distance before/after matching;
  8) save CSVs, plots and README.

The key point is that inferential/statistical analysis is done at source level,
not at overlapping 2 s segment level, to avoid pseudo-replication.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


META_COLS = {
    "domain", "source_type", "mode", "item_id", "audio_path", "source_id",
    "class_name", "macro_class", "sr", "n_samples", "duration_s", "base_id",
    "segment_index", "split"
}

# Main feature set used for LS acoustic morphology after RMS normalization.
# This excludes raw amplitude columns and keeps spectral, temporal and band-ratio descriptors.
KEY_FEATURES = [
    "crest_factor", "zcr", "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "energy_20_50_db", "energy_20_50_ratio",
    "energy_20_80_db", "energy_20_80_ratio",
    "energy_20_500_db", "energy_20_500_ratio",
    "energy_50_150_db", "energy_50_150_ratio",
    "energy_50_1800_db", "energy_50_1800_ratio",
    "energy_80_150_db", "energy_80_150_ratio",
    "energy_150_500_db", "energy_150_500_ratio",
    "energy_200_500_db", "energy_200_500_ratio",
    "energy_500_1000_db", "energy_500_1000_ratio",
    "energy_1000_1800_db", "energy_1000_1800_ratio",
    "energy_1000_2000_db", "energy_1000_2000_ratio",
    "heart_band_20_500_ratio", "heart_band_20_80_ratio",
    "hs_band_20_80_ratio", "hs_band_80_150_ratio", "hs_band_150_500_ratio",
    "lung_band_50_1800_ratio",
    "ls_band_20_50_ratio", "ls_band_50_150_ratio", "ls_band_150_500_ratio",
    "ls_band_500_1000_ratio", "ls_band_1000_1800_ratio",
    "hf_noise_ratio_1000_2000",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_numeric_features(df: pd.DataFrame, feature_set: str) -> List[str]:
    if feature_set == "key":
        features = [c for c in KEY_FEATURES if c in df.columns]
    elif feature_set == "auto":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in num_cols if c not in META_COLS]
        # Avoid absolute/raw amplitude descriptors in RMS-normalized morphology analysis.
        features = [c for c in features if not c.startswith("raw_")]
        features = [c for c in features if c not in {"rms", "rms_db", "peak_abs", "total_psd_power", "total_psd_power_db"}]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    good = []
    for c in features:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() == 0:
            continue
        if np.nanstd(x) < 1e-12:
            continue
        good.append(c)
    return good


def load_and_filter(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str]]:
    usecols = None
    df = pd.read_csv(args.input_csv, usecols=usecols)

    required = {"domain", "source_type", "mode", args.group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[
        (df["domain"].isin([args.source_domain, args.target_domain]))
        & (df["source_type"].astype(str).str.upper() == args.source_type.upper())
        & (df["mode"].astype(str).str.lower() == args.mode.lower())
    ].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check domain/source_type/mode arguments.")

    # Optional debug/downsample switch. Keep disabled for final thesis runs.
    if args.max_segments_per_domain and args.max_segments_per_domain > 0:
        parts = []
        for d, g in df.groupby("domain", sort=False):
            n = min(args.max_segments_per_domain, len(g))
            parts.append(g.sample(n=n, random_state=args.random_state))
        df = pd.concat(parts, ignore_index=True)

    features = finite_numeric_features(df, args.feature_set)
    if not features:
        raise ValueError("No usable numeric features found.")

    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + [args.group_col, "domain"])

    return df, features


def aggregate_source_level(df: pd.DataFrame, features: List[str], group_col: str) -> pd.DataFrame:
    group_cols = ["domain", group_col]
    optional = [c for c in ["class_name", "macro_class", "split"] if c in df.columns]

    feat_agg = df.groupby(group_cols, as_index=False)[features].mean()

    if optional:
        meta = (
            df.groupby(group_cols, as_index=False)[optional]
            .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) else "")
        )
        out = feat_agg.merge(meta, on=group_cols, how="left")
    else:
        out = feat_agg

    counts = df.groupby(group_cols, as_index=False).size().rename(columns={"size": "n_segments"})
    out = out.merge(counts, on=group_cols, how="left")
    return out


def smd_table(df: pd.DataFrame, features: List[str], source_domain: str, target_domain: str) -> pd.DataFrame:
    a = df[df["domain"] == source_domain]
    b = df[df["domain"] == target_domain]
    rows = []
    for f in features:
        xa = a[f].to_numpy(dtype=float)
        xb = b[f].to_numpy(dtype=float)
        ma, mb = np.nanmean(xa), np.nanmean(xb)
        sa, sb = np.nanstd(xa, ddof=1), np.nanstd(xb, ddof=1)
        pooled = math.sqrt((sa * sa + sb * sb) / 2.0) if np.isfinite(sa) and np.isfinite(sb) else np.nan
        smd = (mb - ma) / pooled if pooled and pooled > 0 else np.nan
        rows.append({
            "feature": f,
            f"mean_{source_domain}": ma,
            f"std_{source_domain}": sa,
            f"mean_{target_domain}": mb,
            f"std_{target_domain}": sb,
            "smd_target_minus_source": smd,
            "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
            "n_source": int(np.isfinite(xa).sum()),
            "n_target": int(np.isfinite(xb).sum()),
        })
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def bootstrap_smd_source_level(
    src_df: pd.DataFrame,
    features: List[str],
    source_domain: str,
    target_domain: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    src = src_df[src_df["domain"] == source_domain][features].to_numpy(dtype=float)
    tgt = src_df[src_df["domain"] == target_domain][features].to_numpy(dtype=float)
    n_src, n_tgt = src.shape[0], tgt.shape[0]
    boot = np.empty((n_boot, len(features)), dtype=float)

    for i in range(n_boot):
        ia = rng.integers(0, n_src, size=n_src)
        ib = rng.integers(0, n_tgt, size=n_tgt)
        a = src[ia]
        b = tgt[ib]
        ma = np.nanmean(a, axis=0)
        mb = np.nanmean(b, axis=0)
        sa = np.nanstd(a, axis=0, ddof=1)
        sb = np.nanstd(b, axis=0, ddof=1)
        pooled = np.sqrt((sa * sa + sb * sb) / 2.0)
        boot[i, :] = np.where(pooled > 0, (mb - ma) / pooled, np.nan)

    obs = smd_table(src_df, features, source_domain, target_domain).set_index("feature")
    rows = []
    for j, f in enumerate(features):
        vals = boot[:, j]
        vals = vals[np.isfinite(vals)]
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
        obs_smd = float(obs.loc[f, "smd_target_minus_source"])
        sign_stability = float(np.mean(np.sign(vals) == np.sign(obs_smd))) if len(vals) and obs_smd != 0 else np.nan
        rows.append({
            "feature": f,
            "smd_observed_source_level": obs_smd,
            "abs_smd_observed": abs(obs_smd),
            "ci_low_2p5": ci_low,
            "ci_high_97p5": ci_high,
            "ci_excludes_zero": bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)) if np.isfinite(ci_low) else False,
            "sign_stability": sign_stability,
        })
    return pd.DataFrame(rows).sort_values("abs_smd_observed", ascending=False).reset_index(drop=True)


def permutation_test_mean_abs_smd(
    src_df: pd.DataFrame,
    features: List[str],
    source_domain: str,
    target_domain: str,
    n_perm: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = src_df[features].to_numpy(dtype=float)
    labels = src_df["domain"].to_numpy()
    n_target = int(np.sum(labels == target_domain))

    obs_tbl = smd_table(src_df, features, source_domain, target_domain)
    obs_mean_abs = float(obs_tbl["abs_smd"].mean())
    obs_median_abs = float(obs_tbl["abs_smd"].median())
    obs_max_abs = float(obs_tbl["abs_smd"].max())

    stats = np.empty((n_perm, 3), dtype=float)
    n = len(labels)
    for i in range(n_perm):
        target_idx = rng.choice(n, size=n_target, replace=False)
        mask_t = np.zeros(n, dtype=bool)
        mask_t[target_idx] = True
        a = x[~mask_t]
        b = x[mask_t]
        ma = np.nanmean(a, axis=0)
        mb = np.nanmean(b, axis=0)
        sa = np.nanstd(a, axis=0, ddof=1)
        sb = np.nanstd(b, axis=0, ddof=1)
        pooled = np.sqrt((sa * sa + sb * sb) / 2.0)
        smd = np.where(pooled > 0, (mb - ma) / pooled, np.nan)
        abs_smd = np.abs(smd[np.isfinite(smd)])
        stats[i, 0] = np.mean(abs_smd)
        stats[i, 1] = np.median(abs_smd)
        stats[i, 2] = np.max(abs_smd)

    summary = pd.DataFrame([
        {
            "statistic": "mean_abs_smd",
            "observed": obs_mean_abs,
            "perm_mean": float(np.mean(stats[:, 0])),
            "perm_p95": float(np.percentile(stats[:, 0], 95)),
            "perm_p99": float(np.percentile(stats[:, 0], 99)),
            "p_value_greater_equal": float((np.sum(stats[:, 0] >= obs_mean_abs) + 1) / (n_perm + 1)),
            "n_perm": n_perm,
        },
        {
            "statistic": "median_abs_smd",
            "observed": obs_median_abs,
            "perm_mean": float(np.mean(stats[:, 1])),
            "perm_p95": float(np.percentile(stats[:, 1], 95)),
            "perm_p99": float(np.percentile(stats[:, 1], 99)),
            "p_value_greater_equal": float((np.sum(stats[:, 1] >= obs_median_abs) + 1) / (n_perm + 1)),
            "n_perm": n_perm,
        },
        {
            "statistic": "max_abs_smd",
            "observed": obs_max_abs,
            "perm_mean": float(np.mean(stats[:, 2])),
            "perm_p95": float(np.percentile(stats[:, 2], 95)),
            "perm_p99": float(np.percentile(stats[:, 2], 99)),
            "p_value_greater_equal": float((np.sum(stats[:, 2] >= obs_max_abs) + 1) / (n_perm + 1)),
            "n_perm": n_perm,
        },
    ])
    dist = pd.DataFrame(stats, columns=["mean_abs_smd", "median_abs_smd", "max_abs_smd"])
    return summary, dist


def fit_matching_params(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, np.ndarray]:
    # y: 0 = source, 1 = target
    xs = X_train[y_train == 0]
    xt = X_train[y_train == 1]
    mu_s = np.nanmean(xs, axis=0)
    sd_s = np.nanstd(xs, axis=0, ddof=1)
    mu_t = np.nanmean(xt, axis=0)
    sd_t = np.nanstd(xt, axis=0, ddof=1)
    sd_s = np.where(sd_s < 1e-12, 1.0, sd_s)
    sd_t = np.where(sd_t < 1e-12, 1.0, sd_t)
    return {"mu_s": mu_s, "sd_s": sd_s, "mu_t": mu_t, "sd_t": sd_t}


def apply_target_meanstd_matching(X: np.ndarray, y: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    out = X.copy().astype(float)
    idx = y == 1
    out[idx] = ((out[idx] - params["mu_t"]) / params["sd_t"]) * params["sd_s"] + params["mu_s"]
    return out


def classifier_before_after_matching(
    src_df: pd.DataFrame,
    features: List[str],
    source_domain: str,
    target_domain: str,
    n_splits: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = src_df.copy()
    data["label"] = (data["domain"] == target_domain).astype(int)
    X = data[features].to_numpy(dtype=float).copy()
    y = data["label"].to_numpy(dtype=int)

    # Replace any remaining NaNs with column medians computed on all source-level rows.
    med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])

    min_class = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = max(2, min(n_splits, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    metrics_rows = []
    coef_rows = []
    pred_rows = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        for condition in ["before_matching", "after_meanstd_matching"]:
            if condition == "before_matching":
                Xtr_c, Xte_c = Xtr, Xte
            else:
                params = fit_matching_params(Xtr, ytr)
                Xtr_c = apply_target_meanstd_matching(Xtr, ytr, params)
                Xte_c = apply_target_meanstd_matching(Xte, yte, params)

            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed),
            )
            clf.fit(Xtr_c, ytr)
            prob = clf.predict_proba(Xte_c)[:, 1]
            pred = (prob >= 0.5).astype(int)

            acc = accuracy_score(yte, pred)
            bacc = balanced_accuracy_score(yte, pred)
            auc = roc_auc_score(yte, prob) if len(np.unique(yte)) == 2 else np.nan
            metrics_rows.append({
                "condition": condition,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "roc_auc": auc,
            })

            # Extract standardized logistic coefficients.
            lr = clf.named_steps["logisticregression"]
            coefs = lr.coef_[0]
            for f, c in zip(features, coefs):
                coef_rows.append({
                    "condition": condition,
                    "fold": fold,
                    "feature": f,
                    "coef": c,
                    "abs_coef": abs(c),
                })

            for idx, yt, pr, pp in zip(te, yte, pred, prob):
                pred_rows.append({
                    "condition": condition,
                    "fold": fold,
                    "row_index": int(idx),
                    "domain": data.iloc[idx]["domain"],
                    "true_label_target": int(yt),
                    "pred_label_target": int(pr),
                    "prob_target": float(pp),
                })

    metrics = pd.DataFrame(metrics_rows)
    coefs = pd.DataFrame(coef_rows)
    preds = pd.DataFrame(pred_rows)

    summary = (
        metrics.groupby("condition", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
            n_folds=("fold", "nunique"),
        )
    )
    return metrics, summary, coefs, preds


def source_level_meanstd_matched_table(src_df: pd.DataFrame, features: List[str], source_domain: str, target_domain: str) -> pd.DataFrame:
    # Global version only for descriptive SMD before/after; classifier matching is fold-safe separately.
    data = src_df.copy()
    y = (data["domain"] == target_domain).astype(int).to_numpy()
    X = data[features].to_numpy(dtype=float).copy()
    med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])
    params = fit_matching_params(X, y)
    X_matched = apply_target_meanstd_matching(X, y, params)
    matched = data[["domain"]].copy()
    for j, f in enumerate(features):
        matched[f] = X_matched[:, j]
    before = smd_table(data, features, source_domain, target_domain)
    before["condition"] = "before_matching"
    after = smd_table(matched, features, source_domain, target_domain)
    after["condition"] = "after_global_meanstd_matching_descriptive_only"
    return pd.concat([before, after], ignore_index=True)


def coral_distance(Xs: np.ndarray, Xt: np.ndarray) -> float:
    if Xs.shape[0] < 2 or Xt.shape[0] < 2:
        return np.nan
    Cs = np.cov(Xs, rowvar=False)
    Ct = np.cov(Xt, rowvar=False)
    d = Xs.shape[1]
    return float(np.linalg.norm(Cs - Ct, ord="fro") / max(d, 1))


def coral_before_after(src_df: pd.DataFrame, features: List[str], source_domain: str, target_domain: str) -> pd.DataFrame:
    data = src_df.copy()
    y = (data["domain"] == target_domain).astype(int).to_numpy()
    X = data[features].to_numpy(dtype=float).copy()
    med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])

    # Standardize using source-domain statistics to make covariance units comparable.
    src_mask = y == 0
    mu_s = np.mean(X[src_mask], axis=0)
    sd_s = np.std(X[src_mask], axis=0, ddof=1)
    sd_s = np.where(sd_s < 1e-12, 1.0, sd_s)
    Xz = (X - mu_s) / sd_s

    params = fit_matching_params(X, y)
    Xm = apply_target_meanstd_matching(X, y, params)
    Xmz = (Xm - mu_s) / sd_s

    rows = []
    rows.append({
        "condition": "before_matching_source_standardized",
        "coral_frobenius_over_d": coral_distance(Xz[y == 0], Xz[y == 1]),
        "n_source": int(np.sum(y == 0)),
        "n_target": int(np.sum(y == 1)),
        "n_features": len(features),
    })
    rows.append({
        "condition": "after_global_meanstd_matching_source_standardized",
        "coral_frobenius_over_d": coral_distance(Xmz[y == 0], Xmz[y == 1]),
        "n_source": int(np.sum(y == 0)),
        "n_target": int(np.sum(y == 1)),
        "n_features": len(features),
    })
    return pd.DataFrame(rows)


def plot_top_smd_ci(boot_ci: pd.DataFrame, out_path: Path, top_k: int) -> None:
    d = boot_ci.head(top_k).iloc[::-1].copy()
    y = np.arange(len(d))
    x = d["smd_observed_source_level"].to_numpy(float)
    low = d["ci_low_2p5"].to_numpy(float)
    high = d["ci_high_97p5"].to_numpy(float)
    xerr = np.vstack([x - low, high - x])

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(d))))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"].tolist())
    ax.set_xlabel("SMD: Torabi - EXP_H/ICBHI (source-level, LS RMS)")
    ax.set_title(f"Top {top_k} source-level SMD with bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_auc(summary: pd.DataFrame, out_path: Path) -> None:
    d = summary.copy()
    x = np.arange(len(d))
    y = d["roc_auc_mean"].to_numpy(float)
    err = d["roc_auc_std"].fillna(0.0).to_numpy(float)
    labels = d["condition"].str.replace("_", " ").tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y, yerr=err, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Domain classifier before/after mean-std matching")
    ax.axhline(0.5, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    features: List[str],
    counts: pd.DataFrame,
    seg_smd: pd.DataFrame,
    src_smd: pd.DataFrame,
    boot_ci: pd.DataFrame,
    perm_summary: pd.DataFrame,
    clf_summary: pd.DataFrame,
    coral: pd.DataFrame,
) -> None:
    top = src_smd.head(8)[["feature", "smd_target_minus_source", "abs_smd"]]
    perm_main = perm_summary[perm_summary["statistic"] == "mean_abs_smd"].iloc[0].to_dict()

    with open(out_dir / "README_statistical_domain_gap.md", "w", encoding="utf-8") as f:
        f.write("# Statistical Domain Gap: EXP_H/ICBHI vs Torabi\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Input CSV: `{args.input_csv}`\n")
        f.write(f"- Source domain: `{args.source_domain}`\n")
        f.write(f"- Target domain: `{args.target_domain}`\n")
        f.write(f"- Filter: source_type=`{args.source_type}`, mode=`{args.mode}`\n")
        f.write(f"- Grouping unit for statistical analysis: `{args.group_col}`\n")
        f.write(f"- Feature set: `{args.feature_set}` ({len(features)} features)\n")
        f.write(f"- Bootstrap iterations: {args.n_boot}\n")
        f.write(f"- Permutation iterations: {args.n_perm}\n\n")

        f.write("## Counts\n\n")
        f.write(counts.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Main result: source-level SMD\n\n")
        f.write(top.to_markdown(index=False))
        f.write("\n\n")
        f.write("Positive SMD means Torabi > EXP_H/ICBHI for that feature.\n\n")

        f.write("## Bootstrap interpretation\n\n")
        n_excl = int(boot_ci["ci_excludes_zero"].sum())
        f.write(f"- Features with 95% bootstrap CI excluding zero: {n_excl}/{len(boot_ci)}\n")
        f.write("- Bootstrap is performed at source level, not segment level.\n\n")

        f.write("## Permutation test\n\n")
        f.write(
            f"Observed mean_abs_smd = {perm_main['observed']:.4f}; "
            f"permutation p-value = {perm_main['p_value_greater_equal']:.6f}.\n\n"
        )

        f.write("## Domain classifier before/after matching\n\n")
        f.write(clf_summary.to_markdown(index=False))
        f.write("\n\n")
        f.write(
            "The after-matching classifier removes train-fold first-order mean/std differences "
            "from the target domain. If ROC-AUC remains above chance, the residual gap is not "
            "only a univariate mean/variance shift.\n\n"
        )

        f.write("## CORAL covariance distance\n\n")
        f.write(coral.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Files generated\n\n")
        for name in sorted(p.name for p in out_dir.iterdir() if p.is_file()):
            f.write(f"- `{name}`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Path to segment_level_features_sampled.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--source-domain", default="EXP_H_ALL")
    parser.add_argument("--target-domain", default="TORABI_FOLD2_ALL")
    parser.add_argument("--source-type", default="LS")
    parser.add_argument("--mode", default="rms")
    parser.add_argument("--group-col", default="source_id")
    parser.add_argument("--feature-set", choices=["key", "auto"], default="key")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-segments-per-domain", type=int, default=0, help="Debug only; 0 = disabled")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("[1/9] Loading and filtering input CSV...")
    df, features = load_and_filter(args)
    counts = (
        df.groupby(["domain", "source_type", "mode"], as_index=False)
        .agg(n_segments=(args.group_col, "size"), n_sources=(args.group_col, "nunique"))
    )
    counts.to_csv(out_dir / "00_input_counts.csv", index=False)

    with open(out_dir / "00_config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "features": features}, f, indent=2)

    print(f"  rows after filter: {len(df)}")
    print(f"  features: {len(features)}")
    print(counts.to_string(index=False))

    print("[2/9] Aggregating to source level...")
    src_df = aggregate_source_level(df, features, args.group_col)
    src_df.to_csv(out_dir / "01_source_level_features_LS_rms.csv", index=False)

    print("[3/9] Computing observed SMD at segment and source level...")
    seg_smd = smd_table(df, features, args.source_domain, args.target_domain)
    src_smd = smd_table(src_df, features, args.source_domain, args.target_domain)
    seg_smd.to_csv(out_dir / "02_smd_observed_segment_level.csv", index=False)
    src_smd.to_csv(out_dir / "03_smd_observed_source_level.csv", index=False)

    print("[4/9] Running source-level bootstrap CI for SMD...")
    boot_ci = bootstrap_smd_source_level(
        src_df, features, args.source_domain, args.target_domain, args.n_boot, args.random_state
    )
    boot_ci.to_csv(out_dir / "04_smd_bootstrap_ci_source_level.csv", index=False)

    print("[5/9] Running source-level permutation test...")
    perm_summary, perm_dist = permutation_test_mean_abs_smd(
        src_df, features, args.source_domain, args.target_domain, args.n_perm, args.random_state
    )
    perm_summary.to_csv(out_dir / "05_permutation_test_summary.csv", index=False)
    perm_dist.to_csv(out_dir / "05_permutation_test_distribution.csv", index=False)

    print("[6/9] Running domain classifier before/after mean-std matching...")
    metrics, clf_summary, coefs, preds = classifier_before_after_matching(
        src_df, features, args.source_domain, args.target_domain, args.n_splits, args.random_state
    )
    metrics.to_csv(out_dir / "06_domain_classifier_before_after_matching_metrics_per_fold.csv", index=False)
    clf_summary.to_csv(out_dir / "06_domain_classifier_before_after_matching_summary.csv", index=False)
    coefs.to_csv(out_dir / "06_domain_classifier_before_after_matching_coefficients.csv", index=False)
    preds.to_csv(out_dir / "06_domain_classifier_before_after_matching_predictions.csv", index=False)

    coef_summary = (
        coefs.groupby(["condition", "feature"], as_index=False)
        .agg(coef_mean=("coef", "mean"), abs_coef_mean=("abs_coef", "mean"), abs_coef_std=("abs_coef", "std"))
        .sort_values(["condition", "abs_coef_mean"], ascending=[True, False])
    )
    coef_summary.to_csv(out_dir / "06_domain_classifier_feature_importance_summary.csv", index=False)

    print("[7/9] Computing descriptive SMD before/after global matching and CORAL distance...")
    smd_match = source_level_meanstd_matched_table(src_df, features, args.source_domain, args.target_domain)
    smd_match.to_csv(out_dir / "07_smd_before_after_global_meanstd_matching_descriptive.csv", index=False)
    coral = coral_before_after(src_df, features, args.source_domain, args.target_domain)
    coral.to_csv(out_dir / "08_coral_before_after_matching.csv", index=False)

    print("[8/9] Saving plots...")
    plot_top_smd_ci(boot_ci, out_dir / "09_top_smd_bootstrap_ci.png", args.top_k)
    plot_auc(clf_summary, out_dir / "10_domain_classifier_auc_before_after_matching.png")

    print("[9/9] Writing README...")
    write_readme(out_dir, args, features, counts, seg_smd, src_smd, boot_ci, perm_summary, clf_summary, coral)

    print("\nDONE")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
09_domain_gap_distance_vs_performance.py
========================================

Diagnostic analysis linking the acoustic LS domain gap to model performance.

Goal
----
For each Torabi lung source, compute its acoustic distance from the EXP_H/ICBHI
lung-source reference cloud in RMS-normalized LS feature space, then test whether
sources farther from EXP_H are also harder for the final separator.

Main question
-------------
Are Torabi LS sources that are acoustically farther from EXP_H/ICBHI associated
with lower LS SI-SDR / lower mean SI-SDR / higher failure rates?

Expected inputs
---------------
1) Domain-gap segment-level feature CSV, for example:
   outputs/domain_gap/<run>/segment_level_features_sampled.csv

2) Final error-analysis per-sample metrics CSV, for example:
   outputs/results/ERROR_ANALYSIS/<run>/metrics/per_sample_error_metrics.csv

The script tries to attach each error-analysis sample to its Torabi LS source by
joining (base_id, segment_index) against the Torabi LS rows in the domain-gap
feature CSV. If your per-sample metrics already contain ls_source_id, that column
is used directly.

Outputs
-------
- source-level distance table
- source-level performance table
- merged distance/performance table
- Pearson/Spearman correlations + bootstrap confidence intervals
- scatter plots
- README with interpretation guide

Methodological note
-------------------
The analysis is source-level. This avoids treating overlapping 2 s segments as
fully independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None


# Same feature set used in the statistical domain-gap analysis.
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

META_COLS = {
    "domain", "source_type", "mode", "item_id", "audio_path", "source_id",
    "class_name", "macro_class", "sr", "n_samples", "duration_s", "base_id",
    "segment_index", "split", "n_rows_in_manifest", "n_segments",
}

DEFAULT_PERFORMANCE_METRICS = [
    "si_sdr_l_mean", "si_sdr_l_median", "si_sdr_h_mean", "mean_si_sdr_mean",
    "mix_corr_mean", "mix_nmse_db_mean", "abs_local_snr_mean", "target_local_snr_db_mean",
    "ls_low_sisdr_rate", "hs_low_sisdr_rate", "hs_leak_in_ls_rate", "ls_leak_in_hs_rate",
    "mix_inconsistent_rate", "possible_swap_rate", "ok_rate",
]

DEFAULT_DISTANCE_COLS = [
    "distance_z_centroid",
    "distance_z_nn1",
    "distance_z_nn5_mean",
    "distance_mahalanobis_reg",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_error_metrics_path(path: Path) -> Path:
    """Accept either per_sample_error_metrics.csv or an error-analysis output directory."""
    if path.is_file():
        return path
    candidates = [
        path / "metrics" / "per_sample_error_metrics.csv",
        path / "per_sample_error_metrics.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = list(path.rglob("per_sample_error_metrics.csv")) if path.exists() else []
    if found:
        return found[0]
    raise FileNotFoundError(
        f"Could not resolve per_sample_error_metrics.csv from: {path}\n"
        "Pass either the exact CSV or the error-analysis output directory."
    )


def finite_numeric_features(df: pd.DataFrame, feature_set: str) -> List[str]:
    if feature_set == "key":
        features = [c for c in KEY_FEATURES if c in df.columns]
    elif feature_set == "auto":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in num_cols if c not in META_COLS]
        features = [c for c in features if not c.startswith("raw_")]
        features = [c for c in features if not c.endswith("_power")]
        features = [c for c in features if c not in {"rms", "rms_db", "peak_abs", "total_psd_power", "total_psd_power_db"}]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    good = []
    for c in features:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).sum() > 2 and np.nanstd(vals) > 0:
            good.append(c)
    if not good:
        raise RuntimeError("No usable numeric features found after filtering.")
    return good


def first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def aggregate_to_source_level(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    agg = {f: "mean" for f in features}
    for c in ["class_name", "macro_class", "split"]:
        if c in df.columns:
            agg[c] = first_non_null
    out = df.groupby(["domain", "source_id"], dropna=False).agg(agg).reset_index()
    nseg = df.groupby(["domain", "source_id"], dropna=False).size().reset_index(name="n_feature_segments")
    out = out.merge(nseg, on=["domain", "source_id"], how="left")
    return out


def load_domain_features(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    print("[1/8] Loading domain-gap feature CSV...")
    df = pd.read_csv(args.domain_features_csv)

    required = {"domain", "source_type", "mode", "source_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Domain feature CSV missing required columns: {sorted(missing)}")

    mask = (
        df["domain"].isin([args.source_domain, args.target_domain])
        & (df["source_type"].astype(str) == args.source_type)
        & (df["mode"].astype(str) == args.mode)
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering domain/source_type/mode.")

    features = finite_numeric_features(df, args.feature_set)
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    src_seg = df[df["domain"] == args.source_domain].copy()
    tgt_seg = df[df["domain"] == args.target_domain].copy()
    if src_seg.empty or tgt_seg.empty:
        raise RuntimeError("Source or target domain rows are empty after filtering.")

    src_sources = src_seg["source_id"].nunique()
    tgt_sources = tgt_seg["source_id"].nunique()
    print(f"  rows after filter: {len(df)}")
    print(f"  source rows/sources: {len(src_seg)} / {src_sources}")
    print(f"  target rows/sources: {len(tgt_seg)} / {tgt_sources}")
    print(f"  features: {len(features)}")

    source_level = aggregate_to_source_level(df, features)
    return df, src_seg, tgt_seg, source_level, features


def zscore_with_reference(X: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (X - ref_mean) / np.maximum(ref_std, eps)


def compute_source_distances(source_level: pd.DataFrame, features: Sequence[str], args) -> pd.DataFrame:
    print("[2/8] Computing Torabi LS distance to EXP_H/ICBHI source cloud...")
    src = source_level[source_level["domain"] == args.source_domain].copy()
    tgt = source_level[source_level["domain"] == args.target_domain].copy()
    if src.empty or tgt.empty:
        raise RuntimeError("Cannot compute distances: missing source or target source-level rows.")

    X_src = src[list(features)].to_numpy(dtype=float)
    X_tgt = tgt[list(features)].to_numpy(dtype=float)

    ref_mean = np.nanmean(X_src, axis=0)
    ref_std = np.nanstd(X_src, axis=0, ddof=1)

    # Impute all missing values with EXP_H reference means before z-scoring.
    X_src_imp = np.where(np.isfinite(X_src), X_src, ref_mean)
    X_tgt_imp = np.where(np.isfinite(X_tgt), X_tgt, ref_mean)

    Z_src = zscore_with_reference(X_src_imp, ref_mean, ref_std)
    Z_tgt = zscore_with_reference(X_tgt_imp, ref_mean, ref_std)

    centroid = np.nanmean(Z_src, axis=0)
    d_centroid = np.linalg.norm(Z_tgt - centroid[None, :], axis=1)

    # Nearest-neighbour distances to the EXP_H/ICBHI source cloud.
    # Shape: target_sources x source_sources.
    diff = Z_tgt[:, None, :] - Z_src[None, :, :]
    pairwise = np.linalg.norm(diff, axis=2)
    d_nn1 = np.min(pairwise, axis=1)
    k = int(max(1, min(args.nn_k, pairwise.shape[1])))
    d_nn_k_mean = np.mean(np.sort(pairwise, axis=1)[:, :k], axis=1)

    # Regularized Mahalanobis in EXP_H z-space.
    # Use ridge to avoid singular covariance when features are correlated.
    cov = np.cov(Z_src, rowvar=False)
    p = cov.shape[0]
    ridge = float(args.mahalanobis_ridge)
    if ridge <= 0:
        ridge = 0.05
    scale = float(np.trace(cov) / max(p, 1)) if np.isfinite(np.trace(cov)) else 1.0
    cov_reg = cov + np.eye(p) * ridge * max(scale, 1e-8)
    try:
        inv_cov = np.linalg.pinv(cov_reg)
        centered = Z_tgt - centroid[None, :]
        d_maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inv_cov, centered), 0.0))
    except Exception:
        d_maha = np.full(Z_tgt.shape[0], np.nan)

    out_cols = ["domain", "source_id"]
    for c in ["class_name", "macro_class", "split", "n_feature_segments"]:
        if c in tgt.columns:
            out_cols.append(c)
    out = tgt[out_cols].copy()
    out["distance_z_centroid"] = d_centroid
    out["distance_z_nn1"] = d_nn1
    out[f"distance_z_nn{k}_mean"] = d_nn_k_mean
    # Standardize name used downstream, even if k != 5.
    out["distance_z_nn5_mean"] = d_nn_k_mean
    out["distance_mahalanobis_reg"] = d_maha

    # Add most shifted individual z-features for inspection.
    for i, f in enumerate(features):
        out[f"z_{f}"] = Z_tgt[:, i]

    return out


def parse_base_segment_from_name(text: str) -> Tuple[float, float]:
    """Parse strings like M_008160_s008_orig or 008160_s008_orig."""
    if not isinstance(text, str):
        return np.nan, np.nan
    m = re.search(r"(?P<base>\d+)_s(?P<seg>\d+)", text)
    if not m:
        return np.nan, np.nan
    return float(int(m.group("base"))), float(int(m.group("seg")))


def add_base_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "base_id" not in out.columns or out["base_id"].isna().all():
        text_col = "sample_name" if "sample_name" in out.columns else "sample_id" if "sample_id" in out.columns else None
        if text_col:
            parsed = out[text_col].apply(parse_base_segment_from_name)
            out["base_id"] = [p[0] for p in parsed]
            out["segment_index"] = [p[1] for p in parsed]
    if "segment_index" not in out.columns or out["segment_index"].isna().all():
        text_col = "sample_name" if "sample_name" in out.columns else "sample_id" if "sample_id" in out.columns else None
        if text_col:
            parsed = out[text_col].apply(parse_base_segment_from_name)
            if "base_id" not in out.columns:
                out["base_id"] = [p[0] for p in parsed]
            out["segment_index"] = [p[1] for p in parsed]
    for c in ["base_id", "segment_index"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def attach_ls_source_id(perf: pd.DataFrame, target_seg: pd.DataFrame, args) -> pd.DataFrame:
    """Attach ls_source_id to per-sample error metrics if it is not already present."""
    out = add_base_segment_columns(perf)

    if "ls_source_id" in out.columns and out["ls_source_id"].notna().any():
        print("  per-sample metrics already contain ls_source_id; using it.")
        return out

    required = {"base_id", "segment_index", "source_id"}
    if not required.issubset(target_seg.columns):
        raise ValueError(
            "Cannot infer ls_source_id because target segment features lack base_id/segment_index/source_id. "
            "Pass an error metrics CSV that already contains ls_source_id, or use a domain feature CSV with segment keys."
        )
    if "base_id" not in out.columns or "segment_index" not in out.columns:
        raise ValueError("Cannot infer base_id/segment_index from error metrics sample_id/sample_name.")

    map_cols = ["base_id", "segment_index", "source_id"]
    for c in ["class_name", "macro_class", "split"]:
        if c in target_seg.columns:
            map_cols.append(c)
    mapper = target_seg[map_cols].drop_duplicates(subset=["base_id", "segment_index"]).copy()
    mapper = mapper.rename(columns={
        "source_id": "ls_source_id",
        "class_name": "ls_class_name",
        "macro_class": "ls_macro_class",
        "split": "ls_split",
    })
    mapper["base_id"] = pd.to_numeric(mapper["base_id"], errors="coerce").astype("Int64")
    mapper["segment_index"] = pd.to_numeric(mapper["segment_index"], errors="coerce").astype("Int64")

    before = len(out)
    out = out.merge(mapper, on=["base_id", "segment_index"], how="left")
    matched = int(out["ls_source_id"].notna().sum())
    coverage = matched / max(before, 1)
    print(f"  attached ls_source_id by base_id+segment_index: {matched}/{before} rows ({coverage:.1%})")
    if coverage < args.min_join_coverage:
        warnings.warn(
            f"Low mapping coverage ({coverage:.1%}). Performance aggregation may be biased. "
            "Consider passing a full Torabi segment map or a per-sample metrics CSV with ls_source_id."
        )
    return out


def tag_rate(series: pd.Series, tag: str) -> float:
    if series.empty:
        return float("nan")
    return float(series.fillna("").astype(str).str.contains(tag, regex=False).mean())


def aggregate_performance_by_ls_source(perf: pd.DataFrame, args) -> pd.DataFrame:
    print("[3/8] Aggregating model performance by Torabi LS source...")
    if "ls_source_id" not in perf.columns:
        raise ValueError("ls_source_id not available after mapping.")
    df = perf[perf["ls_source_id"].notna()].copy()
    if df.empty:
        raise RuntimeError("No per-sample error rows have ls_source_id after mapping.")

    # Ensure numeric metrics are numeric.
    numeric_candidates = [
        "si_sdr_l", "si_sdr_h", "mean_si_sdr", "min_si_sdr",
        "sir_l", "sir_h", "sar_l", "sar_h",
        "mix_corr", "mix_nmse_db", "mix_gain_error_db",
        "target_local_snr_db", "hs_hat_corr_l_ref", "ls_hat_corr_h_ref",
        "gain_error_db_l", "gain_error_db_h",
        "possible_swap",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for sid, g in df.groupby("ls_source_id", dropna=False):
        row: Dict[str, float | str] = {"ls_source_id": sid, "n_eval_segments": int(len(g))}
        for meta in ["ls_class_name", "ls_macro_class", "ls_split"]:
            if meta in g.columns:
                row[meta] = first_non_null(g[meta])

        def mean_col(c: str) -> float:
            return float(np.nanmean(g[c])) if c in g.columns and g[c].notna().any() else float("nan")

        def median_col(c: str) -> float:
            return float(np.nanmedian(g[c])) if c in g.columns and g[c].notna().any() else float("nan")

        for c in ["si_sdr_l", "si_sdr_h", "mean_si_sdr", "min_si_sdr", "sir_l", "sir_h", "sar_l", "sar_h", "mix_corr", "mix_nmse_db", "mix_gain_error_db", "target_local_snr_db", "gain_error_db_l", "gain_error_db_h", "hs_hat_corr_l_ref", "ls_hat_corr_h_ref"]:
            if c in g.columns:
                row[f"{c}_mean"] = mean_col(c)
                row[f"{c}_median"] = median_col(c)

        if "target_local_snr_db" in g.columns:
            local = g["target_local_snr_db"].to_numpy(dtype=float)
            row["abs_local_snr_mean"] = float(np.nanmean(np.abs(local)))
            row["ls_very_weak_target_rate"] = float(np.nanmean(local >= 15.0))
            row["hs_very_weak_target_rate"] = float(np.nanmean(local <= -15.0))
            row["severe_imbalance_rate"] = float(np.nanmean(np.abs(local) >= 15.0))

        if "failure_tags" in g.columns:
            row["ls_low_sisdr_rate"] = tag_rate(g["failure_tags"], "LS_LOW_SISDR")
            row["hs_low_sisdr_rate"] = tag_rate(g["failure_tags"], "HS_LOW_SISDR")
            row["hs_leak_in_ls_rate"] = tag_rate(g["failure_tags"], "HS_LEAK_IN_LS_OUTPUT")
            row["ls_leak_in_hs_rate"] = tag_rate(g["failure_tags"], "LS_LEAK_IN_HS_OUTPUT")
            row["mix_inconsistent_rate"] = tag_rate(g["failure_tags"], "MIX_INCONSISTENT")
            row["possible_swap_rate"] = tag_rate(g["failure_tags"], "POSSIBLE_SWAP")
            row["ok_rate"] = tag_rate(g["failure_tags"], "OK_OR_AMBIGUOUS")
        elif "possible_swap" in g.columns:
            row["possible_swap_rate"] = mean_col("possible_swap")

        # Useful alternative failure rates directly from metric thresholds.
        if "si_sdr_l" in g.columns:
            row["ls_sisdr_below_0_rate"] = float(np.nanmean(g["si_sdr_l"].to_numpy(dtype=float) < 0.0))
            row["ls_sisdr_below_3_rate"] = float(np.nanmean(g["si_sdr_l"].to_numpy(dtype=float) < 3.0))
        if "si_sdr_h" in g.columns:
            row["hs_sisdr_below_0_rate"] = float(np.nanmean(g["si_sdr_h"].to_numpy(dtype=float) < 0.0))

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[out["n_eval_segments"] >= args.min_eval_segments_per_source].copy()
    print(f"  LS sources with >= {args.min_eval_segments_per_source} eval segments: {len(out)}")
    return out


def clean_xy(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def corr_one(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float, int]:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan"), int(len(x))
    if method == "pearson":
        if pearsonr is None:
            return float(np.corrcoef(x, y)[0, 1]), float("nan"), int(len(x))
        r, p = pearsonr(x, y)
        return float(r), float(p), int(len(x))
    if method == "spearman":
        if spearmanr is None:
            return float(pd.Series(x).corr(pd.Series(y), method="spearman")), float("nan"), int(len(x))
        r, p = spearmanr(x, y)
        return float(r), float(p), int(len(x))
    raise ValueError(method)


def compute_correlations(merged: pd.DataFrame, distance_cols: Sequence[str], performance_cols: Sequence[str]) -> pd.DataFrame:
    print("[5/8] Computing source-level correlations...")
    rows = []
    for dcol in distance_cols:
        if dcol not in merged.columns:
            continue
        for pcol in performance_cols:
            if pcol not in merged.columns:
                continue
            x, y = clean_xy(merged, dcol, pcol)
            pr, pp, n = corr_one(x, y, "pearson")
            sr, sp, _ = corr_one(x, y, "spearman")
            rows.append({
                "distance_metric": dcol,
                "performance_metric": pcol,
                "n_sources": n,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
                "abs_spearman_r": abs(sr) if np.isfinite(sr) else np.nan,
                "direction_hint": interpret_direction(pcol, sr),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_spearman_r", "distance_metric"], ascending=[False, True])
    return out


def interpret_direction(metric: str, spearman_r_value: float) -> str:
    if not np.isfinite(spearman_r_value):
        return "NA"
    lower_is_worse = any(k in metric for k in ["si_sdr", "sir", "sar", "mix_corr", "ok_rate"])
    higher_is_worse = any(k in metric for k in ["low_sisdr", "leak", "inconsistent", "swap", "abs_local_snr", "weak", "below"])
    if lower_is_worse and spearman_r_value < 0:
        return "farther_sources_perform_worse"
    if higher_is_worse and spearman_r_value > 0:
        return "farther_sources_fail_more"
    if lower_is_worse and spearman_r_value > 0:
        return "farther_sources_perform_better"
    if higher_is_worse and spearman_r_value < 0:
        return "farther_sources_fail_less"
    return "mixed_or_metric_specific"


def bootstrap_correlation_ci(
    merged: pd.DataFrame,
    corr_df: pd.DataFrame,
    args,
) -> pd.DataFrame:
    print("[6/8] Bootstrapping correlation confidence intervals...")
    if corr_df.empty:
        return corr_df
    rng = np.random.default_rng(args.seed)
    rows = []
    n = len(merged)
    for _, row in corr_df.iterrows():
        dcol = row["distance_metric"]
        pcol = row["performance_metric"]
        sub = merged[[dcol, pcol]].copy()
        sub[dcol] = pd.to_numeric(sub[dcol], errors="coerce")
        sub[pcol] = pd.to_numeric(sub[pcol], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        vals = []
        if len(sub) >= 5:
            arr = sub.to_numpy(dtype=float)
            for _ in range(args.n_boot):
                idx = rng.integers(0, len(arr), size=len(arr))
                x = arr[idx, 0]
                y = arr[idx, 1]
                r, _, _ = corr_one(x, y, "spearman")
                if np.isfinite(r):
                    vals.append(r)
        vals = np.asarray(vals, dtype=float)
        if len(vals) > 0:
            ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
            sign_stability = float(max(np.mean(vals > 0), np.mean(vals < 0)))
            ci_excludes_zero = bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
        else:
            ci_low = ci_high = sign_stability = np.nan
            ci_excludes_zero = False
        rec = row.to_dict()
        rec.update({
            "spearman_boot_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
            "spearman_boot_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
            "spearman_boot_sign_stability": sign_stability,
            "spearman_boot_ci_excludes_zero": ci_excludes_zero,
            "n_boot_valid": int(len(vals)),
        })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["abs_spearman_r", "distance_metric"], ascending=[False, True])


def make_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    x, y = clean_xy(df, x_col, y_col)
    if len(x) < 3:
        return
    sr, sp, n = corr_one(x, y, "spearman")
    pr, pp, _ = corr_one(x, y, "pearson")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(x, y, alpha=0.8)

    if len(x) >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        try:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ys = coef[0] * xs + coef[1]
            ax.plot(xs, ys, linewidth=2)
        except Exception:
            pass

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    txt = f"n={n}\nSpearman r={sr:.3f}, p={sp:.3g}\nPearson r={pr:.3f}, p={pp:.3g}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", bbox={"boxstyle": "round", "alpha": 0.15})
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plots(merged: pd.DataFrame, args, out_dir: Path) -> None:
    print("[7/8] Saving plots...")
    plot_specs = [
        ("distance_z_centroid", "si_sdr_l_mean", "07_distance_vs_ls_sisdr.png", "EXP_H distance vs LS SI-SDR"),
        ("distance_z_centroid", "mean_si_sdr_mean", "08_distance_vs_mean_sisdr.png", "EXP_H distance vs mean SI-SDR"),
        ("distance_z_centroid", "ls_low_sisdr_rate", "09_distance_vs_ls_low_sisdr_rate.png", "EXP_H distance vs LS low-SI-SDR rate"),
        ("distance_z_centroid", "abs_local_snr_mean", "10_distance_vs_abs_local_snr.png", "EXP_H distance vs local source imbalance"),
        ("distance_z_nn5_mean", "si_sdr_l_mean", "11_nn_distance_vs_ls_sisdr.png", "EXP_H 5-NN distance vs LS SI-SDR"),
    ]
    for x_col, y_col, fname, title in plot_specs:
        if x_col in merged.columns and y_col in merged.columns:
            make_scatter_plot(merged, x_col, y_col, out_dir / fname, title)


def write_readme(
    out_dir: Path,
    args,
    n_features: int,
    source_dist: pd.DataFrame,
    perf: pd.DataFrame,
    merged: pd.DataFrame,
    corr: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Domain-gap distance vs model performance\n")
    lines.append("## Scope\n")
    lines.append(f"- Source domain reference: `{args.source_domain}`\n")
    lines.append(f"- Target domain: `{args.target_domain}`\n")
    lines.append(f"- Source type: `{args.source_type}`\n")
    lines.append(f"- Feature mode: `{args.mode}`\n")
    lines.append(f"- Feature set: `{args.feature_set}` ({n_features} features)\n")
    lines.append(f"- Statistical unit: Torabi LS source (`source_id`)\n")
    lines.append("\n## Counts\n")
    lines.append(f"- Torabi LS sources with distance estimates: {len(source_dist)}\n")
    lines.append(f"- Torabi LS sources with performance summaries: {len(perf)}\n")
    lines.append(f"- Sources in merged analysis: {len(merged)}\n")
    lines.append("\n## Main interpretation\n")
    if corr.empty:
        lines.append("No valid correlations were computed. Check input coverage and metric names.\n")
    else:
        main = corr[(corr["distance_metric"] == "distance_z_centroid") & (corr["performance_metric"] == "si_sdr_l_mean")]
        if not main.empty:
            r = float(main.iloc[0]["spearman_r"])
            p = float(main.iloc[0]["spearman_p"])
            lo = main.iloc[0].get("spearman_boot_ci_low", np.nan)
            hi = main.iloc[0].get("spearman_boot_ci_high", np.nan)
            lines.append(f"- Main pair: `distance_z_centroid` vs `si_sdr_l_mean`: Spearman r = {r:.3f}, p = {p:.4g}, bootstrap CI = [{lo:.3f}, {hi:.3f}].\n")
            if np.isfinite(r) and r < 0:
                lines.append("- Negative r means that Torabi LS sources farther from the EXP_H/ICBHI reference cloud tend to have lower LS SI-SDR.\n")
            elif np.isfinite(r) and r > 0:
                lines.append("- Positive r means that farther sources do not show worse LS SI-SDR in this analysis. The domain gap may describe global mismatch without being predictive of source-wise performance.\n")
        lines.append("\n## Strongest source-level associations by absolute Spearman r\n")
        show_cols = ["distance_metric", "performance_metric", "n_sources", "spearman_r", "spearman_p", "spearman_boot_ci_low", "spearman_boot_ci_high", "direction_hint"]
        available = [c for c in show_cols if c in corr.columns]
        top = corr.head(10)[available]
        lines.append(top.to_markdown(index=False))
        lines.append("\n")
    lines.append("\n## Caveats\n")
    lines.append("- This is an observational diagnostic analysis, not a causal proof.\n")
    lines.append("- Distances are computed in RMS-normalized LS feature space, not directly on waveform-level neural representations.\n")
    lines.append("- A weak correlation does not invalidate the domain-gap analysis: it would mean that global acoustic mismatch and per-source model failure are only partially aligned, with local source observability remaining a separate bottleneck.\n")
    lines.append("- A strong negative correlation would support the claim that the measured domain gap is not only descriptive, but partially predictive of model degradation.\n")

    (out_dir / "README_distance_to_performance.md").write_text("".join(lines))


def save_input_counts(out_dir: Path, domain_seg: pd.DataFrame, perf_raw: pd.DataFrame, perf_mapped: pd.DataFrame, merged: pd.DataFrame, args) -> None:
    rows = []
    for (domain, source_type, mode), g in domain_seg.groupby(["domain", "source_type", "mode"], dropna=False):
        rows.append({
            "table": "domain_features",
            "domain": domain,
            "source_type": source_type,
            "mode": mode,
            "n_rows": len(g),
            "n_sources": g["source_id"].nunique() if "source_id" in g.columns else np.nan,
        })
    rows.append({
        "table": "error_metrics_raw",
        "domain": args.target_domain,
        "source_type": "M_eval",
        "mode": "model_output",
        "n_rows": len(perf_raw),
        "n_sources": np.nan,
    })
    rows.append({
        "table": "error_metrics_mapped_to_ls_source",
        "domain": args.target_domain,
        "source_type": "LS",
        "mode": "model_output",
        "n_rows": int(perf_mapped["ls_source_id"].notna().sum()) if "ls_source_id" in perf_mapped.columns else 0,
        "n_sources": perf_mapped["ls_source_id"].nunique() if "ls_source_id" in perf_mapped.columns else 0,
    })
    rows.append({
        "table": "merged_source_level",
        "domain": args.target_domain,
        "source_type": "LS",
        "mode": "distance_plus_performance",
        "n_rows": len(merged),
        "n_sources": merged["source_id"].nunique() if "source_id" in merged.columns else len(merged),
    })
    pd.DataFrame(rows).to_csv(out_dir / "00_input_counts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Link LS domain-gap distance to separator performance.")
    parser.add_argument("--domain-features-csv", type=Path, required=True,
                        help="segment_level_features_sampled.csv from domain-gap audit")
    parser.add_argument("--error-metrics-csv", type=Path, required=True,
                        help="per_sample_error_metrics.csv or final error-analysis output directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-domain", default="EXP_H_ALL")
    parser.add_argument("--target-domain", default="TORABI_FOLD2_ALL")
    parser.add_argument("--source-type", default="LS")
    parser.add_argument("--mode", default="rms")
    parser.add_argument("--feature-set", choices=["key", "auto"], default="key")
    parser.add_argument("--nn-k", type=int, default=5)
    parser.add_argument("--mahalanobis-ridge", type=float, default=0.05)
    parser.add_argument("--min-eval-segments-per-source", type=int, default=20)
    parser.add_argument("--min-join-coverage", type=float, default=0.50,
                        help="warn if less than this fraction of error rows maps to LS source")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    with open(args.out_dir / "00_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    domain_seg, src_seg, tgt_seg, source_level, features = load_domain_features(args)
    source_dist = compute_source_distances(source_level, features, args)

    print("[3/8] Loading final error-analysis per-sample metrics...")
    error_path = resolve_error_metrics_path(args.error_metrics_csv)
    print(f"  using error metrics: {error_path}")
    perf_raw = pd.read_csv(error_path)
    perf_mapped = attach_ls_source_id(perf_raw, tgt_seg, args)
    perf_source = aggregate_performance_by_ls_source(perf_mapped, args)

    print("[4/8] Merging source distance and source performance...")
    merged = source_dist.merge(perf_source, left_on="source_id", right_on="ls_source_id", how="inner")
    if merged.empty:
        raise RuntimeError("Merged distance/performance table is empty. Check LS source mapping.")
    print(f"  merged LS sources: {len(merged)}")

    distance_cols = [c for c in DEFAULT_DISTANCE_COLS if c in merged.columns]
    performance_cols = [c for c in DEFAULT_PERFORMANCE_METRICS if c in merged.columns]
    # Add threshold-derived metrics if present.
    for c in ["ls_sisdr_below_0_rate", "ls_sisdr_below_3_rate", "severe_imbalance_rate", "ls_very_weak_target_rate", "hs_very_weak_target_rate"]:
        if c in merged.columns and c not in performance_cols:
            performance_cols.append(c)

    corr = compute_correlations(merged, distance_cols, performance_cols)
    corr_boot = bootstrap_correlation_ci(merged, corr, args)

    # Hardest/farthest source table.
    hard_sort_cols = [c for c in ["distance_z_centroid", "si_sdr_l_mean"] if c in merged.columns]
    if "distance_z_centroid" in merged.columns:
        hardest = merged.sort_values(["distance_z_centroid"], ascending=False).head(20).copy()
    else:
        hardest = merged.copy().head(20)

    # Save CSV outputs.
    print("[8/8] Saving outputs...")
    save_input_counts(args.out_dir, domain_seg, perf_raw, perf_mapped, merged, args)
    source_dist.to_csv(args.out_dir / "01_torabi_ls_source_distance_to_exph.csv", index=False)
    perf_source.to_csv(args.out_dir / "02_torabi_ls_source_performance_summary.csv", index=False)
    merged.to_csv(args.out_dir / "03_distance_performance_merged.csv", index=False)
    corr.to_csv(args.out_dir / "04_distance_performance_correlations.csv", index=False)
    corr_boot.to_csv(args.out_dir / "05_distance_performance_correlations_bootstrap_ci.csv", index=False)
    hardest.to_csv(args.out_dir / "06_top_farthest_torabi_ls_sources.csv", index=False)

    save_plots(merged, args, args.out_dir)
    write_readme(args.out_dir, args, len(features), source_dist, perf_source, merged, corr_boot)

    print("\nDONE")
    print(f"Output directory: {args.out_dir}")
    if not corr_boot.empty:
        main_pair = corr_boot[(corr_boot["distance_metric"] == "distance_z_centroid") & (corr_boot["performance_metric"] == "si_sdr_l_mean")]
        if not main_pair.empty:
            r = main_pair.iloc[0]["spearman_r"]
            p = main_pair.iloc[0]["spearman_p"]
            lo = main_pair.iloc[0].get("spearman_boot_ci_low", np.nan)
            hi = main_pair.iloc[0].get("spearman_boot_ci_high", np.nan)
            print(f"Main pair distance_z_centroid vs si_sdr_l_mean: Spearman r={r:.3f}, p={p:.4g}, CI=[{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()

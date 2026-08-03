#!/usr/bin/env python3
"""
02_bootstrap_ci_error_analysis.py
=================================

Bootstrap confidence intervals for the final SSL error analysis.

Input expected for each run:
    <ERROR_ROOT>/<run_name>/metrics/per_sample_error_metrics.csv

It computes:
  1. global metric CIs;
  2. per-SNR metric CIs;
  3. local-SNR-bin metric CIs;
  4. failure-tag rate CIs;
  5. paired bootstrap deltas vs SSL baseline.

Bootstrap unit:
  base_id, not individual 2 s segments.

Reason:
  each base triplet generates multiple overlapping 2 s windows, so segment-level
  bootstrap would overestimate independence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Runs to compare
# ---------------------------------------------------------------------

RUN_DIRS = {
    "SSL_baseline": "final_ssl_fold2",
    "T10_W15": "frozen_ablation_ssl_weighted_source_fold2_T10_W15",
    "T10_W2": "frozen_ablation_ssl_weighted_source_fold2_T10_W2",
    "T8_W15": "frozen_ablation_ssl_weighted_source_fold2_T8_W15",
    "T8_W2": "frozen_ablation_ssl_weighted_source_fold2_T8_W2",
}

BASELINE_NAME = "SSL_baseline"

METRICS = [
    "si_sdr_h",
    "si_sdr_l",
    "mean_si_sdr",
    "sir_h",
    "sir_l",
    "mix_corr",
    "mix_nmse_db",
    "mix_gain_error_db",
    "gain_error_db_h",
    "gain_error_db_l",
    "possible_swap",
]

FAILURE_TAGS = [
    "HS_LOW_SISDR",
    "LS_LOW_SISDR",
    "LS_LEAK_IN_HS_OUTPUT",
    "HS_LEAK_IN_LS_OUTPUT",
    "MIX_INCONSISTENT",
    "POSSIBLE_SWAP",
    "HS_GAIN_ERROR",
    "LS_GAIN_ERROR",
    "OK_OR_AMBIGUOUS",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_run_csv(error_root: Path, run_label: str, run_dir: str) -> pd.DataFrame:
    path = error_root / run_dir / "metrics" / "per_sample_error_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file for {run_label}: {path}")

    df = pd.read_csv(path)
    df = df.copy()

    required = ["sample_id", "base_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")

    df["run"] = run_label
    df["base_id"] = df["base_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)

    return df


def numeric_array(df: pd.DataFrame, metric: str) -> np.ndarray:
    return pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)


def bootstrap_mean_ci(
    df: pd.DataFrame,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    if metric not in df.columns:
        return {}

    df = df[["base_id", metric]].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    if df.empty:
        return {}

    base_ids = sorted(df["base_id"].unique())
    if len(base_ids) < 2:
        return {}

    values_by_base = {
        b: df.loc[df["base_id"] == b, metric].to_numpy(dtype=float)
        for b in base_ids
    }

    obs = float(df[metric].mean())
    boot_values = []

    for _ in range(n_boot):
        sampled_bases = rng.choice(base_ids, size=len(base_ids), replace=True)
        total_sum = 0.0
        total_n = 0

        for b in sampled_bases:
            vals = values_by_base[b]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            total_sum += float(vals.sum())
            total_n += int(len(vals))

        if total_n > 0:
            boot_values.append(total_sum / total_n)

    boot_values = np.asarray(boot_values, dtype=float)

    return {
        "metric": metric,
        "n_rows": int(len(df)),
        "n_base": int(len(base_ids)),
        "mean": obs,
        "ci_low": float(np.percentile(boot_values, 2.5)),
        "ci_high": float(np.percentile(boot_values, 97.5)),
        "boot_std": float(np.std(boot_values, ddof=1)),
    }


def tag_mask(series: pd.Series, tag: str) -> pd.Series:
    pattern = rf"(^|;){re.escape(tag)}($|;)"
    return series.astype(str).str.contains(pattern, regex=True)


def bootstrap_tag_rate_ci(
    df: pd.DataFrame,
    tag: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    if "failure_tags" not in df.columns:
        return {}

    temp = df[["base_id", "failure_tags"]].copy()
    temp["has_tag"] = tag_mask(temp["failure_tags"], tag).astype(float)

    base_ids = sorted(temp["base_id"].unique())
    if len(base_ids) < 2:
        return {}

    values_by_base = {
        b: temp.loc[temp["base_id"] == b, "has_tag"].to_numpy(dtype=float)
        for b in base_ids
    }

    obs = float(temp["has_tag"].mean())
    boot_values = []

    for _ in range(n_boot):
        sampled_bases = rng.choice(base_ids, size=len(base_ids), replace=True)
        total_sum = 0.0
        total_n = 0

        for b in sampled_bases:
            vals = values_by_base[b]
            total_sum += float(vals.sum())
            total_n += int(len(vals))

        if total_n > 0:
            boot_values.append(total_sum / total_n)

    boot_values = np.asarray(boot_values, dtype=float)

    return {
        "failure_tag": tag,
        "n_rows": int(len(temp)),
        "n_base": int(len(base_ids)),
        "rate": obs,
        "percentage": obs * 100.0,
        "ci_low": float(np.percentile(boot_values, 2.5)),
        "ci_high": float(np.percentile(boot_values, 97.5)),
        "ci_low_pct": float(np.percentile(boot_values, 2.5) * 100.0),
        "ci_high_pct": float(np.percentile(boot_values, 97.5) * 100.0),
        "boot_std": float(np.std(boot_values, ddof=1)),
    }


def bootstrap_paired_delta_ci(
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    if metric not in baseline_df.columns or metric not in run_df.columns:
        return {}

    b = baseline_df[["sample_id", "base_id", metric]].copy()
    r = run_df[["sample_id", metric]].copy()

    b[metric] = pd.to_numeric(b[metric], errors="coerce")
    r[metric] = pd.to_numeric(r[metric], errors="coerce")

    merged = b.merge(r, on="sample_id", suffixes=("_baseline", "_run"))
    merged = merged.dropna(subset=[f"{metric}_baseline", f"{metric}_run"])

    if merged.empty:
        return {}

    merged["delta"] = merged[f"{metric}_run"] - merged[f"{metric}_baseline"]

    base_ids = sorted(merged["base_id"].astype(str).unique())
    if len(base_ids) < 2:
        return {}

    values_by_base = {
        b_id: merged.loc[merged["base_id"].astype(str) == b_id, "delta"].to_numpy(dtype=float)
        for b_id in base_ids
    }

    obs = float(merged["delta"].mean())
    boot_values = []

    for _ in range(n_boot):
        sampled_bases = rng.choice(base_ids, size=len(base_ids), replace=True)
        total_sum = 0.0
        total_n = 0

        for b_id in sampled_bases:
            vals = values_by_base[b_id]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            total_sum += float(vals.sum())
            total_n += int(len(vals))

        if total_n > 0:
            boot_values.append(total_sum / total_n)

    boot_values = np.asarray(boot_values, dtype=float)

    return {
        "metric": metric,
        "n_rows": int(len(merged)),
        "n_base": int(len(base_ids)),
        "delta_mean": obs,
        "delta_ci_low": float(np.percentile(boot_values, 2.5)),
        "delta_ci_high": float(np.percentile(boot_values, 97.5)),
        "boot_std": float(np.std(boot_values, ddof=1)),
    }


def compute_metric_ci_table(
    runs: dict[str, pd.DataFrame],
    n_boot: int,
    seed: int,
    group_col: str | None = None,
) -> pd.DataFrame:
    rows = []

    for run_name, df in runs.items():
        if group_col is None:
            groups = [(None, df)]
        else:
            if group_col not in df.columns:
                continue
            groups = [(g, sub.copy()) for g, sub in df.groupby(group_col, dropna=False)]

        for group_value, sub in groups:
            for metric in METRICS:
                rng = np.random.default_rng(seed)
                out = bootstrap_mean_ci(sub, metric, n_boot=n_boot, rng=rng)
                if not out:
                    continue

                out["run"] = run_name
                if group_col is not None:
                    out["group_col"] = group_col
                    out["group_value"] = str(group_value)
                else:
                    out["group_col"] = "global"
                    out["group_value"] = "all"

                rows.append(out)

    return pd.DataFrame(rows)


def compute_failure_tag_ci_table(
    runs: dict[str, pd.DataFrame],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for run_name, df in runs.items():
        for tag in FAILURE_TAGS:
            rng = np.random.default_rng(seed)
            out = bootstrap_tag_rate_ci(df, tag, n_boot=n_boot, rng=rng)
            if not out:
                continue
            out["run"] = run_name
            rows.append(out)

    return pd.DataFrame(rows)


def compute_delta_table(
    runs: dict[str, pd.DataFrame],
    baseline_name: str,
    n_boot: int,
    seed: int,
    group_col: str | None = None,
) -> pd.DataFrame:
    rows = []

    baseline_df = runs[baseline_name]

    for run_name, run_df in runs.items():
        if run_name == baseline_name:
            continue

        if group_col is None:
            pairs = [("all", baseline_df, run_df)]
        else:
            if group_col not in baseline_df.columns or group_col not in run_df.columns:
                continue

            pairs = []
            common_groups = sorted(
                set(baseline_df[group_col].astype(str).unique())
                & set(run_df[group_col].astype(str).unique())
            )

            for g in common_groups:
                b_sub = baseline_df[baseline_df[group_col].astype(str) == g].copy()
                r_sub = run_df[run_df[group_col].astype(str) == g].copy()
                pairs.append((g, b_sub, r_sub))

        for group_value, b_sub, r_sub in pairs:
            for metric in METRICS:
                rng = np.random.default_rng(seed)
                out = bootstrap_paired_delta_ci(
                    b_sub,
                    r_sub,
                    metric,
                    n_boot=n_boot,
                    rng=rng,
                )
                if not out:
                    continue

                out["run"] = run_name
                out["baseline"] = baseline_name

                if group_col is None:
                    out["group_col"] = "global"
                    out["group_value"] = "all"
                else:
                    out["group_col"] = group_col
                    out["group_value"] = str(group_value)

                rows.append(out)

    return pd.DataFrame(rows)


def write_readme(out_dir: Path, n_boot: int) -> None:
    text = f"""# Bootstrap Confidence Intervals

Bootstrap configuration:
- resampling unit: base_id
- bootstrap iterations: {n_boot}
- CI: percentile 95% interval [2.5%, 97.5%]

Files:
- bootstrap_global_metrics.csv
- bootstrap_by_snr.csv
- bootstrap_by_local_snr_bin.csv
- bootstrap_failure_tags.csv
- paired_delta_vs_baseline_global.csv
- paired_delta_vs_baseline_by_snr.csv
- paired_delta_vs_baseline_by_local_snr_bin.csv

How to read paired deltas:
- delta = run - SSL_baseline
- if delta CI is entirely above 0: run improves over baseline
- if delta CI crosses 0: no robust difference
- if delta CI is entirely below 0: run is worse than baseline

Important:
Very small bins, especially HS_very_weak and LS_very_weak, must be interpreted cautiously.
Their confidence intervals can be wide because there are few independent examples.
"""
    (out_dir / "README_bootstrap_ci.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error-root",
        type=str,
        default="/nas/home/pcallandrone/DeepLearning/outputs/results/ERROR_ANALYSIS",
        help="Root folder containing the error-analysis result directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/nas/home/pcallandrone/DeepLearning/outputs/results/ERROR_ANALYSIS/bootstrap_ci_final_ssl",
        help="Output folder for bootstrap CSVs.",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    error_root = Path(args.error_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from: {error_root}")

    runs = {}
    for run_label, run_dir in RUN_DIRS.items():
        try:
            df = load_run_csv(error_root, run_label, run_dir)
            runs[run_label] = df
            print(f"[OK] {run_label}: {len(df)} rows, {df['base_id'].nunique()} base_id")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    if BASELINE_NAME not in runs:
        raise RuntimeError(f"Baseline run not found: {BASELINE_NAME}")

    print("Computing global metric CIs...")
    global_ci = compute_metric_ci_table(
        runs,
        n_boot=args.n_boot,
        seed=args.seed,
        group_col=None,
    )
    global_ci.to_csv(out_dir / "bootstrap_global_metrics.csv", index=False)

    print("Computing per-SNR metric CIs...")
    snr_ci = compute_metric_ci_table(
        runs,
        n_boot=args.n_boot,
        seed=args.seed,
        group_col="snr_label",
    )
    snr_ci.to_csv(out_dir / "bootstrap_by_snr.csv", index=False)

    print("Computing local-SNR-bin metric CIs...")
    local_ci = compute_metric_ci_table(
        runs,
        n_boot=args.n_boot,
        seed=args.seed,
        group_col="local_snr_bin",
    )
    local_ci.to_csv(out_dir / "bootstrap_by_local_snr_bin.csv", index=False)

    print("Computing failure-tag rate CIs...")
    tag_ci = compute_failure_tag_ci_table(
        runs,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    tag_ci.to_csv(out_dir / "bootstrap_failure_tags.csv", index=False)

    print("Computing paired deltas vs SSL baseline...")
    delta_global = compute_delta_table(
        runs,
        baseline_name=BASELINE_NAME,
        n_boot=args.n_boot,
        seed=args.seed,
        group_col=None,
    )
    delta_global.to_csv(out_dir / "paired_delta_vs_baseline_global.csv", index=False)

    delta_snr = compute_delta_table(
        runs,
        baseline_name=BASELINE_NAME,
        n_boot=args.n_boot,
        seed=args.seed,
        group_col="snr_label",
    )
    delta_snr.to_csv(out_dir / "paired_delta_vs_baseline_by_snr.csv", index=False)

    delta_local = compute_delta_table(
        runs,
        baseline_name=BASELINE_NAME,
        n_boot=args.n_boot,
        seed=args.seed,
        group_col="local_snr_bin",
    )
    delta_local.to_csv(out_dir / "paired_delta_vs_baseline_by_local_snr_bin.csv", index=False)

    write_readme(out_dir, args.n_boot)

    print("\nDone.")
    print(f"Output written to: {out_dir}")


if __name__ == "__main__":
    main()

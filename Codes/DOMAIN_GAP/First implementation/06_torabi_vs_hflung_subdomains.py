#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_torabi_vs_hflung_subdomains.py

Goal:
Compare Torabi Fold2 LS segment-level RMS features against:
  1) EXP_H / ICBHI selected LS reference
  2) HF_Lung top50 closest to EXP_H/ICBHI
  3) HF_Lung very close p95=1
  4) HF_Lung close
  5) HF_Lung partial
  6) HF_Lung far
  7) HF_Lung partial+far

This uses already extracted CSVs:
  - Torabi/EXP_H domain gap audit: segment_level_features_sampled.csv
  - HF_Lung trend audit: selected_icbhi_reference_segment_features.csv
  - HF_Lung trend audit: hflung_segment_features_with_similarity.csv
  - HF_Lung trend audit: hflung_files_ranked_by_similarity_to_selected_icbhi.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_TORABI_GAP_DIR = "/nas/home/pcallandrone/DeepLearning/outputs/domain_gap/domain_gap_EXP_H_vs_TORABI_FOLD2"
DEFAULT_HFLUNG_AUDIT_DIR = "/nas/home/pcallandrone/DeepLearning/outputs/domain_gap/HFLUNG_trend_vs_SELECTED_ICBHI_SMOKE"
DEFAULT_OUT_DIR = "/nas/home/pcallandrone/DeepLearning/outputs/domain_gap/TORABI_vs_HFLUNG_CLOSE_SUBDOMAINS"


FEATURES = [
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "crest_factor",
    "energy_20_50_ratio",
    "energy_20_80_ratio",
    "heart_band_20_500_ratio",
    "lung_band_50_1800_ratio",
    "ls_band_50_150_ratio",
    "ls_band_150_500_ratio",
    "ls_band_500_1000_ratio",
    "ls_band_1000_1800_ratio",
    "hf_noise_ratio_1000_2000",
]


IMPORTANT_FEATURES = [
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "crest_factor",
    "energy_20_80_ratio",
    "energy_20_50_ratio",
    "lung_band_50_1800_ratio",
    "ls_band_50_150_ratio",
    "ls_band_150_500_ratio",
    "ls_band_500_1000_ratio",
]


def load_torabi_ls_rms(torabi_gap_dir: Path) -> pd.DataFrame:
    p = torabi_gap_dir / "segment_level_features_sampled.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    df = pd.read_csv(p)

    needed = {"domain", "source_type", "mode"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {p}: {missing}")

    df = df[
        (df["source_type"] == "LS")
        & (df["mode"] == "rms")
        & (df["domain"] == "TORABI_FOLD2")
    ].copy()

    if df.empty:
        raise RuntimeError("No TORABI_FOLD2 LS/rms rows found.")

    return df


def load_exph_from_torabi_gap(torabi_gap_dir: Path) -> pd.DataFrame:
    p = torabi_gap_dir / "segment_level_features_sampled.csv"
    df = pd.read_csv(p)

    df = df[
        (df["source_type"] == "LS")
        & (df["mode"] == "rms")
        & (df["domain"] == "EXP_H")
    ].copy()

    if df.empty:
        raise RuntimeError("No EXP_H LS/rms rows found in Torabi gap file.")

    return df


def compute_smd(reference_df: pd.DataFrame, target_df: pd.DataFrame, reference_name: str, target_name: str, features):
    rows = []

    for feat in features:
        x = reference_df[feat].dropna().values
        y = target_df[feat].dropna().values

        mx = float(np.mean(x))
        my = float(np.mean(y))
        sx = float(np.std(x, ddof=1))
        sy = float(np.std(y, ddof=1))

        pooled = np.sqrt((sx * sx + sy * sy) / 2.0) + 1e-12
        smd = (my - mx) / pooled

        rows.append({
            "reference_group": reference_name,
            "target_group": target_name,
            "feature": feat,
            "reference_n_segments": len(reference_df),
            "target_n_segments": len(target_df),
            "reference_mean": mx,
            "target_mean": my,
            "reference_std": sx,
            "target_std": sy,
            "diff_target_minus_reference": my - mx,
            "smd_target_minus_reference": smd,
            "abs_smd": abs(smd),
        })

    return pd.DataFrame(rows)


def make_group_summary(feature_smd_df: pd.DataFrame):
    return (
        feature_smd_df
        .groupby(["reference_group", "target_group"], as_index=False)
        .agg(
            reference_n_segments=("reference_n_segments", "max"),
            target_n_segments=("target_n_segments", "max"),
            mean_abs_smd=("abs_smd", "mean"),
            median_abs_smd=("abs_smd", "median"),
            max_abs_smd=("abs_smd", "max"),
        )
        .sort_values("mean_abs_smd", ascending=False)
    )


def make_barplot(summary_df: pd.DataFrame, out_path: Path):
    df = summary_df.copy()
    df["comparison"] = df["target_group"] + " vs " + df["reference_group"]
    df = df.sort_values("mean_abs_smd", ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(df["comparison"], df["mean_abs_smd"])
    plt.xlabel("Mean |SMD|")
    plt.title("Torabi Fold2 LS gap vs real-patient subdomains")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--torabi_gap_dir", type=str, default=DEFAULT_TORABI_GAP_DIR)
    parser.add_argument("--hflung_audit_dir", type=str, default=DEFAULT_HFLUNG_AUDIT_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torabi_gap_dir = Path(args.torabi_gap_dir)
    hflung_audit_dir = Path(args.hflung_audit_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Torabi Fold2 vs EXP_H/ICBHI and HF_Lung close subdomains")
    print("=" * 90)
    print(f"Torabi gap dir    : {torabi_gap_dir}")
    print(f"HF_Lung audit dir : {hflung_audit_dir}")
    print(f"Output dir        : {out_dir}")
    print("=" * 90)

    # ------------------------------------------------------------
    # Load Torabi and EXP_H from previous domain-gap audit
    # ------------------------------------------------------------
    torabi_df = load_torabi_ls_rms(torabi_gap_dir)
    exph_gap_df = load_exph_from_torabi_gap(torabi_gap_dir)

    # ------------------------------------------------------------
    # Load HF_Lung trend audit outputs
    # ------------------------------------------------------------
    ref_icbhi_path = hflung_audit_dir / "selected_icbhi_reference_segment_features.csv"
    hflung_seg_path = hflung_audit_dir / "hflung_segment_features_with_similarity.csv"
    rank_path = hflung_audit_dir / "hflung_files_ranked_by_similarity_to_selected_icbhi.csv"

    ref_icbhi_df = pd.read_csv(ref_icbhi_path)
    hflung_seg_df = pd.read_csv(hflung_seg_path)
    rank_df = pd.read_csv(rank_path)

    # Use common available features only
    features = [
        f for f in FEATURES
        if f in torabi_df.columns
        and f in exph_gap_df.columns
        and f in ref_icbhi_df.columns
        and f in hflung_seg_df.columns
    ]

    if not features:
        raise RuntimeError("No common features found.")

    print(f"\nFeatures used: {len(features)}")
    print(features)

    # ------------------------------------------------------------
    # Build HF_Lung file groups
    # ------------------------------------------------------------
    files_top_n = set(rank_df.head(args.top_n)["source_file"])

    files_very_close = set(
        rank_df[rank_df["frac_inside_selected_icbhi_p95"] >= 0.999999]["source_file"]
    )

    files_close = set(
        rank_df[rank_df["similarity_label"] == "close_to_selected_ICBHI"]["source_file"]
    )

    files_partial = set(
        rank_df[rank_df["similarity_label"] == "partial_overlap"]["source_file"]
    )

    files_far = set(
        rank_df[rank_df["similarity_label"] == "far_from_selected_ICBHI"]["source_file"]
    )

    files_partial_far = files_partial | files_far

    hflung_groups = {
        f"HF_Lung_top{args.top_n}": files_top_n,
        "HF_Lung_very_close_p95_1": files_very_close,
        "HF_Lung_close": files_close,
        "HF_Lung_partial": files_partial,
        "HF_Lung_far": files_far,
        "HF_Lung_partial_plus_far": files_partial_far,
        "HF_Lung_all_sampled": set(rank_df["source_file"]),
    }

    group_dfs = {}
    for group_name, files in hflung_groups.items():
        group_dfs[group_name] = hflung_seg_df[hflung_seg_df["source_file"].isin(files)].copy()

    # ------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------
    all_smd = []

    # 1) Reconfirm classic gap: Torabi vs EXP_H from previous audit
    all_smd.append(
        compute_smd(
            reference_df=exph_gap_df,
            target_df=torabi_df,
            reference_name="EXP_H_ICBHI_selected_from_gap_audit",
            target_name="TORABI_FOLD2",
            features=features,
        )
    )

    # 2) Torabi vs selected ICBHI reference from HF_Lung audit
    all_smd.append(
        compute_smd(
            reference_df=ref_icbhi_df,
            target_df=torabi_df,
            reference_name="EXP_H_ICBHI_selected_from_HFLung_audit",
            target_name="TORABI_FOLD2",
            features=features,
        )
    )

    # 3) Torabi vs HF_Lung subgroups
    for group_name, df in group_dfs.items():
        if df.empty:
            print(f"[WARN] Empty group skipped: {group_name}")
            continue

        all_smd.append(
            compute_smd(
                reference_df=df,
                target_df=torabi_df,
                reference_name=group_name,
                target_name="TORABI_FOLD2",
                features=features,
            )
        )

    feature_smd_df = pd.concat(all_smd, ignore_index=True)
    summary_df = make_group_summary(feature_smd_df)

    # Compact table for important features
    important = [f for f in IMPORTANT_FEATURES if f in features]
    compact_df = feature_smd_df[feature_smd_df["feature"].isin(important)].copy()

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    feature_smd_df.to_csv(out_dir / "torabi_vs_realpatient_subdomains_feature_smd.csv", index=False)
    summary_df.to_csv(out_dir / "torabi_vs_realpatient_subdomains_summary.csv", index=False)
    compact_df.to_csv(out_dir / "torabi_vs_realpatient_subdomains_important_features.csv", index=False)

    make_barplot(summary_df, out_dir / "torabi_vs_realpatient_subdomains_mean_abs_smd.png")

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("IMPORTANT FEATURES")
    print("=" * 90)

    show_cols = [
        "reference_group",
        "target_group",
        "feature",
        "reference_mean",
        "target_mean",
        "smd_target_minus_reference",
        "abs_smd",
    ]

    for ref_name in summary_df["reference_group"].tolist():
        sub = compact_df[compact_df["reference_group"] == ref_name].copy()
        sub = sub.sort_values("abs_smd", ascending=False)
        print("\n" + "-" * 90)
        print(f"TORABI_FOLD2 vs {ref_name}")
        print("-" * 90)
        print(sub[show_cols].to_string(index=False))

    # Text summary
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Torabi Fold2 vs EXP_H/ICBHI and HF_Lung close subdomains\n")
        f.write("=" * 80 + "\n\n")

        f.write("Purpose\n")
        f.write(
            "Check whether Torabi Fold2 remains distant not only from EXP_H/ICBHI, "
            "but also from the HF_Lung subset that is acoustically close to EXP_H/ICBHI.\n\n"
        )

        f.write("Group-level mean |SMD|\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation guide\n")
        f.write("- High Torabi vs EXP_H/ICBHI and high Torabi vs HF_Lung very close: manikin gap is robust.\n")
        f.write("- High Torabi vs EXP_H/ICBHI but low Torabi vs HF_Lung very close: gap is more dataset-specific.\n")
        f.write("- Increasing gap from HF_Lung very close to partial/far indicates a real-patient acoustic gradient.\n")

    print("\nSaved:")
    print(out_dir / "torabi_vs_realpatient_subdomains_summary.csv")
    print(out_dir / "torabi_vs_realpatient_subdomains_feature_smd.csv")
    print(out_dir / "torabi_vs_realpatient_subdomains_important_features.csv")
    print(out_dir / "torabi_vs_realpatient_subdomains_mean_abs_smd.png")
    print(out_dir / "summary.txt")


if __name__ == "__main__":
    main()

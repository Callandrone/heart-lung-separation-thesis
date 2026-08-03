#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_domain_gap_train_vs_all_Stability_check.py

Compare domain-gap audit outputs between train-only and train+val ALL runs.
Focuses on LS segment-level RMS key features by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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


def read_gap(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path / "segment_level_gap_summary.csv")
    df = df[(df["source_type"].eq("LS")) & (df["mode"].eq("rms"))].copy()
    keep = ["feature", "source_n", "target_n", "source_mean", "target_mean", "smd_target_minus_source", "abs_smd"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={
        "source_n": f"source_n_{label}",
        "target_n": f"target_n_{label}",
        "source_mean": f"source_mean_{label}",
        "target_mean": f"target_mean_{label}",
        "smd_target_minus_source": f"smd_{label}",
        "abs_smd": f"abs_smd_{label}",
    })
    return df


def stability_label(row: pd.Series, smd_tol: float, sign_flip_tol: float) -> str:
    a = row["smd_train"]
    b = row["smd_all"]
    if pd.isna(a) or pd.isna(b):
        return "missing"
    if abs(a) >= sign_flip_tol and abs(b) >= sign_flip_tol and np.sign(a) != np.sign(b):
        return "sign_flip"
    if abs(abs(b) - abs(a)) <= smd_tol:
        return "stable"
    if abs(b) < abs(a):
        return "weaker_in_all"
    return "stronger_in_all"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--all-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--smd-tol", type=float, default=0.20)
    p.add_argument("--sign-flip-tol", type=float, default=0.30)
    p.add_argument("--only-key-features", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = read_gap(args.train_dir, "train")
    all_ = read_gap(args.all_dir, "all")
    comp = train.merge(all_, on="feature", how="outer")
    comp["delta_smd_all_minus_train"] = comp["smd_all"] - comp["smd_train"]
    comp["delta_abs_smd_all_minus_train"] = comp["abs_smd_all"] - comp["abs_smd_train"]
    comp["stability"] = comp.apply(lambda r: stability_label(r, args.smd_tol, args.sign_flip_tol), axis=1)
    comp = comp.sort_values("abs_smd_all", ascending=False).reset_index(drop=True)
    comp.to_csv(args.out_dir / "train_vs_all_LS_segment_RMS_comparison.csv", index=False)

    key = comp[comp["feature"].isin(KEY_FEATURES)].copy()
    key.to_csv(args.out_dir / "train_vs_all_LS_segment_RMS_key_features.csv", index=False)

    lines = []
    lines.append("# Train-only vs ALL comparison: LS segment-level RMS")
    lines.append("")
    lines.append("## Key features")
    show = key[["feature", "smd_train", "smd_all", "delta_smd_all_minus_train", "stability", "source_mean_all", "target_mean_all"]].copy()
    lines.append(show.to_markdown(index=False))
    lines.append("")
    lines.append("## Stability counts")
    lines.append(comp["stability"].value_counts(dropna=False).to_markdown())
    (args.out_dir / "README_train_vs_all.md").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", args.out_dir)
    print(" - train_vs_all_LS_segment_RMS_comparison.csv")
    print(" - train_vs_all_LS_segment_RMS_key_features.csv")
    print(" - README_train_vs_all.md")


if __name__ == "__main__":
    main()

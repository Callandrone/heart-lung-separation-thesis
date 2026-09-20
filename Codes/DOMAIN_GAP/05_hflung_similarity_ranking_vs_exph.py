#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_hflung_similarity_ranking_vs_exph.py

Question:
    Among all HF_Lung recordings, are there files / subsets that acoustically
    resemble the selected ICBHI / EXP_H lung distribution?

This is NOT training.
This is NOT augmentation.
This does NOT reconstruct mixtures.

It uses EXP_H processed L_*.wav segments as the selected ICBHI reference cloud,
then ranks HF_Lung recordings by similarity to that reference cloud.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_AUDIT_PATH = HERE / "04_hflung_global_vs_exph_baseline.py"
DEFAULT_EXPH_DIR = REPO_ROOT / "dataset" / "processed" / "experiment_H_full_both"
DEFAULT_HFLUNG_DIR = REPO_ROOT / "dataset" / "raw" / "HF_Lung_V1"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "domain_gap"
    / "HFLUNG_trend_vs_SELECTED_ICBHI"
)


def import_baseline_audit(path: str):
    spec = importlib.util.spec_from_file_location("hflung_baseline_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_sample(files, max_files=None, seed=42):
    files = list(files)
    if max_files is None or len(files) <= max_files:
        return files

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=max_files, replace=False)
    return [files[i] for i in sorted(idx)]


def get_exph_lung_segment_files(audit, exph_dir: Path):
    """
    EXP_H processed folder contains 2s files.
    We keep only lung-side files L_*.wav.
    """
    files = audit.find_audio_files(exph_dir)
    files = [p for p in files if p.name.startswith("L_") and p.suffix.lower() in audit.AUDIO_EXTS]
    return sorted(files)


def add_hflung_metadata(df: pd.DataFrame, hflung_root: Path) -> pd.DataFrame:
    df = df.copy()

    rel_parts = []
    parent_1 = []
    parent_2 = []
    parent_3 = []

    for p in df["source_file"].astype(str):
        path = Path(p)
        try:
            rel = path.relative_to(hflung_root)
            parts = rel.parts
        except Exception:
            parts = path.parts

        rel_parts.append("/".join(parts))
        parent_1.append(parts[0] if len(parts) >= 2 else "")
        parent_2.append(parts[1] if len(parts) >= 3 else "")
        parent_3.append(parts[2] if len(parts) >= 4 else "")

    df["relative_path"] = rel_parts
    df["folder_1"] = parent_1
    df["folder_2"] = parent_2
    df["folder_3"] = parent_3

    return df


def compute_feature_smd(ref_df, query_df, feature_cols):
    rows = []

    for feat in feature_cols:
        x = ref_df[feat].dropna().values
        y = query_df[feat].dropna().values

        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
        pooled = np.sqrt((sx * sx + sy * sy) / 2.0) + 1e-12
        smd = (my - mx) / pooled

        rows.append({
            "feature": feat,
            "EXP_H_selected_mean": mx,
            "HF_Lung_mean": my,
            "SMD_HFLUNG_minus_SELECTED_ICBHI": smd,
            "abs_SMD": abs(smd),
        })

    return pd.DataFrame(rows).sort_values("abs_SMD", ascending=False)


def label_similarity(frac_p95, frac_p99, median_knn):
    if frac_p95 >= 0.50:
        return "close_to_selected_ICBHI"
    if frac_p95 >= 0.25 or frac_p99 >= 0.50:
        return "partial_overlap"
    return "far_from_selected_ICBHI"


def make_pca_plot(ref_df, query_df, feature_cols, out_path, max_points_per_domain=8000, seed=42):
    rng = np.random.default_rng(seed)

    ref_plot = ref_df.copy()
    query_plot = query_df.copy()

    if len(ref_plot) > max_points_per_domain:
        ref_plot = ref_plot.iloc[rng.choice(len(ref_plot), max_points_per_domain, replace=False)]
    if len(query_plot) > max_points_per_domain:
        query_plot = query_plot.iloc[rng.choice(len(query_plot), max_points_per_domain, replace=False)]

    plot_df = pd.concat([ref_plot, query_plot], ignore_index=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(plot_df[feature_cols].values)

    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(X)

    plot_df["PC1"] = Z[:, 0]
    plot_df["PC2"] = Z[:, 1]

    plt.figure(figsize=(9, 7))

    for domain in ["EXP_H_SELECTED", "HF_LUNG"]:
        sub = plot_df[plot_df["domain"] == domain]
        plt.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.35, label=domain)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    plt.title("HF_Lung vs selected ICBHI/EXP_H LS segments")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline_audit_path",
        "--audit55_path",
        dest="baseline_audit_path",
        type=str,
        default=str(DEFAULT_BASELINE_AUDIT_PATH),
    )
    parser.add_argument("--exph_dir", type=str, default=str(DEFAULT_EXPH_DIR))
    parser.add_argument("--hflung_dir", type=str, default=str(DEFAULT_HFLUNG_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    parser.add_argument("--max_exph_segments", type=int, default=20000)
    parser.add_argument("--max_hflung_files", type=int, default=None)

    parser.add_argument("--sr", type=int, default=4000)
    parser.add_argument("--max_seconds", type=float, default=15.0)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--hop_seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    audit = import_baseline_audit(args.baseline_audit_path)

    exph_dir = Path(args.exph_dir)
    hflung_dir = Path(args.hflung_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("HF_Lung trend audit vs selected ICBHI / EXP_H")
    print("=" * 90)
    print(f"EXP_H selected dir : {exph_dir}")
    print(f"HF_Lung dir        : {hflung_dir}")
    print(f"Output dir         : {out_dir}")
    print("=" * 90)

    # ------------------------------------------------------------
    # EXP_H selected reference cloud
    # ------------------------------------------------------------
    exph_files = get_exph_lung_segment_files(audit, exph_dir)
    print(f"[EXP_H_SELECTED] Total L_*.wav processed segments found: {len(exph_files)}")

    exph_files = random_sample(exph_files, args.max_exph_segments, seed=args.seed)
    print(f"[EXP_H_SELECTED] Segments used as reference cloud: {len(exph_files)}")

    # ------------------------------------------------------------
    # HF_Lung query domain
    # ------------------------------------------------------------
    hflung_files = audit.find_audio_files(hflung_dir)
    hflung_files = sorted(hflung_files)
    print(f"[HF_LUNG] Total audio files found: {len(hflung_files)}")

    hflung_files = random_sample(hflung_files, args.max_hflung_files, seed=args.seed)
    print(f"[HF_LUNG] Files used as query domain: {len(hflung_files)}")

    if len(exph_files) == 0:
        raise RuntimeError("No EXP_H selected L_*.wav files found.")
    if len(hflung_files) == 0:
        raise RuntimeError("No HF_Lung audio files found.")

    # ------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------
    ref_df = audit.extract_feature_table(
        files=exph_files,
        domain="EXP_H_SELECTED",
        sr=args.sr,
        max_seconds=args.max_seconds,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_files=None,
    )

    query_df = audit.extract_feature_table(
        files=hflung_files,
        domain="HF_LUNG",
        sr=args.sr,
        max_seconds=args.max_seconds,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_files=None,
    )

    if ref_df.empty:
        raise RuntimeError("No EXP_H selected features extracted.")
    if query_df.empty:
        raise RuntimeError("No HF_Lung features extracted.")

    query_df = add_hflung_metadata(query_df, hflung_dir)

    feature_cols = audit.FEATURE_COLUMNS

    # ------------------------------------------------------------
    # Standardize on selected ICBHI / EXP_H only
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref_df[feature_cols].values)
    X_query = scaler.transform(query_df[feature_cols].values)

    # Reference centroid distance
    ref_centroid = np.mean(X_ref, axis=0)

    ref_centroid_dist = np.linalg.norm(X_ref - ref_centroid, axis=1)
    query_centroid_dist = np.linalg.norm(X_query - ref_centroid, axis=1)

    ref_thresholds = {
        "ref_centroid_p50": float(np.percentile(ref_centroid_dist, 50)),
        "ref_centroid_p75": float(np.percentile(ref_centroid_dist, 75)),
        "ref_centroid_p90": float(np.percentile(ref_centroid_dist, 90)),
        "ref_centroid_p95": float(np.percentile(ref_centroid_dist, 95)),
        "ref_centroid_p99": float(np.percentile(ref_centroid_dist, 99)),
    }

    # KNN to selected ICBHI / EXP_H cloud
    nn = NearestNeighbors(n_neighbors=10, metric="euclidean", n_jobs=-1)
    nn.fit(X_ref)
    dists, idxs = nn.kneighbors(X_query)

    query_df["dist_to_selected_icbhi_centroid"] = query_centroid_dist
    query_df["nn1_dist_to_selected_icbhi"] = dists[:, 0]
    query_df["nn5_mean_dist_to_selected_icbhi"] = dists[:, :5].mean(axis=1)
    query_df["nn10_mean_dist_to_selected_icbhi"] = dists[:, :10].mean(axis=1)

    query_df["inside_selected_icbhi_p95_centroid"] = (
        query_df["dist_to_selected_icbhi_centroid"] <= ref_thresholds["ref_centroid_p95"]
    )
    query_df["inside_selected_icbhi_p99_centroid"] = (
        query_df["dist_to_selected_icbhi_centroid"] <= ref_thresholds["ref_centroid_p99"]
    )

    # ------------------------------------------------------------
    # Aggregate per HF_Lung file
    # ------------------------------------------------------------
    agg = (
        query_df
        .groupby(
            ["source_file", "source_name", "relative_path", "folder_1", "folder_2", "folder_3", "phase_hint"],
            as_index=False
        )
        .agg(
            n_segments=("segment_idx", "count"),

            median_centroid_dist=("dist_to_selected_icbhi_centroid", "median"),
            mean_centroid_dist=("dist_to_selected_icbhi_centroid", "mean"),
            p75_centroid_dist=("dist_to_selected_icbhi_centroid", lambda x: float(np.percentile(x, 75))),

            median_nn1_dist=("nn1_dist_to_selected_icbhi", "median"),
            mean_nn1_dist=("nn1_dist_to_selected_icbhi", "mean"),
            p75_nn1_dist=("nn1_dist_to_selected_icbhi", lambda x: float(np.percentile(x, 75))),

            median_nn5_mean_dist=("nn5_mean_dist_to_selected_icbhi", "median"),
            median_nn10_mean_dist=("nn10_mean_dist_to_selected_icbhi", "median"),

            frac_inside_selected_icbhi_p95=("inside_selected_icbhi_p95_centroid", "mean"),
            frac_inside_selected_icbhi_p99=("inside_selected_icbhi_p99_centroid", "mean"),
        )
    )

    agg["similarity_label"] = [
        label_similarity(p95, p99, nn1)
        for p95, p99, nn1 in zip(
            agg["frac_inside_selected_icbhi_p95"],
            agg["frac_inside_selected_icbhi_p99"],
            agg["median_nn1_dist"],
        )
    ]

    # Main ranking:
    # 1) more segments inside selected ICBHI p95
    # 2) lower median centroid distance
    # 3) lower median NN distance
    agg = agg.sort_values(
        by=[
            "frac_inside_selected_icbhi_p95",
            "frac_inside_selected_icbhi_p99",
            "median_centroid_dist",
            "median_nn1_dist",
        ],
        ascending=[False, False, True, True],
    )

    # ------------------------------------------------------------
    # Group trends by folders / phase
    # ------------------------------------------------------------
    group_cols = ["folder_1", "folder_2", "folder_3", "phase_hint"]

    group_summary = (
        agg
        .groupby(group_cols, as_index=False)
        .agg(
            n_files=("source_file", "count"),
            median_frac_inside_p95=("frac_inside_selected_icbhi_p95", "median"),
            mean_frac_inside_p95=("frac_inside_selected_icbhi_p95", "mean"),
            median_centroid_dist=("median_centroid_dist", "median"),
            median_nn1_dist=("median_nn1_dist", "median"),
        )
        .sort_values(
            by=["median_frac_inside_p95", "median_centroid_dist"],
            ascending=[False, True],
        )
    )

    # ------------------------------------------------------------
    # Feature-level SMD: global HF_Lung vs selected ICBHI
    # ------------------------------------------------------------
    smd_df = compute_feature_smd(ref_df, query_df, feature_cols)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    ref_df.to_csv(out_dir / "selected_icbhi_reference_segment_features.csv", index=False)
    query_df.to_csv(out_dir / "hflung_segment_features_with_similarity.csv", index=False)
    agg.to_csv(out_dir / "hflung_files_ranked_by_similarity_to_selected_icbhi.csv", index=False)
    group_summary.to_csv(out_dir / "hflung_group_trends_by_folder_phase.csv", index=False)
    smd_df.to_csv(out_dir / "global_feature_smd_hflung_minus_selected_icbhi.csv", index=False)

    make_pca_plot(
        ref_df=ref_df,
        query_df=query_df,
        feature_cols=feature_cols,
        out_path=out_dir / "pca_selected_icbhi_vs_hflung_segments.png",
        seed=args.seed,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    label_counts = agg["similarity_label"].value_counts().to_dict()

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("HF_Lung trend audit vs selected ICBHI / EXP_H\n")
        f.write("=" * 80 + "\n\n")

        f.write("Question\n")
        f.write("Among all HF_Lung recordings, are there files/subsets similar to the selected ICBHI/EXP_H lung distribution?\n\n")

        f.write("Data\n")
        f.write(f"- EXP_H selected reference segments: {len(ref_df)}\n")
        f.write(f"- HF_Lung files: {agg['source_file'].nunique()}\n")
        f.write(f"- HF_Lung segments: {len(query_df)}\n\n")

        f.write("Reference centroid thresholds from selected ICBHI / EXP_H\n")
        for k, v in ref_thresholds.items():
            f.write(f"- {k}: {v:.4f}\n")

        f.write("\nGlobal distance\n")
        f.write(f"- Mean |SMD| HF_Lung vs selected ICBHI: {smd_df['abs_SMD'].mean():.4f}\n")
        f.write(f"- Median |SMD| HF_Lung vs selected ICBHI: {smd_df['abs_SMD'].median():.4f}\n\n")

        f.write("HF_Lung file-level similarity labels\n")
        for k, v in label_counts.items():
            f.write(f"- {k}: {v}\n")

        f.write("\nTop 20 HF_Lung files closest to selected ICBHI\n")
        top_cols = [
            "source_name",
            "relative_path",
            "n_segments",
            "frac_inside_selected_icbhi_p95",
            "frac_inside_selected_icbhi_p99",
            "median_centroid_dist",
            "median_nn1_dist",
            "similarity_label",
        ]
        f.write(agg.head(20)[top_cols].to_string(index=False))
        f.write("\n\n")

        f.write("Top feature differences\n")
        f.write(smd_df.head(10)[["feature", "SMD_HFLUNG_minus_SELECTED_ICBHI", "abs_SMD"]].to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation guide\n")
        f.write("- Many close_to_selected_ICBHI files: HF_Lung contains a subdomain similar to the selected ICBHI distribution.\n")
        f.write("- Mostly partial_overlap: there is a weak/intermediate trend, but HF_Lung is not fully covered by selected ICBHI.\n")
        f.write("- Mostly far_from_selected_ICBHI: HF_Lung is largely a different respiratory domain.\n")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Output dir: {out_dir}")

    print("\nGlobal Mean |SMD|:")
    print(f"{smd_df['abs_SMD'].mean():.4f}")

    print("\nHF_Lung similarity labels:")
    print(agg["similarity_label"].value_counts())

    print("\nTop 15 closest HF_Lung files:")
    print(
        agg.head(15)[[
            "source_name",
            "relative_path",
            "frac_inside_selected_icbhi_p95",
            "frac_inside_selected_icbhi_p99",
            "median_centroid_dist",
            "median_nn1_dist",
            "similarity_label",
        ]].to_string(index=False)
    )

    print("\nSaved files:")
    print("- summary.txt")
    print("- hflung_files_ranked_by_similarity_to_selected_icbhi.csv")
    print("- hflung_group_trends_by_folder_phase.csv")
    print("- global_feature_smd_hflung_minus_selected_icbhi.csv")
    print("- pca_selected_icbhi_vs_hflung_segments.png")


if __name__ == "__main__":
    main()

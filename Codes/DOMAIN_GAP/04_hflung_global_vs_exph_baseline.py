#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_hflung_global_vs_exph_baseline.py

Audit-only script.
Goal:
    Compare EXP_H / ICBHI selected lung sources against HLung lung recordings.

It only extracts LS segment-level RMS-normalized acoustic features and computes:
    - feature-wise SMD
    - source-level nearest-neighbor distance HLung -> EXP_H
    - EXP_H internal nearest-neighbor baseline
    - PCA visualization

Interpretation:
    If HLung sources are far from the selected EXP_H/ICBHI sources, then EXP_H/ICBHI
    should be treated as one real-patient reference domain, not as a universal
    representation of all real clinical lung sounds.
"""

import argparse
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


# ============================================================
# REPOSITORY-RELATIVE DEFAULT PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(REPO_ROOT))).expanduser().resolve()

DEFAULT_EXPH_DIR = (
    PROJECT_ROOT / "dataset" / "processed" / "experiment_H_full_both"
)
DEFAULT_HLUNG_DIR = PROJECT_ROOT / "dataset" / "raw" / "HF_Lung_V1"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "outputs" / "domain_gap" / "HLUNG_vs_EXPH_similarity_audit"
)


# ============================================================
# AUDIO I/O
# ============================================================

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def load_audio_any(path: Path, target_sr: int = 4000) -> tuple[np.ndarray, int]:
    """
    Robust audio loading.
    First tries soundfile.
    Falls back to librosa if available.
    """
    path = Path(path)

    try:
        import soundfile as sf

        x, sr = sf.read(str(path), always_2d=False)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        x = x.astype(np.float32)
    except Exception:
        try:
            import librosa

            x, sr = librosa.load(str(path), sr=None, mono=True)
            x = x.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Could not load audio file: {path}\nError: {e}")

    if not np.isfinite(x).all():
        x = np.nan_to_num(x)

    if sr != target_sr:
        x = resample_to_target_sr(x, sr, target_sr)
        sr = target_sr

    return x.astype(np.float32), sr


def resample_to_target_sr(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """
    Resample using scipy.signal.resample_poly.
    """
    if sr == target_sr:
        return x

    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g

    y = signal.resample_poly(x, up, down)
    return y.astype(np.float32)


def preprocess_audio(
    x: np.ndarray,
    sr: int,
    max_seconds: float = 15.0,
    min_seconds: float = 2.0,
) -> np.ndarray | None:
    """
    Mono already done.
    DC removal.
    Keep first max_seconds to match EXP_H/Torabi protocol.
    """
    if x is None or len(x) == 0:
        return None

    x = x.astype(np.float32)
    x = x - np.mean(x)

    max_len = int(max_seconds * sr)
    min_len = int(min_seconds * sr)

    if len(x) < min_len:
        return None

    if len(x) > max_len:
        x = x[:max_len]

    peak = np.max(np.abs(x)) + 1e-12
    if peak <= 1e-8:
        return None

    return x


def make_segments(
    x: np.ndarray,
    sr: int,
    segment_seconds: float = 2.0,
    hop_seconds: float = 0.5,
) -> list[np.ndarray]:
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)

    if len(x) < seg_len:
        return []

    segments = []
    for start in range(0, len(x) - seg_len + 1, hop_len):
        seg = x[start:start + seg_len].astype(np.float32)
        segments.append(seg)

    return segments


# ============================================================
# FEATURE EXTRACTION
# ============================================================

FEATURE_COLUMNS = [
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


def rms_normalize(seg: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    seg = seg.astype(np.float32)
    seg = seg - np.mean(seg)
    rms = np.sqrt(np.mean(seg ** 2))

    if rms < eps:
        return None

    return seg / (rms + eps)


def band_energy_ratio(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    total_mask = (freqs >= 20.0) & (freqs <= 2000.0)
    band_mask = (freqs >= low) & (freqs < high)

    total = np.sum(power[total_mask]) + 1e-12
    band = np.sum(power[band_mask])

    return float(band / total)


def extract_features_from_segment(seg: np.ndarray, sr: int = 4000) -> dict | None:
    """
    Segment-level RMS-normalized features.
    This mirrors the domain-gap audit logic:
        LS · segment-level · RMS-normalized
    """
    seg = rms_normalize(seg)
    if seg is None:
        return None

    # ZCR
    signs = np.sign(seg)
    signs[signs == 0] = 1
    zcr = np.mean(signs[1:] != signs[:-1])

    # Crest factor
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-12
    crest_factor = np.max(np.abs(seg)) / rms

    # FFT power spectrum
    window = np.hanning(len(seg)).astype(np.float32)
    xw = seg * window
    spec = np.fft.rfft(xw)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)

    valid = (freqs >= 20.0) & (freqs <= 2000.0)
    f = freqs[valid]
    p = power[valid] + 1e-20

    total_p = np.sum(p) + 1e-20

    spectral_centroid = np.sum(f * p) / total_p
    spectral_bandwidth = np.sqrt(np.sum(((f - spectral_centroid) ** 2) * p) / total_p)

    spectral_flatness = np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-20)

    features = {
        "zcr": float(zcr),
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "spectral_flatness": float(spectral_flatness),
        "crest_factor": float(crest_factor),
        "energy_20_50_ratio": band_energy_ratio(freqs, power, 20, 50),
        "energy_20_80_ratio": band_energy_ratio(freqs, power, 20, 80),
        "heart_band_20_500_ratio": band_energy_ratio(freqs, power, 20, 500),
        "lung_band_50_1800_ratio": band_energy_ratio(freqs, power, 50, 1800),
        "ls_band_50_150_ratio": band_energy_ratio(freqs, power, 50, 150),
        "ls_band_150_500_ratio": band_energy_ratio(freqs, power, 150, 500),
        "ls_band_500_1000_ratio": band_energy_ratio(freqs, power, 500, 1000),
        "ls_band_1000_1800_ratio": band_energy_ratio(freqs, power, 1000, 1800),
        "hf_noise_ratio_1000_2000": band_energy_ratio(freqs, power, 1000, 2000),
    }

    return features


# ============================================================
# PATH DISCOVERY
# ============================================================

def find_audio_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []

    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)

    return sorted(files)


def find_manifest_candidates(root: Path) -> list[Path]:
    root = Path(root)
    candidates = []

    for p in root.rglob("*.csv"):
        name = p.name.lower()
        if any(k in name for k in ["manifest", "metadata", "split", "source"]):
            candidates.append(p)

    return sorted(candidates)


def resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return base_dir / p


def load_exph_lung_files_from_manifest(
    exph_dir: Path,
    explicit_manifest: Path | None = None,
    explicit_l_col: str | None = None,
) -> list[Path]:
    """
    Tries to find unique LS source paths from EXP_H manifest.
    If it fails, fallback should be recursive audio discovery.
    """
    exph_dir = Path(exph_dir)

    if explicit_manifest is not None:
        manifest_candidates = [Path(explicit_manifest)]
    else:
        manifest_candidates = find_manifest_candidates(exph_dir)

    lung_col_keywords = [
        "l_path",
        "ls_path",
        "lung_path",
        "source_l_path",
        "source_ls_path",
        "source_lung_path",
        "path_l",
        "path_ls",
        "path_lung",
        "l_file",
        "ls_file",
        "lung_file",
        "lung",
        "ls",
    ]

    for manifest_path in manifest_candidates:
        try:
            df = pd.read_csv(manifest_path)
        except Exception:
            continue

        if df.empty:
            continue

        cols = list(df.columns)

        if explicit_l_col is not None:
            candidate_cols = [explicit_l_col] if explicit_l_col in cols else []
        else:
            candidate_cols = []
            for c in cols:
                cl = c.lower()
                if any(k == cl or k in cl for k in lung_col_keywords):
                    candidate_cols.append(c)

        # Prefer columns that look like paths
        path_like_cols = []
        for c in candidate_cols:
            sample_values = df[c].dropna().astype(str).head(20).tolist()
            if any(Path(v).suffix.lower() in AUDIO_EXTS for v in sample_values):
                path_like_cols.append(c)

        for col in path_like_cols:
            paths = []
            for v in df[col].dropna().astype(str).unique():
                p = resolve_path(v, exph_dir)
                if p.exists() and p.suffix.lower() in AUDIO_EXTS:
                    paths.append(p)

            paths = sorted(set(paths))
            if len(paths) > 0:
                print(f"[EXP_H] Using manifest: {manifest_path}")
                print(f"[EXP_H] Using lung column: {col}")
                print(f"[EXP_H] Unique LS files found: {len(paths)}")
                return paths

    return []


def fallback_exph_audio_discovery(exph_dir: Path) -> list[Path]:
    """
    Fallback if manifest loading fails.
    It searches for audio files whose path hints LS/lung.
    """
    all_audio = find_audio_files(exph_dir)

    lung_hint_patterns = [
        r"(^|[/_\\-])ls($|[/_\\-])",
        r"lung",
        r"resp",
        r"icbhi",
    ]

    selected = []
    for p in all_audio:
        s = str(p).lower()
        if any(re.search(pattern, s) for pattern in lung_hint_patterns):
            selected.append(p)

    selected = sorted(set(selected))

    if len(selected) == 0:
        warnings.warn(
            "Could not identify LS files by path hints. "
            "Falling back to all audio files under EXP_H directory."
        )
        selected = all_audio

    print(f"[EXP_H] Fallback audio discovery selected files: {len(selected)}")
    return selected


def detect_hlung_phase_hint(path: Path) -> str:
    """
    Optional metadata only.
    HLung may contain inspiration/expiration labels in filename or folders.
    """
    s = str(path).lower()

    if any(k in s for k in ["inspiration", "inspiratory", "insp", "inhale", "inhalation"]):
        return "inspiration"
    if any(k in s for k in ["expiration", "expiratory", "exp", "exhale", "exhalation"]):
        return "expiration"
    return "unknown"


# ============================================================
# FEATURE TABLE BUILDING
# ============================================================

def extract_feature_table(
    files: list[Path],
    domain: str,
    sr: int,
    max_seconds: float,
    segment_seconds: float,
    hop_seconds: float,
    max_files: int | None = None,
) -> pd.DataFrame:
    rows = []

    if max_files is not None and len(files) > max_files:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(files), size=max_files, replace=False)
        files = [files[i] for i in sorted(idx)]

    print(f"[{domain}] Files to process: {len(files)}")

    for i, path in enumerate(files, start=1):
        if i % 50 == 0 or i == 1 or i == len(files):
            print(f"[{domain}] Processing {i}/{len(files)}")

        try:
            x, file_sr = load_audio_any(path, target_sr=sr)
            x = preprocess_audio(x, sr=file_sr, max_seconds=max_seconds)
            if x is None:
                continue

            segments = make_segments(
                x,
                sr=sr,
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
            )

            if len(segments) == 0:
                continue

            for seg_idx, seg in enumerate(segments):
                feats = extract_features_from_segment(seg, sr=sr)
                if feats is None:
                    continue

                row = {
                    "domain": domain,
                    "source_file": str(path),
                    "source_name": path.name,
                    "segment_idx": seg_idx,
                    "phase_hint": detect_hlung_phase_hint(path) if domain.upper() == "HLUNG" else "not_applicable",
                }
                row.update(feats)
                rows.append(row)

        except Exception as e:
            warnings.warn(f"Skipping {path}: {e}")

    df = pd.DataFrame(rows)

    print(f"[{domain}] Segment rows extracted: {len(df)}")
    if not df.empty:
        print(f"[{domain}] Unique sources extracted: {df['source_file'].nunique()}")

    return df


def aggregate_source_level(segment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate segment-level features to source-level by averaging segments.
    """
    group_cols = ["domain", "source_file", "source_name", "phase_hint"]

    source_df = (
        segment_df
        .groupby(group_cols, as_index=False)[FEATURE_COLUMNS]
        .mean()
    )

    return source_df


# ============================================================
# ANALYSIS
# ============================================================

def compute_smd_table(source_df: pd.DataFrame) -> pd.DataFrame:
    exph = source_df[source_df["domain"] == "EXP_H"]
    hlung = source_df[source_df["domain"] == "HLUNG"]

    rows = []
    for feat in FEATURE_COLUMNS:
        x = exph[feat].dropna().values
        y = hlung[feat].dropna().values

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)

        pooled = np.sqrt((std_x ** 2 + std_y ** 2) / 2.0) + 1e-12
        smd = (mean_y - mean_x) / pooled

        rows.append({
            "feature": feat,
            "EXP_H_mean": mean_x,
            "HLUNG_mean": mean_y,
            "EXP_H_std": std_x,
            "HLUNG_std": std_y,
            "SMD_HLUNG_minus_EXPH": smd,
            "abs_SMD": abs(smd),
        })

    smd_df = pd.DataFrame(rows).sort_values("abs_SMD", ascending=False)

    return smd_df


def compute_nearest_neighbors(source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exph = source_df[source_df["domain"] == "EXP_H"].copy().reset_index(drop=True)
    hlung = source_df[source_df["domain"] == "HLUNG"].copy().reset_index(drop=True)

    scaler = StandardScaler()
    X_exph = scaler.fit_transform(exph[FEATURE_COLUMNS].values)
    X_hlung = scaler.transform(hlung[FEATURE_COLUMNS].values)

    # HLung -> EXP_H distance
    D = pairwise_distances(X_hlung, X_exph, metric="euclidean")
    nn_idx = np.argmin(D, axis=1)
    nn_dist = D[np.arange(D.shape[0]), nn_idx]

    nn_rows = []
    for i in range(len(hlung)):
        nn_rows.append({
            "hlung_source_file": hlung.loc[i, "source_file"],
            "hlung_source_name": hlung.loc[i, "source_name"],
            "hlung_phase_hint": hlung.loc[i, "phase_hint"],
            "nearest_exph_source_file": exph.loc[nn_idx[i], "source_file"],
            "nearest_exph_source_name": exph.loc[nn_idx[i], "source_name"],
            "distance_to_nearest_EXPH": nn_dist[i],
        })

    nn_df = pd.DataFrame(nn_rows).sort_values("distance_to_nearest_EXPH")

    # EXP_H internal nearest neighbor baseline
    D_internal = pairwise_distances(X_exph, X_exph, metric="euclidean")
    np.fill_diagonal(D_internal, np.inf)
    internal_nn = np.min(D_internal, axis=1)

    internal_df = pd.DataFrame({
        "exph_source_file": exph["source_file"],
        "exph_source_name": exph["source_name"],
        "distance_to_nearest_other_EXPH": internal_nn,
    }).sort_values("distance_to_nearest_other_EXPH")

    p50 = float(np.percentile(internal_nn, 50))
    p75 = float(np.percentile(internal_nn, 75))
    p90 = float(np.percentile(internal_nn, 90))
    p95 = float(np.percentile(internal_nn, 95))

    summary = {
        "EXPH_internal_NN_median": p50,
        "EXPH_internal_NN_p75": p75,
        "EXPH_internal_NN_p90": p90,
        "EXPH_internal_NN_p95": p95,
        "HLUNG_to_EXPH_NN_mean": float(np.mean(nn_dist)),
        "HLUNG_to_EXPH_NN_median": float(np.median(nn_dist)),
        "HLUNG_to_EXPH_NN_p75": float(np.percentile(nn_dist, 75)),
        "HLUNG_to_EXPH_NN_p90": float(np.percentile(nn_dist, 90)),
        "HLUNG_to_EXPH_NN_p95": float(np.percentile(nn_dist, 95)),
        "HLUNG_fraction_within_EXPH_p95": float(np.mean(nn_dist <= p95)),
        "HLUNG_fraction_beyond_EXPH_p95": float(np.mean(nn_dist > p95)),
    }

    return nn_df, internal_df, summary


def make_pca_plot(source_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    scaler = StandardScaler()
    X = scaler.fit_transform(source_df[FEATURE_COLUMNS].values)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    pca_df = source_df[["domain", "source_file", "source_name", "phase_hint"]].copy()
    pca_df["PC1"] = Z[:, 0]
    pca_df["PC2"] = Z[:, 1]

    plt.figure(figsize=(9, 7))

    for domain in ["EXP_H", "HLUNG"]:
        sub = pca_df[pca_df["domain"] == domain]
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            label=domain,
            alpha=0.75,
            s=35,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    plt.title("EXP_H / ICBHI vs HLung - LS source-level feature PCA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    return pca_df


def write_summary(
    out_path: Path,
    n_exph_sources: int,
    n_hlung_sources: int,
    n_exph_segments: int,
    n_hlung_segments: int,
    smd_df: pd.DataFrame,
    nn_summary: dict,
) -> None:
    mean_abs_smd = float(smd_df["abs_SMD"].mean())
    median_abs_smd = float(smd_df["abs_SMD"].median())

    top_features = smd_df.head(8)[["feature", "SMD_HLUNG_minus_EXPH", "abs_SMD"]]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("HLung vs EXP_H / ICBHI Similarity Audit\n")
        f.write("=" * 70 + "\n\n")

        f.write("Scope\n")
        f.write("- Feature scope: LS segment-level RMS-normalized features\n")
        f.write("- Source aggregation: mean over 2 s / 0.5 s segments\n")
        f.write("- Purpose: audit-only, no training, no dataset modification\n\n")

        f.write("Dataset sizes\n")
        f.write(f"- EXP_H sources: {n_exph_sources}\n")
        f.write(f"- HLung sources: {n_hlung_sources}\n")
        f.write(f"- EXP_H segment rows: {n_exph_segments}\n")
        f.write(f"- HLung segment rows: {n_hlung_segments}\n\n")

        f.write("Feature-level distance\n")
        f.write(f"- Mean |SMD|: {mean_abs_smd:.4f}\n")
        f.write(f"- Median |SMD|: {median_abs_smd:.4f}\n\n")

        f.write("Nearest-neighbor distance summary\n")
        for k, v in nn_summary.items():
            f.write(f"- {k}: {v:.4f}\n")

        f.write("\nTop SMD features\n")
        f.write(top_features.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation guide\n")
        f.write("- If HLUNG_to_EXPH_NN_median is close to EXP_H internal NN median/p75, HLung overlaps acoustically with EXP_H.\n")
        f.write("- If HLUNG_to_EXPH_NN_median is much larger than EXP_H internal NN p95, HLung is mostly outside the selected EXP_H/ICBHI coverage.\n")
        f.write("- If HLUNG_fraction_beyond_EXPH_p95 is high, EXP_H/ICBHI should be treated as one real-patient reference domain, not universal clinical LS coverage.\n")
        f.write("- This does not invalidate EXP_H vs Torabi results; it only quantifies real-patient heterogeneity.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exph_dir", type=str, default=str(DEFAULT_EXPH_DIR))
    parser.add_argument("--hlung_dir", type=str, default=str(DEFAULT_HLUNG_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    parser.add_argument("--exph_manifest", type=str, default=None)
    parser.add_argument("--exph_l_col", type=str, default=None)

    parser.add_argument("--sr", type=int, default=4000)
    parser.add_argument("--max_seconds", type=float, default=15.0)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--hop_seconds", type=float, default=0.5)

    parser.add_argument(
        "--max_hlung_files",
        type=int,
        default=None,
        help="Optional cap for quick smoke test. Use None for all files.",
    )
    parser.add_argument(
        "--segment_level_only",
        action="store_true",
        help="Use segment-level comparison only. Recommended when EXP_H dir already contains 2s processed segments.",
    )

    parser.add_argument(
        "--max_exph_files",
        type=int,
        default=None,
        help="Optional cap for EXP_H files. Use for smoke test.",
    )

    args = parser.parse_args()

    exph_dir = Path(args.exph_dir)
    hlung_dir = Path(args.hlung_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HLung vs EXP_H / ICBHI Similarity Audit")
    print("=" * 80)
    print(f"EXP_H dir : {exph_dir}")
    print(f"HLung dir : {hlung_dir}")
    print(f"Out dir   : {out_dir}")
    print("=" * 80)

    # ------------------------------------------------------------
    # EXP_H files
    # ------------------------------------------------------------
    exph_manifest = Path(args.exph_manifest) if args.exph_manifest else None

    exph_files = load_exph_lung_files_from_manifest(
        exph_dir=exph_dir,
        explicit_manifest=exph_manifest,
        explicit_l_col=args.exph_l_col,
    )

    if len(exph_files) == 0:
        exph_files = fallback_exph_audio_discovery(exph_dir)

    exph_files = sorted(set(exph_files))

    # ------------------------------------------------------------
    # HLung files
    # ------------------------------------------------------------
    hlung_files = find_audio_files(hlung_dir)
    hlung_files = sorted(set(hlung_files))

    if len(hlung_files) == 0:
        raise RuntimeError(f"No HLung audio files found under: {hlung_dir}")

    print(f"[EXP_H] Final LS file count: {len(exph_files)}")
    print(f"[HLUNG] Final audio file count: {len(hlung_files)}")

    # ------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------
    exph_seg_df = extract_feature_table(
        files=exph_files,
        domain="EXP_H",
        sr=args.sr,
        max_seconds=args.max_seconds,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_files=args.max_exph_files,
    )

    hlung_seg_df = extract_feature_table(
        files=hlung_files,
        domain="HLUNG",
        sr=args.sr,
        max_seconds=args.max_seconds,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_files=args.max_hlung_files,
    )

    if exph_seg_df.empty:
        raise RuntimeError("No EXP_H segment features extracted. Check paths/manifest.")
    if hlung_seg_df.empty:
        raise RuntimeError("No HLung segment features extracted. Check paths/audio format.")

    segment_df = pd.concat([exph_seg_df, hlung_seg_df], ignore_index=True)

    if args.segment_level_only:
        print("[MODE] Segment-level only audit enabled.")
        analysis_df = segment_df.copy()
        analysis_df["source_file"] = analysis_df["source_file"].astype(str)
        analysis_df["source_name"] = analysis_df["source_name"].astype(str)
    else:
        source_df = aggregate_source_level(segment_df)
        analysis_df = source_df.copy()

    # ------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------
    smd_df = compute_smd_table(analysis_df)

    if args.segment_level_only:
        nn_df = None
        internal_df = None
        nn_summary = {}
        pca_df = make_pca_plot(analysis_df, out_dir / "pca_segments_EXPH_vs_HLUNG.png")
    else:
        nn_df, internal_df, nn_summary = compute_nearest_neighbors(analysis_df)
        pca_df = make_pca_plot(analysis_df, out_dir / "pca_sources_EXPH_vs_HLUNG.png")

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    segment_df.to_csv(out_dir / "segment_features_LS_RMS.csv", index=False)
    analysis_df.to_csv(out_dir / "analysis_features_LS_RMS.csv", index=False)
    smd_df.to_csv(out_dir / "feature_smd_HLUNG_minus_EXPH.csv", index=False)
    pca_df.to_csv(out_dir / "pca_coordinates.csv", index=False)

    if not args.segment_level_only:
        nn_df.to_csv(out_dir / "nearest_neighbors_HLUNG_to_EXPH.csv", index=False)
        internal_df.to_csv(out_dir / "nearest_neighbors_EXPH_internal.csv", index=False)

    if args.segment_level_only:
        n_exph_items = analysis_df[analysis_df["domain"] == "EXP_H"]["source_file"].nunique()
        n_hlung_items = analysis_df[analysis_df["domain"] == "HLUNG"]["source_file"].nunique()
    else:
        n_exph_items = source_df[source_df["domain"] == "EXP_H"]["source_file"].nunique()
        n_hlung_items = source_df[source_df["domain"] == "HLUNG"]["source_file"].nunique()

    write_summary(
        out_path=out_dir / "summary.txt",
        n_exph_sources=n_exph_items,
        n_hlung_sources=n_hlung_items,
        n_exph_segments=len(exph_seg_df),
        n_hlung_segments=len(hlung_seg_df),
        smd_df=smd_df,
        nn_summary=nn_summary,
    )

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print("\nDataset sizes:")
    if args.segment_level_only:
        print("Segment rows:")
        print(analysis_df.groupby("domain").size())
        print("\nUnique files:")
        print(analysis_df.groupby("domain")["source_file"].nunique())
    else:
        print(source_df.groupby("domain")["source_file"].nunique())

    print("\nMean |SMD|:")
    print(f"{smd_df['abs_SMD'].mean():.4f}")

    print("\nTop SMD features:")
    print(smd_df.head(10)[["feature", "SMD_HLUNG_minus_EXPH", "abs_SMD"]].to_string(index=False))

    if nn_summary:
        print("\nNearest-neighbor summary:")
        for k, v in nn_summary.items():
            print(f"{k}: {v:.4f}")
    else:
        print("\nNearest-neighbor summary: skipped in segment-level-only mode")

    print("\nOutputs saved in:")
    print(out_dir)


if __name__ == "__main__":
    main()
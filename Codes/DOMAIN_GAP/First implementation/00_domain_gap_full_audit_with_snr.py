#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Domain gap audit between EXP_H full-both and a Torabi source-disjoint fold.

Main goal:
- compare source-domain EXP_H against target-domain Torabi fold;
- analyse HS and LS separately;
- compute raw and RMS-normalized acoustic features;
- compute standardized mean differences (SMD);
- generate PCA projections;
- audit Torabi local source observability through local SNR.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except Exception as exc:
    raise RuntimeError("Missing dependency: soundfile. Install with: pip install soundfile") from exc

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None


EPS = 1e-12


# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def pick_existing_path(row: pd.Series, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        value = str(value)
        if value and Path(value).exists():
            return value
    return None


def read_audio(path: str, expected_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=False)

    if x.ndim == 2:
        x = x.mean(axis=1)

    x = np.asarray(x, dtype=np.float32)

    if expected_sr is not None and sr != expected_sr:
        # Avoid hidden resampling in the audit. If this happens, it is important to know.
        raise ValueError(f"Unexpected sample rate for {path}: got {sr}, expected {expected_sr}")

    return x, sr


def safe_db(x: float) -> float:
    return 10.0 * math.log10(max(float(x), EPS))


def safe_amp_db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), EPS))


# ---------------------------------------------------------------------
# Split filtering
# ---------------------------------------------------------------------

def load_filtered_manifest(processed_dir: Path, split_name: str, fold_no: int) -> pd.DataFrame:
    manifest_path = processed_dir / "manifest_synth_supervised.csv"
    split_path = processed_dir / "source_disjoint_split_smoke.csv"

    manifest = read_csv_required(manifest_path)
    split_df = read_csv_required(split_path)

    if "fold_no" in split_df.columns:
        split_df = split_df[split_df["fold_no"].astype(int) == int(fold_no)].copy()

    if "split" not in split_df.columns:
        raise ValueError(f"Split file has no 'split' column: {split_path}")

    split_df = split_df[split_df["split"].astype(str).str.lower() == split_name.lower()].copy()

    if "base_id" not in split_df.columns:
        raise ValueError(f"Split file has no 'base_id' column: {split_path}")

    allowed_base_ids = set(split_df["base_id"].astype(int).unique())

    if "base_id" not in manifest.columns:
        raise ValueError(f"Manifest has no 'base_id' column: {manifest_path}")

    out = manifest[manifest["base_id"].astype(int).isin(allowed_base_ids)].copy()

    if "split" in out.columns:
        # Keep coherent split if manifest has it. This is just an additional guard.
        manifest_split = out["split"].astype(str).str.lower()
        if (manifest_split == split_name.lower()).any():
            out = out[manifest_split == split_name.lower()].copy()

    out.reset_index(drop=True, inplace=True)
    return out


def sample_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def band_power(freqs: np.ndarray, power: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(power[mask].sum())


def extract_features_from_signal(
    x: np.ndarray,
    sr: int,
    mode: str,
    source_type: str,
    domain: str,
    item_id: str,
    audio_path: str,
) -> Dict[str, float | str]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    raw_rms = float(np.sqrt(np.mean(x ** 2) + EPS))
    raw_peak = float(np.max(np.abs(x)) + EPS)

    if mode == "rms":
        x_feat = x / raw_rms
    elif mode == "raw":
        x_feat = x
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    rms = float(np.sqrt(np.mean(x_feat ** 2) + EPS))
    peak = float(np.max(np.abs(x_feat)) + EPS)
    crest = float(peak / max(rms, EPS))

    # Basic time-domain features
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x_feat).astype(np.int8)))))

    # Frequency features
    n = len(x_feat)
    if n < 8:
        raise ValueError(f"Signal too short: {audio_path}")

    window = np.hanning(n)
    xw = x_feat * window
    spec = np.fft.rfft(xw)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    total_power = float(power.sum() + EPS)

    bands = {
        "energy_20_50": band_power(freqs, power, 20, 50),
        "energy_20_80": band_power(freqs, power, 20, 80),
        "energy_20_500": band_power(freqs, power, 20, 500),
        "energy_50_150": band_power(freqs, power, 50, 150),
        "energy_50_1800": band_power(freqs, power, 50, 1800),
        "energy_80_150": band_power(freqs, power, 80, 150),
        "energy_150_500": band_power(freqs, power, 150, 500),
        "energy_200_500": band_power(freqs, power, 200, 500),
        "energy_500_1000": band_power(freqs, power, 500, 1000),
        "energy_1000_1800": band_power(freqs, power, 1000, 1800),
        "energy_1800_2000": band_power(freqs, power, 1800, min(2000, sr / 2)),
        "energy_1000_2000": band_power(freqs, power, 1000, min(2000, sr / 2)),
    }

    spectral_centroid = float((freqs * power).sum() / total_power)
    spectral_bandwidth = float(np.sqrt((((freqs - spectral_centroid) ** 2) * power).sum() / total_power))

    p_nonzero = power[power > 0]
    spectral_flatness = float(np.exp(np.mean(np.log(p_nonzero + EPS))) / (np.mean(p_nonzero) + EPS))

    result: Dict[str, float | str] = {
        "domain": domain,
        "source_type": source_type,
        "mode": mode,
        "item_id": item_id,
        "audio_path": audio_path,
        "sr": sr,
        "n_samples": n,
        "duration_s": n / sr,
        "raw_rms": raw_rms,
        "raw_rms_db": safe_amp_db(raw_rms),
        "raw_peak_abs": raw_peak,
        "rms": rms,
        "rms_db": safe_amp_db(rms),
        "peak_abs": peak,
        "crest_factor": crest,
        "zcr": zcr,
        "total_psd_power": total_power,
        "total_psd_power_db": safe_db(total_power),
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_flatness": spectral_flatness,
    }

    for key, value in bands.items():
        result[f"{key}_power"] = float(value)
        result[f"{key}_db"] = safe_db(value)
        result[f"{key}_ratio"] = float(value / total_power)

    # Aliases useful for reading the CSV.
    result["heart_band_20_500_ratio"] = result["energy_20_500_ratio"]
    result["heart_band_20_80_ratio"] = result["energy_20_80_ratio"]
    result["hs_band_20_80_ratio"] = result["energy_20_80_ratio"]
    result["hs_band_80_150_ratio"] = result["energy_80_150_ratio"]
    result["hs_band_150_500_ratio"] = result["energy_150_500_ratio"]

    result["lung_band_50_1800_ratio"] = result["energy_50_1800_ratio"]
    result["ls_band_20_50_ratio"] = result["energy_20_50_ratio"]
    result["ls_band_50_150_ratio"] = result["energy_50_150_ratio"]
    result["ls_band_150_500_ratio"] = result["energy_150_500_ratio"]
    result["ls_band_500_1000_ratio"] = result["energy_500_1000_ratio"]
    result["ls_band_1000_1800_ratio"] = result["energy_1000_1800_ratio"]
    result["hf_noise_ratio_1000_2000"] = result["energy_1000_2000_ratio"]

    return result


def collect_source_level_items(
    df: pd.DataFrame,
    source_type: str,
    domain: str,
    max_sources: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if source_type == "HS":
        id_col = "hs_source_id"
        preferred_path_cols = ["hs_source_path", "source_h", "h_path"]
    elif source_type == "LS":
        id_col = "ls_source_id"
        preferred_path_cols = ["ls_source_path", "source_l", "l_path"]
    else:
        raise ValueError(source_type)

    if id_col not in df.columns:
        raise ValueError(f"Missing column {id_col}")

    rows = []

    # One row per source id. Prefer true standalone path if available.
    for sid, g in df.groupby(id_col):
        g = g.copy()

        audio_path = None
        selected_row = None

        for _, row in g.iterrows():
            p = pick_existing_path(row, preferred_path_cols)
            if p is not None:
                audio_path = p
                selected_row = row
                break

        if audio_path is None:
            continue

        rows.append({
            "domain": domain,
            "source_type": source_type,
            "source_id": str(sid),
            "audio_path": audio_path,
            "n_rows_in_manifest": len(g),
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise RuntimeError(f"No source-level audio paths found for {domain} {source_type}")

    if max_sources is not None and max_sources > 0 and len(out) > max_sources:
        out = out.sample(n=max_sources, random_state=seed).reset_index(drop=True)

    return out


def extract_source_features(
    items: pd.DataFrame,
    modes: List[str],
    expected_sr: int,
) -> pd.DataFrame:
    records = []

    for idx, row in items.iterrows():
        path = row["audio_path"]
        x, sr = read_audio(path, expected_sr=expected_sr)

        for mode in modes:
            rec = extract_features_from_signal(
                x=x,
                sr=sr,
                mode=mode,
                source_type=row["source_type"],
                domain=row["domain"],
                item_id=row["source_id"],
                audio_path=path,
            )
            rec["n_rows_in_manifest"] = row["n_rows_in_manifest"]
            records.append(rec)

    return pd.DataFrame(records)


def extract_segment_features(
    df: pd.DataFrame,
    source_type: str,
    domain: str,
    modes: List[str],
    expected_sr: int,
    max_segments: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if source_type == "HS":
        path_col = "h_path"
        source_id_col = "hs_source_id"
        class_col_candidates = ["hs_class_name", "hs_target_class", "hs_label_name"]
    elif source_type == "LS":
        path_col = "l_path"
        source_id_col = "ls_source_id"
        class_col_candidates = ["ls_class_name", "ls_target_class", "ls_lung_class"]
    else:
        raise ValueError(source_type)

    if path_col not in df.columns:
        raise ValueError(f"Missing {path_col}")

    sampled = sample_rows(df, max_segments, seed)
    records = []

    for idx, row in sampled.iterrows():
        path = str(row[path_col])
        if not Path(path).exists():
            continue

        x, sr = read_audio(path, expected_sr=expected_sr)

        item_id = str(row.get("name", f"row_{idx}"))

        for mode in modes:
            rec = extract_features_from_signal(
                x=x,
                sr=sr,
                mode=mode,
                source_type=source_type,
                domain=domain,
                item_id=item_id,
                audio_path=path,
            )

            rec["base_id"] = row.get("base_id", np.nan)
            rec["segment_index"] = row.get("segment_index", np.nan)
            rec["snr_label"] = row.get("snr_label", np.nan)
            rec["effective_snr_db"] = row.get("effective_snr_db", np.nan)
            rec["source_id"] = row.get(source_id_col, "")

            class_value = ""
            for c in class_col_candidates:
                if c in row.index and not pd.isna(row[c]):
                    class_value = row[c]
                    break
            rec["class_name"] = class_value

            records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Gap summary
# ---------------------------------------------------------------------

def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "sr", "n_samples", "duration_s",
        "base_id", "segment_index",
        "snr_label", "effective_snr_db",
    }

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def compute_gap_summary(
    features: pd.DataFrame,
    domain_source: str,
    domain_target: str,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = numeric_feature_columns(features)

    rows = []

    grouped = features.groupby(["source_type", "mode"], dropna=False)

    for (source_type, mode), g in grouped:
        src = g[g["domain"] == domain_source]
        tgt = g[g["domain"] == domain_target]

        if len(src) == 0 or len(tgt) == 0:
            continue

        for feat in feature_cols:
            s = pd.to_numeric(src[feat], errors="coerce").dropna()
            t = pd.to_numeric(tgt[feat], errors="coerce").dropna()

            if len(s) < 2 or len(t) < 2:
                continue

            mean_s = float(s.mean())
            mean_t = float(t.mean())
            std_s = float(s.std(ddof=1))
            std_t = float(t.std(ddof=1))
            pooled = math.sqrt(max(((len(s) - 1) * std_s ** 2 + (len(t) - 1) * std_t ** 2) / max(len(s) + len(t) - 2, 1), EPS))
            smd = (mean_t - mean_s) / pooled

            rows.append({
                "source_type": source_type,
                "mode": mode,
                "feature": feat,
                "source_domain": domain_source,
                "target_domain": domain_target,
                "source_n": len(s),
                "target_n": len(t),
                "source_mean": mean_s,
                "target_mean": mean_t,
                "source_std": std_s,
                "target_std": std_t,
                "diff_target_minus_source": mean_t - mean_s,
                "smd_target_minus_source": smd,
                "abs_smd": abs(smd),
                "source_median": float(s.median()),
                "target_median": float(t.median()),
                "source_q25": float(s.quantile(0.25)),
                "source_q75": float(s.quantile(0.75)),
                "target_q25": float(t.quantile(0.25)),
                "target_q75": float(t.quantile(0.75)),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["abs_smd"], ascending=False).reset_index(drop=True)
    return out


def save_basic_distribution_summary(features: pd.DataFrame, out_path: Path) -> None:
    cols = numeric_feature_columns(features)
    rows = []

    for (domain, source_type, mode), g in features.groupby(["domain", "source_type", "mode"], dropna=False):
        for feat in cols:
            x = pd.to_numeric(g[feat], errors="coerce").dropna()
            if len(x) == 0:
                continue
            rows.append({
                "domain": domain,
                "source_type": source_type,
                "mode": mode,
                "feature": feat,
                "n": len(x),
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "median": float(x.median()),
                "q25": float(x.quantile(0.25)),
                "q75": float(x.quantile(0.75)),
                "min": float(x.min()),
                "max": float(x.max()),
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------
# PCA plots
# ---------------------------------------------------------------------

def make_pca_plots(features: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if plt is None or PCA is None or StandardScaler is None:
        print("[WARN] matplotlib/sklearn not available, skipping PCA plots.")
        return

    plot_features = [
        "raw_rms_db",
        "peak_abs",
        "crest_factor",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_flatness",
        "energy_20_500_ratio",
        "energy_50_1800_ratio",
        "energy_20_50_ratio",
        "energy_150_500_ratio",
        "energy_500_1000_ratio",
        "energy_1000_1800_ratio",
        "hf_noise_ratio_1000_2000",
    ]

    available = [c for c in plot_features if c in features.columns]

    for source_type in sorted(features["source_type"].dropna().unique()):
        for mode in sorted(features["mode"].dropna().unique()):
            g = features[(features["source_type"] == source_type) & (features["mode"] == mode)].copy()
            if len(g) < 4:
                continue

            X = g[available].apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

            if len(X) < 4:
                continue

            meta = g.loc[X.index].copy()

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)

            pca = PCA(n_components=2)
            Z = pca.fit_transform(Xs)

            proj = pd.DataFrame({
                "pc1": Z[:, 0],
                "pc2": Z[:, 1],
                "domain": meta["domain"].values,
                "source_type": source_type,
                "mode": mode,
                "item_id": meta["item_id"].values,
                "audio_path": meta["audio_path"].values,
            })

            proj_path = out_dir / f"{prefix}_pca_{source_type}_{mode}.csv"
            proj.to_csv(proj_path, index=False)

            plt.figure(figsize=(8, 6))
            for domain in sorted(proj["domain"].unique()):
                d = proj[proj["domain"] == domain]
                plt.scatter(d["pc1"], d["pc2"], s=18, alpha=0.65, label=domain)

            evr = pca.explained_variance_ratio_
            plt.title(f"{prefix} PCA - {source_type} - {mode} | EVR {evr[0]:.2f}, {evr[1]:.2f}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_pca_{source_type}_{mode}.png", dpi=160)
            plt.close()


# ---------------------------------------------------------------------
# Local SNR audit
# ---------------------------------------------------------------------

def classify_local_snr(x: float) -> str:
    if x >= 20:
        return "HS_extremely_dominant"
    if x >= 15:
        return "HS_dominant"
    if x <= -20:
        return "LS_extremely_dominant"
    if x <= -15:
        return "LS_dominant"
    return "balanced_or_moderate"


def local_snr_audit(
    df: pd.DataFrame,
    out_dir: Path,
    expected_sr: int,
    max_segments: Optional[int],
    seed: int,
) -> pd.DataFrame:
    sampled = sample_rows(df, max_segments, seed)

    records = []

    for idx, row in sampled.iterrows():
        hp = str(row["h_path"])
        lp = str(row["l_path"])
        mp = str(row["m_path"])

        if not Path(hp).exists() or not Path(lp).exists():
            continue

        h, sr_h = read_audio(hp, expected_sr=expected_sr)
        l, sr_l = read_audio(lp, expected_sr=expected_sr)

        h = h - np.mean(h)
        l = l - np.mean(l)

        rms_h = float(np.sqrt(np.mean(h ** 2) + EPS))
        rms_l = float(np.sqrt(np.mean(l ** 2) + EPS))

        local_snr = 20.0 * math.log10(max(rms_h, EPS) / max(rms_l, EPS))

        rec = {
            "name": row.get("name", f"row_{idx}"),
            "base_id": row.get("base_id", np.nan),
            "segment_index": row.get("segment_index", np.nan),
            "snr_label": row.get("snr_label", np.nan),
            "effective_snr_db": row.get("effective_snr_db", np.nan),
            "hs_source_id": row.get("hs_source_id", ""),
            "ls_source_id": row.get("ls_source_id", ""),
            "hs_class": row.get("hs_class_name", row.get("hs_target_class", "")),
            "ls_class": row.get("ls_class_name", row.get("ls_target_class", "")),
            "rms_h": rms_h,
            "rms_l": rms_l,
            "local_snr_h_over_l_db": local_snr,
            "abs_local_snr_db": abs(local_snr),
            "observability_bin": classify_local_snr(local_snr),
            "h_path": hp,
            "l_path": lp,
            "m_path": mp,
        }
        records.append(rec)

    out = pd.DataFrame(records)
    out.to_csv(out_dir / "torabi_local_snr_segments.csv", index=False)

    if len(out) == 0:
        return out

    summary_rows = []

    def add_summary(group_cols: List[str], name: str):
        g = out.groupby(group_cols, dropna=False)
        for keys, d in g:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"summary_type": name, "n": len(d)}
            for c, v in zip(group_cols, keys):
                row[c] = v
            row.update({
                "mean_local_snr_h_over_l_db": float(d["local_snr_h_over_l_db"].mean()),
                "median_local_snr_h_over_l_db": float(d["local_snr_h_over_l_db"].median()),
                "mean_abs_local_snr_db": float(d["abs_local_snr_db"].mean()),
                "pct_hs_dominant_15db": float((d["local_snr_h_over_l_db"] >= 15).mean() * 100),
                "pct_ls_dominant_15db": float((d["local_snr_h_over_l_db"] <= -15).mean() * 100),
                "pct_extreme_abs_20db": float((d["abs_local_snr_db"] >= 20).mean() * 100),
            })
            summary_rows.append(row)

    add_summary(["snr_label"], "by_snr")
    add_summary(["ls_class"], "by_ls_class")
    add_summary(["hs_class"], "by_hs_class")
    add_summary(["snr_label", "ls_class"], "by_snr_and_ls_class")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "torabi_local_snr_summary.csv", index=False)

    if plt is not None:
        plt.figure(figsize=(8, 5))
        plt.hist(out["local_snr_h_over_l_db"].dropna(), bins=60)
        plt.axvline(-15, linestyle="--")
        plt.axvline(15, linestyle="--")
        plt.title("Torabi local SNR distribution: 20log10(RMS_H/RMS_L)")
        plt.xlabel("local SNR H/L [dB]")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "torabi_local_snr_hist.png", dpi=160)
        plt.close()

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    source_manifest: pd.DataFrame,
    target_manifest: pd.DataFrame,
    source_items: pd.DataFrame,
    target_items: pd.DataFrame,
) -> None:
    readme = f"""# Domain gap audit: EXP_H vs Torabi fold

## Source domain
{args.source_dir}

split = {args.source_split}

segments selected in manifest = {len(source_manifest)}

unique HS sources = {source_manifest['hs_source_id'].nunique() if 'hs_source_id' in source_manifest.columns else 'NA'}
unique LS sources = {source_manifest['ls_source_id'].nunique() if 'ls_source_id' in source_manifest.columns else 'NA'}

## Target domain
{args.target_dir}

split = {args.target_split}

segments selected in manifest = {len(target_manifest)}

unique HS sources = {target_manifest['hs_source_id'].nunique() if 'hs_source_id' in target_manifest.columns else 'NA'}
unique LS sources = {target_manifest['ls_source_id'].nunique() if 'ls_source_id' in target_manifest.columns else 'NA'}

## Feature extraction

Modes:
{args.normalizations}

Source-level items extracted:
source = {len(source_items)}
target = {len(target_items)}

Segment max per domain/source type:
{args.max_segments_per_domain}

## Main output files

- source_level_features.csv
- source_level_gap_summary.csv
- source_level_top_domain_gaps.csv
- source_level_distribution_summary.csv
- segment_level_features_sampled.csv
- segment_level_gap_summary.csv
- segment_level_top_domain_gaps.csv
- torabi_local_snr_segments.csv
- torabi_local_snr_summary.csv
- PCA .csv and .png files

## Interpretation guide

- `smd_target_minus_source > 0`: Torabi has higher mean value than EXP_H.
- `smd_target_minus_source < 0`: Torabi has lower mean value than EXP_H.
- `abs_smd` ranks the strongest domain gaps.
- Compare `raw` and `rms` modes:
  - large raw gap but small rms gap suggests mainly gain/amplitude mismatch;
  - persistent rms gap suggests spectral/morphological mismatch.
- For LS, inspect especially:
  - `energy_20_500_ratio`
  - `ls_band_20_50_ratio`
  - `ls_band_50_150_ratio`
  - `crest_factor`
  - `hf_noise_ratio_1000_2000`
"""
    (out_dir / "README_domain_gap_audit.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--target-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--source-domain-name", type=str, default="EXP_H")
    parser.add_argument("--target-domain-name", type=str, default="TORABI_FOLD2")

    parser.add_argument("--source-split", type=str, default="train")
    parser.add_argument("--target-split", type=str, default="train")
    parser.add_argument("--fold-no", type=int, default=1)

    parser.add_argument("--expected-sr", type=int, default=4000)
    parser.add_argument("--normalizations", type=str, default="raw,rms")

    parser.add_argument("--max-sources-per-domain", type=int, default=0)
    parser.add_argument("--max-segments-per-domain", type=int, default=12000)
    parser.add_argument("--max-local-snr-segments", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    modes = [x.strip() for x in args.normalizations.split(",") if x.strip()]
    max_sources = args.max_sources_per_domain if args.max_sources_per_domain > 0 else None

    print("=" * 100)
    print("DOMAIN GAP AUDIT")
    print("=" * 100)
    print(f"Source dir : {source_dir}")
    print(f"Target dir : {target_dir}")
    print(f"Output dir : {out_dir}")
    print(f"Source split: {args.source_split}")
    print(f"Target split: {args.target_split}")
    print(f"Fold no     : {args.fold_no}")
    print(f"Modes       : {modes}")

    source_manifest = load_filtered_manifest(source_dir, args.source_split, args.fold_no)
    target_manifest = load_filtered_manifest(target_dir, args.target_split, args.fold_no)

    print("\nLoaded manifests:")
    print(f"  {args.source_domain_name}: {len(source_manifest)} segment rows")
    print(f"  {args.target_domain_name}: {len(target_manifest)} segment rows")

    print("\nUnique sources:")
    print(f"  {args.source_domain_name} HS: {source_manifest['hs_source_id'].nunique()}")
    print(f"  {args.source_domain_name} LS: {source_manifest['ls_source_id'].nunique()}")
    print(f"  {args.target_domain_name} HS: {target_manifest['hs_source_id'].nunique()}")
    print(f"  {args.target_domain_name} LS: {target_manifest['ls_source_id'].nunique()}")

    # ------------------------------------------------------------
    # Source-level audit
    # ------------------------------------------------------------
    print("\n[1/4] Source-level feature extraction...")

    source_hs_items = collect_source_level_items(source_manifest, "HS", args.source_domain_name, max_sources, args.seed)
    source_ls_items = collect_source_level_items(source_manifest, "LS", args.source_domain_name, max_sources, args.seed)
    target_hs_items = collect_source_level_items(target_manifest, "HS", args.target_domain_name, max_sources, args.seed)
    target_ls_items = collect_source_level_items(target_manifest, "LS", args.target_domain_name, max_sources, args.seed)

    source_items = pd.concat([source_hs_items, source_ls_items], ignore_index=True)
    target_items = pd.concat([target_hs_items, target_ls_items], ignore_index=True)

    source_items.to_csv(out_dir / "source_domain_items.csv", index=False)
    target_items.to_csv(out_dir / "target_domain_items.csv", index=False)

    source_features = extract_source_features(source_items, modes, args.expected_sr)
    target_features = extract_source_features(target_items, modes, args.expected_sr)

    source_level_features = pd.concat([source_features, target_features], ignore_index=True)
    source_level_features.to_csv(out_dir / "source_level_features.csv", index=False)

    source_gap = compute_gap_summary(
        source_level_features,
        domain_source=args.source_domain_name,
        domain_target=args.target_domain_name,
    )
    source_gap.to_csv(out_dir / "source_level_gap_summary.csv", index=False)
    source_gap.head(80).to_csv(out_dir / "source_level_top_domain_gaps.csv", index=False)

    save_basic_distribution_summary(source_level_features, out_dir / "source_level_distribution_summary.csv")
    make_pca_plots(source_level_features, out_dir, "source_level")

    # ------------------------------------------------------------
    # Segment-level sampled audit
    # ------------------------------------------------------------
    print("\n[2/4] Segment-level sampled feature extraction...")

    seg_parts = []
    for source_type in ["HS", "LS"]:
        seg_parts.append(
            extract_segment_features(
                source_manifest,
                source_type=source_type,
                domain=args.source_domain_name,
                modes=modes,
                expected_sr=args.expected_sr,
                max_segments=args.max_segments_per_domain,
                seed=args.seed,
            )
        )
        seg_parts.append(
            extract_segment_features(
                target_manifest,
                source_type=source_type,
                domain=args.target_domain_name,
                modes=modes,
                expected_sr=args.expected_sr,
                max_segments=args.max_segments_per_domain,
                seed=args.seed,
            )
        )

    segment_level_features = pd.concat(seg_parts, ignore_index=True)
    segment_level_features.to_csv(out_dir / "segment_level_features_sampled.csv", index=False)

    segment_gap = compute_gap_summary(
        segment_level_features,
        domain_source=args.source_domain_name,
        domain_target=args.target_domain_name,
    )
    segment_gap.to_csv(out_dir / "segment_level_gap_summary.csv", index=False)
    segment_gap.head(80).to_csv(out_dir / "segment_level_top_domain_gaps.csv", index=False)

    save_basic_distribution_summary(segment_level_features, out_dir / "segment_level_distribution_summary.csv")
    make_pca_plots(segment_level_features, out_dir, "segment_level")

    # ------------------------------------------------------------
    # Local SNR audit on Torabi
    # ------------------------------------------------------------
    print("\n[3/4] Torabi local SNR audit...")

    local_snr_df = local_snr_audit(
        target_manifest,
        out_dir=out_dir,
        expected_sr=args.expected_sr,
        max_segments=args.max_local_snr_segments,
        seed=args.seed,
    )

    # ------------------------------------------------------------
    # README
    # ------------------------------------------------------------
    print("\n[4/4] Writing README and final combined files...")

    all_top = []
    if len(source_gap):
        tmp = source_gap.copy()
        tmp["level"] = "source_level"
        all_top.append(tmp)
    if len(segment_gap):
        tmp = segment_gap.copy()
        tmp["level"] = "segment_level"
        all_top.append(tmp)

    if all_top:
        top_all = pd.concat(all_top, ignore_index=True)
        top_all = top_all.sort_values("abs_smd", ascending=False).reset_index(drop=True)
        top_all.to_csv(out_dir / "top_domain_gaps_all_levels.csv", index=False)

    write_readme(out_dir, args, source_manifest, target_manifest, source_items, target_items)

    print("\nDONE.")
    print(f"Results saved in: {out_dir}")
    print("\nMost important files to inspect first:")
    print(f"  {out_dir / 'source_level_top_domain_gaps.csv'}")
    print(f"  {out_dir / 'segment_level_top_domain_gaps.csv'}")
    print(f"  {out_dir / 'torabi_local_snr_summary.csv'}")
    print(f"  {out_dir / 'top_domain_gaps_all_levels.csv'}")


if __name__ == "__main__":
    main()

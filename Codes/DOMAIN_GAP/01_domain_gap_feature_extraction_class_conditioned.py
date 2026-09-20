#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_domain_gap_feature_extraction.py

Domain-gap audit for processed supervised datasets generated as M/H/L segment triplets.

Main use case:
  1) Re-run the global audit with train+val (split=all) for:
       EXP_H_ALL vs TORABI_FOLD2_ALL
       HFLUNG_OPTIONA_ALL vs TORABI_FOLD2_ALL
  2) Add LS class-conditioned audit at segment-level RMS:
       normal, crackles, wheezes, other_mixed

The script does not train models and does not modify datasets.
It reads manifest_synth_supervised.csv from each processed dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_PCA = True
except Exception:
    HAS_PCA = False

TARGET_SR_DEFAULT = 4000
EPS = 1e-10

KEY_FEATURES_LS = [
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

FEATURE_BANDS = {
    "energy_20_50": (20, 50),
    "energy_20_80": (20, 80),
    "energy_20_500": (20, 500),
    "energy_50_150": (50, 150),
    "energy_50_1800": (50, 1800),
    "energy_80_150": (80, 150),
    "energy_150_500": (150, 500),
    "energy_200_500": (200, 500),
    "energy_500_1000": (500, 1000),
    "energy_1000_1800": (1000, 1800),
    "energy_1800_2000": (1800, 2000),
    "energy_1000_2000": (1000, 2000),
}


def norm_text(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def safe_id(x: object) -> str:
    s = norm_text(x)
    s = re.sub(r"[^A-Za-z0-9_\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def class_key(x: object) -> str:
    s = norm_text(x).lower().strip()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"_+", "_", s)
    return s or "unknown"


def macro_lung_class(x: object) -> str:
    """Map heterogeneous LS labels to coarse classes for class-conditioned audit."""
    s = class_key(x)
    if not s or s == "unknown":
        return "unknown"
    if "normal" in s and "abnormal" not in s:
        return "normal"
    if any(k in s for k in ["crackle", "crackles", "fine", "coarse", "crep", "rales"]):
        return "crackles"
    if any(k in s for k in ["wheeze", "wheezes", "wheezing"]):
        # If an explicit mixed/both label is present, keep it in mixed.
        if any(k in s for k in ["both", "mixed", "crackle_wheeze", "wheeze_crackle"]):
            return "other_mixed"
        return "wheezes"
    if any(k in s for k in ["both", "mixed", "other", "rhonchi", "rhonch", "stridor", "pathological", "probably", "abnormal"]):
        return "other_mixed"
    return "other_mixed"


def macro_heart_class(x: object) -> str:
    s = class_key(x)
    if "normal" in s and "abnormal" not in s:
        return "normal"
    if not s or s == "unknown":
        return "unknown"
    return "abnormal"


def parse_modes(s: str) -> List[str]:
    out = []
    for part in str(s).split(","):
        m = part.strip().lower()
        if m in {"", "none"}:
            continue
        if m not in {"raw", "rms", "peak"}:
            raise ValueError(f"Unsupported normalization mode: {m}")
        if m not in out:
            out.append(m)
    return out or ["raw"]


def load_manifest(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "manifest_synth_supervised.csv"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"m_path", "h_path", "l_path"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Manifest {path} missing columns: {sorted(missing)}")
    if "split" not in df.columns:
        df["split"] = "all"
    return df


def filter_split(df: pd.DataFrame, split: str, fold_no: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    if fold_no is not None and "fold_no" in out.columns:
        out = out[out["fold_no"].astype(str).eq(str(fold_no))].copy()
    split = str(split).lower().strip()
    if split not in {"all", "trainval", "train+val", "train_val"}:
        out = out[out["split"].astype(str).str.lower().eq(split)].copy()
    return out.reset_index(drop=True)


def first_existing_col(df: pd.DataFrame, cols: Iterable[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def get_segment_items(df: pd.DataFrame, domain: str, source_type: str) -> pd.DataFrame:
    assert source_type in {"HS", "LS"}
    if source_type == "HS":
        audio_col = "h_path"
        src_col = first_existing_col(df, ["hs_source_id", "h_source_id", "source_id_h", "source_h_id"])
        cls_col = first_existing_col(df, ["hs_class_name", "h_class_name", "heart_class", "hs_target_class", "target_class_h"])
    else:
        audio_col = "l_path"
        src_col = first_existing_col(df, ["ls_source_id", "l_source_id", "source_id_l", "source_l_id"])
        cls_col = first_existing_col(df, ["ls_class_name", "l_class_name", "lung_class", "ls_target_class", "target_class_l", "crop_lung_class", "target_class"])

    rows = pd.DataFrame({
        "domain": domain,
        "source_type": source_type,
        "audio_path": df[audio_col].astype(str),
        "base_id": df.get("base_id", pd.Series(range(len(df)), index=df.index)).astype(str),
        "segment_index": df.get("segment_index", pd.Series(-1, index=df.index)).astype(int),
        "split": df.get("split", pd.Series("unknown", index=df.index)).astype(str),
    })
    if src_col:
        rows["source_id"] = df[src_col].astype(str)
    else:
        rows["source_id"] = rows["audio_path"].map(lambda p: safe_id(Path(p).stem.split("_s")[0]))
    if cls_col:
        rows["class_name"] = df[cls_col].map(class_key)
    else:
        rows["class_name"] = "unknown"
    if source_type == "LS":
        rows["macro_class"] = rows["class_name"].map(macro_lung_class)
    else:
        rows["macro_class"] = rows["class_name"].map(macro_heart_class)
    rows["item_id"] = rows["base_id"].astype(str) + "_s" + rows["segment_index"].astype(str).str.zfill(3)
    return rows.reset_index(drop=True)


def get_source_items(df: pd.DataFrame, domain: str, source_type: str) -> pd.DataFrame:
    assert source_type in {"HS", "LS"}
    if source_type == "HS":
        full_col = first_existing_col(df, ["source_h", "h_source_path", "source_h_path"])
        seg_col = "h_path"
        src_col = first_existing_col(df, ["hs_source_id", "h_source_id", "source_id_h", "source_h_id"])
        cls_col = first_existing_col(df, ["hs_class_name", "h_class_name", "heart_class", "hs_target_class", "target_class_h"])
    else:
        full_col = first_existing_col(df, ["source_l", "l_source_path", "source_l_path"])
        seg_col = "l_path"
        src_col = first_existing_col(df, ["ls_source_id", "l_source_id", "source_id_l", "source_l_id"])
        cls_col = first_existing_col(df, ["ls_class_name", "l_class_name", "lung_class", "ls_target_class", "target_class_l", "crop_lung_class", "target_class"])

    tmp = df.copy()
    if src_col:
        tmp["_source_id"] = tmp[src_col].astype(str)
    else:
        tmp["_source_id"] = tmp[seg_col].astype(str).map(lambda p: safe_id(Path(p).stem.split("_s")[0]))
    if cls_col:
        tmp["_class_name"] = tmp[cls_col].map(class_key)
    else:
        tmp["_class_name"] = "unknown"

    if full_col and full_col in tmp.columns:
        tmp["_audio_path"] = tmp[full_col].astype(str)
    else:
        # Fallback: first segment per source, not ideal but keeps script usable.
        tmp["_audio_path"] = tmp[seg_col].astype(str)

    grouped = tmp.groupby("_source_id", sort=True).first().reset_index()
    rows = pd.DataFrame({
        "domain": domain,
        "source_type": source_type,
        "source_id": grouped["_source_id"].astype(str),
        "audio_path": grouped["_audio_path"].astype(str),
        "class_name": grouped["_class_name"].astype(str),
        "n_rows_in_manifest": tmp.groupby("_source_id", sort=True).size().values,
    })
    if source_type == "LS":
        rows["macro_class"] = rows["class_name"].map(macro_lung_class)
    else:
        rows["macro_class"] = rows["class_name"].map(macro_heart_class)
    rows["item_id"] = rows["source_id"]
    return rows.reset_index(drop=True)


def sample_segments(items: pd.DataFrame, max_segments_per_domain: int, seed: int) -> pd.DataFrame:
    if max_segments_per_domain <= 0 or len(items) <= max_segments_per_domain:
        return items.reset_index(drop=True)
    # Stratify approximately by source_id to avoid one source dominating.
    rng = np.random.default_rng(seed)
    pieces = []
    per_source = max(1, int(math.ceil(max_segments_per_domain / max(items["source_id"].nunique(), 1))))
    for _, g in items.groupby("source_id", sort=True):
        take = min(len(g), per_source)
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        pieces.append(items.loc[idx])
    out = pd.concat(pieces, ignore_index=True)
    if len(out) > max_segments_per_domain:
        idx = rng.choice(out.index.to_numpy(), size=max_segments_per_domain, replace=False)
        out = out.loc[idx]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_audio(path: Path, expected_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if int(sr) != int(expected_sr):
        raise RuntimeError(f"Unexpected SR for {path}: got {sr}, expected {expected_sr}")
    if len(y) == 0:
        raise RuntimeError(f"Empty audio: {path}")
    y = y - float(np.mean(y))
    return y.astype(np.float32), int(sr)


def apply_norm(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return y.astype(np.float32)
    if mode == "rms":
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + EPS))
        return (y / (rms + EPS)).astype(np.float32)
    if mode == "peak":
        peak = float(np.max(np.abs(y)) + EPS)
        return (y / (peak + EPS)).astype(np.float32)
    raise ValueError(mode)


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def bandpower(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return trapz(psd[mask], freqs[mask])


def zcr(y: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    return float(np.mean(np.signbit(y[1:]) != np.signbit(y[:-1])))


def extract_one(item: pd.Series, mode: str, expected_sr: int) -> Dict[str, object]:
    path = Path(str(item["audio_path"]))
    y_raw, sr = load_audio(path, expected_sr)
    y = apply_norm(y_raw, mode)

    raw_rms = float(np.sqrt(np.mean(y_raw.astype(np.float64) ** 2) + EPS))
    raw_peak = float(np.max(np.abs(y_raw)) + EPS)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + EPS))
    peak = float(np.max(np.abs(y)) + EPS)

    nperseg = min(2048, len(y))
    freqs, psd = welch(y.astype(np.float64), fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    psd = np.asarray(psd, dtype=np.float64) + EPS
    total = trapz(psd, freqs) + EPS
    psd_sum = float(np.sum(psd) + EPS)
    centroid = float(np.sum(freqs * psd) / psd_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))
    flatness = float(np.exp(np.mean(np.log(psd))) / (np.mean(psd) + EPS))

    out: Dict[str, object] = {
        "domain": item.get("domain"),
        "source_type": item.get("source_type"),
        "mode": mode,
        "item_id": item.get("item_id"),
        "audio_path": str(path),
        "source_id": item.get("source_id", "unknown"),
        "class_name": item.get("class_name", "unknown"),
        "macro_class": item.get("macro_class", "unknown"),
        "sr": sr,
        "n_samples": len(y),
        "duration_s": len(y) / sr,
        "raw_rms": raw_rms,
        "raw_rms_db": 20.0 * np.log10(raw_rms + EPS),
        "raw_peak_abs": raw_peak,
        "rms": rms,
        "rms_db": 20.0 * np.log10(rms + EPS),
        "peak_abs": peak,
        "crest_factor": peak / (rms + EPS),
        "zcr": zcr(y),
        "total_psd_power": total,
        "total_psd_power_db": 10.0 * np.log10(total + EPS),
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_flatness": flatness,
    }
    for name, (lo, hi) in FEATURE_BANDS.items():
        p = bandpower(freqs, psd, lo, hi)
        out[f"{name}_power"] = p
        out[f"{name}_db"] = 10.0 * np.log10(p + EPS)
        out[f"{name}_ratio"] = p / total

    out["heart_band_20_500_ratio"] = out["energy_20_500_ratio"]
    out["heart_band_20_80_ratio"] = out["energy_20_80_ratio"]
    out["hs_band_20_80_ratio"] = out["energy_20_80_ratio"]
    out["hs_band_80_150_ratio"] = out["energy_80_150_ratio"]
    out["hs_band_150_500_ratio"] = out["energy_150_500_ratio"]
    out["lung_band_50_1800_ratio"] = out["energy_50_1800_ratio"]
    out["ls_band_20_50_ratio"] = out["energy_20_50_ratio"]
    out["ls_band_50_150_ratio"] = out["energy_50_150_ratio"]
    out["ls_band_150_500_ratio"] = out["energy_150_500_ratio"]
    out["ls_band_500_1000_ratio"] = out["energy_500_1000_ratio"]
    out["ls_band_1000_1800_ratio"] = out["energy_1000_1800_ratio"]
    out["hf_noise_ratio_1000_2000"] = out["energy_1000_2000_ratio"]

    for meta_col in ["base_id", "segment_index", "split", "n_rows_in_manifest"]:
        if meta_col in item:
            out[meta_col] = item.get(meta_col)
    return out


def extract_features(items: pd.DataFrame, modes: List[str], expected_sr: int, desc: str) -> pd.DataFrame:
    rows = []
    for _, item in tqdm(items.iterrows(), total=len(items), desc=desc):
        for mode in modes:
            try:
                rows.append(extract_one(item, mode, expected_sr))
            except Exception as e:
                rows.append({
                    "domain": item.get("domain"),
                    "source_type": item.get("source_type"),
                    "mode": mode,
                    "item_id": item.get("item_id"),
                    "audio_path": item.get("audio_path"),
                    "source_id": item.get("source_id", "unknown"),
                    "class_name": item.get("class_name", "unknown"),
                    "macro_class": item.get("macro_class", "unknown"),
                    "load_error": str(e),
                })
    return pd.DataFrame(rows)


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "sr", "n_samples", "duration_s", "base_id", "segment_index", "n_rows_in_manifest",
    }
    meta = {
        "domain", "source_type", "mode", "item_id", "audio_path", "source_id",
        "class_name", "macro_class", "split", "load_error",
    }
    cols = []
    for c in df.columns:
        if c in blocked or c in meta:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def smd_summary(df: pd.DataFrame, source_domain: str, target_domain: str, group_cols: List[str]) -> pd.DataFrame:
    ok = df.copy()
    if "load_error" in ok.columns:
        ok = ok[ok["load_error"].fillna("").astype(str).eq("")].copy()
    features = numeric_feature_columns(ok)
    rows = []
    for keys, g in ok.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = dict(zip(group_cols, keys))
        a = g[g["domain"].eq(source_domain)]
        b = g[g["domain"].eq(target_domain)]
        if len(a) == 0 or len(b) == 0:
            continue
        for feat in features:
            av = pd.to_numeric(a[feat], errors="coerce").dropna().astype(float)
            bv = pd.to_numeric(b[feat], errors="coerce").dropna().astype(float)
            if len(av) < 2 or len(bv) < 2:
                continue
            ma, mb = float(av.mean()), float(bv.mean())
            sa, sb = float(av.std(ddof=1)), float(bv.std(ddof=1))
            pooled = math.sqrt(((len(av) - 1) * sa * sa + (len(bv) - 1) * sb * sb) / max(len(av) + len(bv) - 2, 1))
            if pooled < EPS:
                smd = (mb - ma) / EPS
            else:
                smd = (mb - ma) / pooled
            rows.append({
                **key_payload,
                "feature": feat,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "source_n": int(len(av)),
                "target_n": int(len(bv)),
                "source_mean": ma,
                "target_mean": mb,
                "source_std": sa,
                "target_std": sb,
                "diff_target_minus_source": mb - ma,
                "smd_target_minus_source": float(smd),
                "abs_smd": abs(float(smd)),
                "source_median": float(av.median()),
                "target_median": float(bv.median()),
                "source_q25": float(av.quantile(0.25)),
                "source_q75": float(av.quantile(0.75)),
                "target_q25": float(bv.quantile(0.25)),
                "target_q75": float(bv.quantile(0.75)),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["abs_smd"], ascending=False).reset_index(drop=True)
    return out


def distribution_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    ok = df.copy()
    if "load_error" in ok.columns:
        ok = ok[ok["load_error"].fillna("").astype(str).eq("")].copy()
    features = numeric_feature_columns(ok)
    rows = []
    for keys, g in ok.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = dict(zip(group_cols, keys))
        for feat in features:
            vals = pd.to_numeric(g[feat], errors="coerce").dropna().astype(float)
            if len(vals) == 0:
                continue
            rows.append({
                **key_payload,
                "feature": feat,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(vals.median()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })
    return pd.DataFrame(rows)


def save_pca(df: pd.DataFrame, out_dir: Path, level: str, source_type: str, mode: str, source_domain: str, target_domain: str) -> None:
    if not HAS_PCA:
        return
    ok = df.copy()
    if "load_error" in ok.columns:
        ok = ok[ok["load_error"].fillna("").astype(str).eq("")].copy()
    ok = ok[(ok["source_type"].eq(source_type)) & (ok["mode"].eq(mode))].copy()
    if len(ok) < 5 or ok["domain"].nunique() < 2:
        return
    cols = [c for c in numeric_feature_columns(ok) if c.endswith("_ratio") or c in ["zcr", "spectral_centroid", "spectral_bandwidth", "spectral_flatness", "crest_factor"]]
    cols = [c for c in cols if ok[c].notna().all()]
    if len(cols) < 2:
        return
    X = StandardScaler().fit_transform(ok[cols].astype(float).values)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    proj = ok[["domain", "source_type", "mode", "source_id", "class_name", "macro_class"]].copy()
    proj["PC1"] = Z[:, 0]
    proj["PC2"] = Z[:, 1]
    proj["PC1_explained_ratio"] = float(pca.explained_variance_ratio_[0])
    proj["PC2_explained_ratio"] = float(pca.explained_variance_ratio_[1])
    csv_path = out_dir / f"{level}_pca_{source_type}_{mode}.csv"
    png_path = out_dir / f"{level}_pca_{source_type}_{mode}.png"
    proj.to_csv(csv_path, index=False)

    plt.figure(figsize=(7, 5))
    for dom, g in proj.groupby("domain"):
        plt.scatter(g["PC1"], g["PC2"], s=12, alpha=0.65, label=dom)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    plt.title(f"{level} {source_type} {mode}: {source_domain} vs {target_domain}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def write_readme(out_dir: Path, args: argparse.Namespace, source_df: pd.DataFrame, target_df: pd.DataFrame, source_seg: pd.DataFrame, target_seg: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# Domain gap audit: {args.source_domain_name} vs {args.target_domain_name}")
    lines.append("")
    lines.append("## Source domain")
    lines.append(str(args.source_dir))
    lines.append(f"split = {args.source_split}")
    lines.append(f"segments selected in manifest = {len(source_df)}")
    lines.append(f"unique HS sources = {get_segment_items(source_df, args.source_domain_name, 'HS')['source_id'].nunique()}")
    lines.append(f"unique LS sources = {get_segment_items(source_df, args.source_domain_name, 'LS')['source_id'].nunique()}")
    lines.append("")
    lines.append("## Target domain")
    lines.append(str(args.target_dir))
    lines.append(f"split = {args.target_split}")
    lines.append(f"segments selected in manifest = {len(target_df)}")
    lines.append(f"unique HS sources = {get_segment_items(target_df, args.target_domain_name, 'HS')['source_id'].nunique()}")
    lines.append(f"unique LS sources = {get_segment_items(target_df, args.target_domain_name, 'LS')['source_id'].nunique()}")
    lines.append("")
    lines.append("## Feature extraction")
    lines.append(f"Modes: {args.normalizations}")
    lines.append(f"Segment max per domain/source type: {args.max_segments_per_domain}")
    lines.append("")
    lines.append("## Class-conditioned LS audit")
    lines.append("Macro classes: normal, crackles, wheezes, other_mixed.")
    lines.append("Use LS segment-level RMS for the thesis interpretation.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("smd_target_minus_source > 0 means target/Torabi has higher mean than source.")
    lines.append("smd_target_minus_source < 0 means target/Torabi has lower mean than source.")
    lines.append("Persistent high |SMD| after RMS normalization indicates spectral/morphological gap, not only gain.")
    (out_dir / "README_domain_gap_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--target-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-domain-name", type=str, required=True)
    p.add_argument("--target-domain-name", type=str, required=True)
    p.add_argument("--source-split", type=str, default="train")
    p.add_argument("--target-split", type=str, default="train")
    p.add_argument("--fold-no", type=int, default=None)
    p.add_argument("--expected-sr", type=int, default=TARGET_SR_DEFAULT)
    p.add_argument("--normalizations", type=str, default="raw,rms")
    p.add_argument("--max-segments-per-domain", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-pca", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = parse_modes(args.normalizations)

    source_manifest_all = load_manifest(args.source_dir)
    target_manifest_all = load_manifest(args.target_dir)
    source_df = filter_split(source_manifest_all, args.source_split, args.fold_no)
    target_df = filter_split(target_manifest_all, args.target_split, args.fold_no)
    if len(source_df) == 0:
        raise RuntimeError("No source rows after split filtering")
    if len(target_df) == 0:
        raise RuntimeError("No target rows after split filtering")

    print("=" * 100)
    print("DOMAIN GAP AUDIT")
    print("=" * 100)
    print(f"Source: {args.source_domain_name} rows={len(source_df)} split={args.source_split}")
    print(f"Target: {args.target_domain_name} rows={len(target_df)} split={args.target_split}")
    print(f"Modes: {modes}")

    # Source-level items.
    source_items = pd.concat([
        get_source_items(source_df, args.source_domain_name, "HS"),
        get_source_items(source_df, args.source_domain_name, "LS"),
    ], ignore_index=True)
    target_items = pd.concat([
        get_source_items(target_df, args.target_domain_name, "HS"),
        get_source_items(target_df, args.target_domain_name, "LS"),
    ], ignore_index=True)
    source_items.to_csv(out_dir / "source_domain_items.csv", index=False)
    target_items.to_csv(out_dir / "target_domain_items.csv", index=False)

    source_level_items = pd.concat([source_items, target_items], ignore_index=True)
    source_level_features = extract_features(source_level_items, modes, args.expected_sr, "Source-level features")
    source_level_features.to_csv(out_dir / "source_level_features.csv", index=False)
    source_level_gap = smd_summary(source_level_features, args.source_domain_name, args.target_domain_name, ["source_type", "mode"])
    source_level_gap.to_csv(out_dir / "source_level_gap_summary.csv", index=False)
    source_level_gap.head(80).to_csv(out_dir / "source_level_top_domain_gaps.csv", index=False)
    source_level_dist = distribution_summary(source_level_features, ["domain", "source_type", "mode"])
    source_level_dist.to_csv(out_dir / "source_level_distribution_summary.csv", index=False)

    # Segment-level items with balanced sampling per domain/source type.
    seg_pieces = []
    for domain_name, df0, seed_offset in [
        (args.source_domain_name, source_df, 0),
        (args.target_domain_name, target_df, 1000),
    ]:
        for stype in ["HS", "LS"]:
            items = get_segment_items(df0, domain_name, stype)
            items = sample_segments(items, args.max_segments_per_domain, args.seed + seed_offset + (0 if stype == "HS" else 1))
            seg_pieces.append(items)
    segment_items = pd.concat(seg_pieces, ignore_index=True)
    segment_level_features = extract_features(segment_items, modes, args.expected_sr, "Segment-level features")
    segment_level_features.to_csv(out_dir / "segment_level_features_sampled.csv", index=False)
    segment_level_gap = smd_summary(segment_level_features, args.source_domain_name, args.target_domain_name, ["source_type", "mode"])
    segment_level_gap.to_csv(out_dir / "segment_level_gap_summary.csv", index=False)
    segment_level_gap.head(80).to_csv(out_dir / "segment_level_top_domain_gaps.csv", index=False)
    segment_level_dist = distribution_summary(segment_level_features, ["domain", "source_type", "mode"])
    segment_level_dist.to_csv(out_dir / "segment_level_distribution_summary.csv", index=False)

    # Class-conditioned LS audit: segment-level only.
    ls_seg = segment_level_features[segment_level_features["source_type"].eq("LS")].copy()
    ls_class_gap = smd_summary(ls_seg, args.source_domain_name, args.target_domain_name, ["source_type", "mode", "macro_class"])
    ls_class_gap.to_csv(out_dir / "class_conditioned_LS_segment_gap_summary.csv", index=False)
    ls_class_dist = distribution_summary(ls_seg, ["domain", "source_type", "mode", "macro_class"])
    ls_class_dist.to_csv(out_dir / "class_conditioned_LS_segment_distribution_summary.csv", index=False)

    # Compact key-feature table for thesis reading.
    key = ls_class_gap[(ls_class_gap["mode"].eq("rms")) & (ls_class_gap["feature"].isin(KEY_FEATURES_LS))].copy()
    if len(key):
        key = key.sort_values(["macro_class", "abs_smd"], ascending=[True, False])
        key.to_csv(out_dir / "class_conditioned_LS_segment_RMS_key_features.csv", index=False)

    # PCA outputs.
    if not args.skip_pca:
        for level, feat in [("source_level", source_level_features), ("segment_level", segment_level_features)]:
            for stype in ["HS", "LS"]:
                for mode in modes:
                    save_pca(feat, out_dir, level, stype, mode, args.source_domain_name, args.target_domain_name)

    # Class distribution tables.
    class_lines = []
    class_lines.append("Segment-level sampled LS macro-class counts")
    class_lines.append(str(pd.crosstab(ls_seg["domain"], [ls_seg["mode"], ls_seg["macro_class"]])))
    class_lines.append("\nSource-level LS macro-class counts")
    src_ls = pd.concat([source_items, target_items], ignore_index=True)
    src_ls = src_ls[src_ls["source_type"].eq("LS")]
    class_lines.append(str(pd.crosstab(src_ls["domain"], src_ls["macro_class"])))
    (out_dir / "class_distribution_summary.txt").write_text("\n".join(class_lines), encoding="utf-8")

    # README and args.
    write_readme(out_dir, args, source_df, target_df, source_items, target_items)
    (out_dir / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    print("\nSaved:", out_dir)
    print("Main files:")
    print(" - segment_level_gap_summary.csv")
    print(" - class_conditioned_LS_segment_gap_summary.csv")
    print(" - class_conditioned_LS_segment_RMS_key_features.csv")


if __name__ == "__main__":
    main()

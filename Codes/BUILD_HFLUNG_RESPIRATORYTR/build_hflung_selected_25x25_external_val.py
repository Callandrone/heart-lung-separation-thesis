#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_hflung_selected_25x25_external_val.py

Build an external synthetic validation set:

    HS PhysioNet selected by quality CSV
    +
    LS HF_Lung_V1 selected from the corrected similarity ranking
    -> controlled additive mixtures M = H + L
    -> shared triplet peak normalization
    -> 2 s / 0.5 s segmentation

Intended use
------------
Evaluation only. Do NOT use this dataset for training or model selection.

Default design
--------------
- 25 HS sources from PhysioNet quality/selected manifest
- 25 LS sources from HF_Lung ranking, normally the first 25 from top50/very-close
- SNR grid: {-6, -3, 0, +3, +6}
- base triplets: 25 x 25 x 5 = 3125
- segments: 3125 x 27 = 84375

Outputs
-------
PROJECT_ROOT/dataset/HFLUNG_SELECTED_25X25_EXTERNAL_VAL_SELECTED
PROJECT_ROOT/dataset/MIX_HFLUNG_SELECTED_25X25_EXTERNAL_VAL_4K
PROJECT_ROOT/dataset/processed/hflung_selected_25x25_external_val

The processed folder is compatible with evaluate_polarity_control.py:
    M_000001_s000_orig.wav
    H_000001_s000_orig.wav
    L_000001_s000_orig.wav
    manifest_synth_supervised.csv
    source_disjoint_split_smoke.csv

Example
-------
python Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_25x25_external_val.py \
  --project-root . \
  --selection top25 \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from math import gcd
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_SR = 4000
DURATION_SEC = 15.0
TARGET_LEN = int(TARGET_SR * DURATION_SEC)
SEG_SECONDS = 2.0
HOP_SECONDS = 0.5
SEG_SAMPLES = int(TARGET_SR * SEG_SECONDS)
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)
EXPECTED_SEGMENTS = 27
PEAK_VALUE = 0.95
WAV_SUBTYPE = "FLOAT"
EPS = 1e-10
DEFAULT_SNRS = [-6.0, -3.0, 0.0, 3.0, 6.0]
AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff"}


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


def parse_snr_values(values: List[str]) -> List[float]:
    out: List[float] = []
    for v in values:
        x = float(str(v).strip())
        if x not in out:
            out.append(x)
    return out


def snr_label(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr).replace(".", "p")


def rms_power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2) + EPS)


def safe_rms(x: np.ndarray) -> float:
    return float(np.sqrt(rms_power(x)))


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def residual_snr_db(target: np.ndarray, estimate: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    err = target - estimate
    return float(10.0 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(err ** 2) + EPS)))


def pick_col(df: pd.DataFrame, candidates: Iterable[str], label: str, required: bool = True) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_to_actual:
            return lower_to_actual[cand.lower()]
    if required:
        raise RuntimeError(f"Cannot find {label}. Tried {list(candidates)}. Available columns: {df.columns.tolist()}")
    return None


def normalize_path_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace("\\\\", "/", regex=False).str.strip()


def resolve_existing_path(value: object, root: Optional[Path] = None) -> Optional[Path]:
    raw = norm_text(value)
    if not raw or raw.lower() == "nan":
        return None
    p = Path(raw)
    if p.exists():
        return p
    if root is not None:
        p2 = root / raw
        if p2.exists():
            return p2
    return p


def load_any_mono(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if int(orig_sr) != int(sr):
        g = gcd(int(sr), int(orig_sr))
        y = resample_poly(y, int(sr) // g, int(orig_sr) // g).astype(np.float32)
    return y.astype(np.float32)


def make_fixed_15s_4k(path: Path) -> Tuple[np.ndarray, bool]:
    y = load_any_mono(path, TARGET_SR)
    y = y - float(np.mean(y))
    if len(y) == 0 or float(np.max(np.abs(y))) < EPS:
        raise ValueError(f"Silent or near-silent file: {path}")
    if len(y) < TARGET_LEN:
        reps = int(np.ceil(TARGET_LEN / max(len(y), 1)))
        y = np.tile(y, reps)[:TARGET_LEN]
        repeated = True
    else:
        y = y[:TARGET_LEN]
        repeated = False
    return y.astype(np.float32), repeated


def write_fixed_copy(src_path: Path, dst_path: Path) -> Tuple[float, bool]:
    y, repeated = make_fixed_15s_4k(src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), y, TARGET_SR, subtype=WAV_SUBTYPE)
    return safe_rms(y), repeated


def match_length(*signals: np.ndarray) -> List[np.ndarray]:
    n = min(len(x) for x in signals)
    return [np.asarray(x[:n], dtype=np.float32) for x in signals]


def make_mixture(h: np.ndarray, l: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """SNR convention: 20log10(RMS_H/RMS_L). Positive SNR means heart stronger."""
    ph = rms_power(h)
    pl = rms_power(l)
    if ph < EPS or pl < EPS:
        raise ValueError("Near-zero source power.")

    alpha = 1.0
    beta = np.sqrt(ph / pl) * (10.0 ** (-float(snr_db) / 20.0))

    H = alpha * h
    L = beta * l
    M = H + L

    shared_peak = max(float(np.max(np.abs(M))), float(np.max(np.abs(H))), float(np.max(np.abs(L))), EPS)
    norm_scale = PEAK_VALUE / shared_peak
    M = (M * norm_scale).astype(np.float32)
    H = (H * norm_scale).astype(np.float32)
    L = (L * norm_scale).astype(np.float32)
    return M, H, L, float(alpha), float(beta), float(norm_scale)


def segment_signal(signal: np.ndarray) -> List[np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    segments = []
    start = 0
    while start + SEG_SAMPLES <= len(signal):
        segments.append(signal[start:start + SEG_SAMPLES].astype(np.float32))
        start += HOP_SAMPLES
    if len(segments) != EXPECTED_SEGMENTS:
        raise RuntimeError(f"Expected {EXPECTED_SEGMENTS} segments, got {len(segments)}")
    return segments


def save_triplet_segment(m_path: Path, h_path: Path, l_path: Path, M: np.ndarray, H: np.ndarray, L: np.ndarray) -> float:
    peak = max(float(np.max(np.abs(M))), float(np.max(np.abs(H))), float(np.max(np.abs(L))), EPS)
    factor = PEAK_VALUE / peak
    M = (M * factor).astype(np.float32)
    H = (H * factor).astype(np.float32)
    L = (L * factor).astype(np.float32)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(m_path), M, TARGET_SR, subtype=WAV_SUBTYPE)
    sf.write(str(h_path), H, TARGET_SR, subtype=WAV_SUBTYPE)
    sf.write(str(l_path), L, TARGET_SR, subtype=WAV_SUBTYPE)
    return float(factor)


def read_hs_sources(args: argparse.Namespace) -> pd.DataFrame:
    # If a quality CSV is explicitly passed, prefer it.
    # Otherwise fall back to the already selected EXP_H manifest when available.
    if args.hs_quality_csv is None and args.hs_selected_csv is not None and Path(args.hs_selected_csv).exists():
        df = pd.read_csv(args.hs_selected_csv)
        path_col = pick_col(df, ["output_path", "fixed_path", "fixed_audio_path", "processed_path", "source_path", "audio_path", "wav_path", "path"], "HS audio path")
        id_col = pick_col(df, ["source_id", "record_id", "file_id", "name", "output_file"], "HS id", required=False)
        class_col = pick_col(df, ["target_class", "class_name", "label", "diagnosis", "record_class"], "HS class", required=False)
        if "split" in df.columns and args.hs_split.lower() != "all":
            df = df[df["split"].astype(str).str.lower().str.strip().eq(args.hs_split.lower())].copy()
        origin = "hs_selected_manifest"
    else:
        if args.hs_quality_csv is None:
            raise RuntimeError("Pass either --hs-selected-csv or --hs-quality-csv")
        df = pd.read_csv(args.hs_quality_csv)
        path_col = pick_col(df, ["output_path", "fixed_path", "fixed_audio_path", "processed_path", "source_path", "audio_path", "wav_path", "path"], "HS audio path")
        id_col = pick_col(df, ["source_id", "record_id", "file_id", "name", "output_file"], "HS id", required=False)
        class_col = pick_col(df, ["target_class", "class_name", "label", "diagnosis", "record_class", "cluster_id"], "HS class", required=False)
        if "passes_hard_quality" in df.columns:
            df = df[df["passes_hard_quality"].astype(str).str.lower().isin(["true", "1", "yes"]) | (df["passes_hard_quality"] == True)].copy()
        if "quality_pass" in df.columns:
            df = df[df["quality_pass"].astype(str).str.lower().isin(["true", "1", "yes"]) | (df["quality_pass"] == True)].copy()
        sort_cols = [c for c in ["quality_score", "quality", "score", "seg_rms_p10", "heart_band_ratio_20_500", "fixed_rms"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
        origin = "hs_quality_csv"

    df = df.copy()
    df["resolved_path"] = [resolve_existing_path(x) for x in df[path_col].tolist()]
    df = df[df["resolved_path"].notna()].copy()
    missing = [str(p) for p in df["resolved_path"].tolist() if not Path(p).exists()]
    if missing:
        raise RuntimeError("HS CSV contains missing paths. First examples:\n" + "\n".join(missing[:10]))

    out_rows = []
    seen = set()
    for i, row in df.iterrows():
        p = Path(row["resolved_path"])
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        sid = norm_text(row.get(id_col)) if id_col else p.stem
        cls = norm_text(row.get(class_col)) if class_col else "unknown"
        out_rows.append({
            "source_id": f"HS_{safe_id(sid)}",
            "original_source_id": sid,
            "target_class": safe_id(cls).lower(),
            "source_path": str(p),
            "source_kind": "hs",
            "selection_origin": origin,
        })
        if len(out_rows) >= int(args.n_hs):
            break

    if len(out_rows) < int(args.n_hs):
        raise RuntimeError(f"Not enough HS sources: need={args.n_hs}, selected={len(out_rows)}")
    return pd.DataFrame(out_rows).reset_index(drop=True)


def infer_file_type(name: str) -> str:
    s = str(name).lower()
    if s.startswith("trunc_"):
        return "trunc"
    if s.startswith("steth_"):
        return "steth"
    return "unknown"


def infer_location(name: str) -> str:
    m = re.search(r"-L(\d+)_", str(name))
    return "L" + m.group(1) if m else "unknown"


def resolve_hflung_path(row: pd.Series, hflung_root: Path) -> Path:
    for c in ["resolved_audio_path", "source_file", "audio_path", "output_path", "processed_path", "source_path", "path"]:
        if c in row.index:
            p = resolve_existing_path(row[c], hflung_root)
            if p is not None and p.exists():
                return p
    if "relative_path" in row.index:
        p = hflung_root / str(row["relative_path"])
        if p.exists():
            return p
    if "source_name" in row.index:
        matches = list(hflung_root.rglob(str(row["source_name"])))
        if matches:
            return matches[0]
    raise RuntimeError(f"Cannot resolve HF_Lung audio path for row with columns={row.index.tolist()}")


def read_hflung_sources(args: argparse.Namespace) -> pd.DataFrame:
    hroot = Path(args.hflung_root)
    rank_path = Path(args.hflung_label_csv) if args.hflung_label_csv and Path(args.hflung_label_csv).exists() else Path(args.hflung_rank_csv)
    if not rank_path.exists():
        raise RuntimeError(f"HF_Lung ranking/label CSV not found: {rank_path}")

    df = pd.read_csv(rank_path)
    if "source_name" not in df.columns:
        name_col = pick_col(df, ["source_name", "file_name", "filename", "audio_file", "original_file"], "HF_Lung source name")
        df["source_name"] = df[name_col].astype(str)

    # Preserve the ranking order generated by the audit.
    df = df.reset_index(drop=False).rename(columns={"index": "ranking_index"})

    selection = args.selection.lower()
    if selection in {"top25", "top50_first25", "top"}:
        sub = df.head(max(int(args.n_ls), 25)).copy()
    elif selection == "top50":
        sub = df.head(50).copy()
    elif selection in {"very_close", "very_close_p95_1"}:
        if "is_very_close_p95_1" in df.columns:
            mask = df["is_very_close_p95_1"].astype(bool)
        elif "frac_inside_selected_icbhi_p95" in df.columns:
            mask = pd.to_numeric(df["frac_inside_selected_icbhi_p95"], errors="coerce") >= 0.999999
        else:
            raise RuntimeError("Cannot select very_close: missing is_very_close_p95_1 or frac_inside_selected_icbhi_p95")
        sub = df[mask].copy()
    elif selection == "close":
        if "similarity_label" not in df.columns:
            raise RuntimeError("Cannot select close: missing similarity_label")
        sub = df[df["similarity_label"].astype(str).eq("close_to_selected_ICBHI")].copy()
    else:
        raise RuntimeError(f"Unknown --selection={args.selection}")

    if args.prefer_trunc:
        sub = sub[sub["source_name"].astype(str).str.lower().str.startswith("trunc_")].copy()

    if args.allowed_locations:
        allowed = {x.strip().upper() for x in args.allowed_locations.split(",") if x.strip()}
        sub["_location_tmp"] = sub["source_name"].apply(infer_location)
        sub = sub[sub["_location_tmp"].astype(str).str.upper().isin(allowed)].copy()

    if len(sub) < int(args.n_ls):
        raise RuntimeError(f"Not enough HF_Lung LS sources after filters: need={args.n_ls}, available={len(sub)}")

    out_rows = []
    seen = set()
    for _, row in sub.iterrows():
        p = resolve_hflung_path(row, hroot)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        source_name = norm_text(row.get("source_name")) or p.name
        sid = safe_id(Path(source_name).stem)
        file_type = norm_text(row.get("file_type")) or infer_file_type(source_name)
        location = norm_text(row.get("location")) or infer_location(source_name)
        condition_group = norm_text(row.get("condition_group")) or norm_text(row.get("detected_labels")) or "unknown"
        out = {
            "source_id": f"LS_HFLUNG_{safe_id(sid)}",
            "original_source_id": sid,
            "target_class": "hflung_selected",
            "source_path": str(p),
            "source_kind": "ls",
            "selection_origin": "hflung_similarity_audit",
            "source_name": source_name,
            "ranking_index": int(row.get("ranking_index", len(out_rows))),
            "similarity_label": norm_text(row.get("similarity_label")),
            "frac_inside_selected_icbhi_p95": row.get("frac_inside_selected_icbhi_p95", np.nan),
            "median_centroid_dist": row.get("median_centroid_dist", np.nan),
            "file_type": file_type,
            "location": location,
            "condition_group": condition_group,
            "has_any_adventitious": row.get("has_any_adventitious", np.nan),
            "has_wheeze": row.get("has_wheeze", np.nan),
            "has_rhonchi": row.get("has_rhonchi", np.nan),
            "has_stridor": row.get("has_stridor", np.nan),
            "has_crackle": row.get("has_crackle", np.nan),
        }
        out_rows.append(out)
        if len(out_rows) >= int(args.n_ls):
            break

    if len(out_rows) < int(args.n_ls):
        raise RuntimeError(f"Not enough unique HF_Lung LS sources: need={args.n_ls}, selected={len(out_rows)}")

    return pd.DataFrame(out_rows).reset_index(drop=True)


def write_selected_sources(hs_df: pd.DataFrame, ls_df: pd.DataFrame, selected_root: Path, overwrite: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected_root.exists() and overwrite:
        shutil.rmtree(selected_root)
    hs_dir = selected_root / "HS_PHYSIONET"
    ls_dir = selected_root / "LS_HFLUNG_SELECTED"
    hs_dir.mkdir(parents=True, exist_ok=True)
    ls_dir.mkdir(parents=True, exist_ok=True)

    hs_rows = []
    for i, (_, row) in enumerate(tqdm(hs_df.iterrows(), total=len(hs_df), desc="Writing fixed PhysioNet HS"), start=1):
        src = Path(str(row["source_path"]))
        out_name = f"hs_physionet_{i:03d}_{safe_id(row['original_source_id'])}.wav"
        out_path = hs_dir / out_name
        fixed_rms, repeated = write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "split": "val",
            "output_file": out_name,
            "output_path": str(out_path),
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(repeated),
        })
        hs_rows.append(d)

    ls_rows = []
    for i, (_, row) in enumerate(tqdm(ls_df.iterrows(), total=len(ls_df), desc="Writing fixed HF_Lung LS"), start=1):
        src = Path(str(row["source_path"]))
        out_name = f"ls_hflung_{i:03d}_{safe_id(row['original_source_id'])}.wav"
        out_path = ls_dir / out_name
        fixed_rms, repeated = write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "split": "val",
            "output_file": out_name,
            "output_path": str(out_path),
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(repeated),
        })
        ls_rows.append(d)

    hs_manifest = pd.DataFrame(hs_rows)
    ls_manifest = pd.DataFrame(ls_rows)
    hs_manifest.to_csv(selected_root / "selected_hs_physionet_25.csv", index=False)
    ls_manifest.to_csv(selected_root / "selected_ls_hflung_25.csv", index=False)
    pd.concat([hs_manifest.assign(modality="HS"), ls_manifest.assign(modality="LS")], ignore_index=True, sort=False).to_csv(
        selected_root / "selected_sources_combined.csv", index=False
    )
    return hs_manifest, ls_manifest


def create_mixtures(hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame, mix_root: Path, snrs: List[float], overwrite: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mix_root.exists() and overwrite:
        shutil.rmtree(mix_root)
    mix_dir = mix_root / "Mix"
    mix_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    case_id = 1

    h_cache: Dict[str, np.ndarray] = {}
    l_cache: Dict[str, np.ndarray] = {}

    for _, h_row in tqdm(hs_manifest.iterrows(), total=len(hs_manifest), desc="Creating HF_Lung external mixtures"):
        h_path = str(h_row["output_path"])
        if h_path not in h_cache:
            h_cache[h_path] = make_fixed_15s_4k(Path(h_path))[0]
        h = h_cache[h_path]
        for _, l_row in ls_manifest.iterrows():
            l_path = str(l_row["output_path"])
            if l_path not in l_cache:
                l_cache[l_path] = make_fixed_15s_4k(Path(l_path))[0]
            l = l_cache[l_path]
            for snr_db in snrs:
                M, H, L, alpha, beta, norm_scale = make_mixture(h, l, snr_db)
                base_id = f"{case_id:06d}"
                m_name = f"M_{base_id}.wav"
                h_name = f"H_{base_id}.wav"
                l_name = f"L_{base_id}.wav"
                sf.write(str(mix_dir / m_name), M, TARGET_SR, subtype=WAV_SUBTYPE)
                sf.write(str(mix_dir / h_name), H, TARGET_SR, subtype=WAV_SUBTYPE)
                sf.write(str(mix_dir / l_name), L, TARGET_SR, subtype=WAV_SUBTYPE)

                row = {
                    "base_id": base_id,
                    "case_id": int(case_id),
                    "split": "val",
                    "m_file": m_name,
                    "h_file": h_name,
                    "l_file": l_name,
                    "hs_source_id": str(h_row["source_id"]),
                    "ls_source_id": str(l_row["source_id"]),
                    "hs_source_path": h_row.get("source_path"),
                    "ls_source_path": l_row.get("source_path"),
                    "hs_target_class": h_row.get("target_class"),
                    "ls_target_class": l_row.get("target_class"),
                    "hflung_source_name": l_row.get("source_name"),
                    "hflung_file_type": l_row.get("file_type"),
                    "hflung_location": l_row.get("location"),
                    "hflung_similarity_label": l_row.get("similarity_label"),
                    "hflung_frac_inside_selected_icbhi_p95": l_row.get("frac_inside_selected_icbhi_p95"),
                    "hflung_condition_group": l_row.get("condition_group"),
                    "hflung_has_any_adventitious": l_row.get("has_any_adventitious"),
                    "snr_label": snr_label(snr_db),
                    "effective_snr_db": float(snr_db),
                    "alpha": alpha,
                    "beta": beta,
                    "norm_scale": norm_scale,
                    "corr_m_h_plus_l": corrcoef_safe(M, H + L),
                    "residual_snr_m_vs_h_plus_l_db": residual_snr_db(M, H + L),
                    "rms_h_target": safe_rms(H),
                    "rms_l_target": safe_rms(L),
                    "rms_mixture": safe_rms(M),
                }
                rows.append(row)
                split_rows.append({
                    "fold_no": 1,
                    "base_id": base_id,
                    "split": "val",
                    "hs_source_id": str(h_row["source_id"]),
                    "ls_source_id": str(l_row["source_id"]),
                    "snr_label": snr_label(snr_db),
                    "effective_snr_db": float(snr_db),
                })
                case_id += 1

    metadata = pd.DataFrame(rows)
    split_df = pd.DataFrame(split_rows)
    metadata.to_csv(mix_root / "metadata.csv", index=False)
    split_df.to_csv(mix_root / "source_disjoint_split_smoke.csv", index=False)
    return metadata, split_df


def discover_full_triplets(mix_root: Path) -> List[Dict[str, Path]]:
    mix_dir = mix_root / "Mix"
    rows = []
    for m_path in sorted(mix_dir.glob("M_*.wav")):
        base_id = m_path.stem.replace("M_", "")
        h_path = mix_dir / f"H_{base_id}.wav"
        l_path = mix_dir / f"L_{base_id}.wav"
        if not h_path.exists() or not l_path.exists():
            raise RuntimeError(f"Incomplete triplet for base_id={base_id}")
        rows.append({"base_id": base_id, "M": m_path, "H": h_path, "L": l_path})
    if not rows:
        raise RuntimeError(f"No full triplets found in {mix_dir}")
    return rows


def segment_processed_dataset(mix_root: Path, processed_dir: Path, overwrite: bool) -> pd.DataFrame:
    if processed_dir.exists() and overwrite:
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(mix_root / "metadata.csv")
    metadata["base_id"] = metadata["base_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    meta_by_base = metadata.set_index("base_id", drop=False)
    triplets = discover_full_triplets(mix_root)

    rows: List[Dict[str, object]] = []
    for case in tqdm(triplets, desc=f"Segmenting {processed_dir.name}"):
        base_id = str(case["base_id"]).zfill(6)
        meta = meta_by_base.loc[base_id]
        M_full = load_any_mono(case["M"], TARGET_SR)
        H_full = load_any_mono(case["H"], TARGET_SR)
        L_full = load_any_mono(case["L"], TARGET_SR)
        M_full, H_full, L_full = match_length(M_full, H_full, L_full)
        full_add = residual_snr_db(M_full, H_full + L_full)
        M_segs = segment_signal(M_full)
        H_segs = segment_signal(H_full)
        L_segs = segment_signal(L_full)
        for seg_idx, (M_seg, H_seg, L_seg) in enumerate(zip(M_segs, H_segs, L_segs)):
            name = f"{base_id}_s{seg_idx:03d}_orig"
            m_out = processed_dir / f"M_{name}.wav"
            h_out = processed_dir / f"H_{name}.wav"
            l_out = processed_dir / f"L_{name}.wav"
            factor = save_triplet_segment(m_out, h_out, l_out, M_seg, H_seg, L_seg)
            row = {
                "base_id": base_id,
                "segment_index": int(seg_idx),
                "name": name,
                "split": meta.get("split"),
                "m_path": str(m_out),
                "h_path": str(h_out),
                "l_path": str(l_out),
                "source_m": str(case["M"]),
                "source_h": str(case["H"]),
                "source_l": str(case["L"]),
                "segment_samples": SEG_SAMPLES,
                "hop_samples": HOP_SAMPLES,
                "shared_save_factor": factor,
                "full_additivity_snr_db": full_add,
                "segment_additivity_snr_db": residual_snr_db(M_seg, H_seg + L_seg),
            }
            for col in metadata.columns:
                if col not in row:
                    row[col] = meta.get(col)
            rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(processed_dir / "manifest_synth_supervised.csv", index=False)
    shutil.copy2(mix_root / "source_disjoint_split_smoke.csv", processed_dir / "source_disjoint_split_smoke.csv")
    return manifest


def write_summary(args: argparse.Namespace, selected_root: Path, mix_root: Path, processed_dir: Path, hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame, metadata: pd.DataFrame, manifest: pd.DataFrame) -> None:
    expected_bases = int(args.n_hs) * int(args.n_ls) * len(args.snrs)
    expected_segments = expected_bases * EXPECTED_SEGMENTS

    actual = {
        "hs_sources": int(len(hs_manifest)),
        "ls_sources": int(len(ls_manifest)),
        "base_triplets": int(len(metadata)),
        "segments": int(len(manifest)),
        "expected_base_triplets": expected_bases,
        "expected_segments": expected_segments,
        "snrs": list(args.snrs),
        "processed_dir": str(processed_dir),
        "split_csv": str(processed_dir / "source_disjoint_split_smoke.csv"),
        "selected_root": str(selected_root),
        "mix_root": str(mix_root),
        "min_full_additivity_snr_db": float(metadata["residual_snr_m_vs_h_plus_l_db"].min()),
        "min_segment_additivity_snr_db": float(manifest["segment_additivity_snr_db"].min()),
    }

    if actual["base_triplets"] != expected_bases:
        raise RuntimeError(f"Base triplet count mismatch: expected={expected_bases}, actual={actual['base_triplets']}")
    if actual["segments"] != expected_segments:
        raise RuntimeError(f"Segment count mismatch: expected={expected_segments}, actual={actual['segments']}")
    if actual["min_full_additivity_snr_db"] < 50.0:
        raise RuntimeError(f"Full additivity below 50 dB: {actual['min_full_additivity_snr_db']}")
    if actual["min_segment_additivity_snr_db"] < 50.0:
        raise RuntimeError(f"Segment additivity below 50 dB: {actual['min_segment_additivity_snr_db']}")

    lines = []
    lines.append("HF_Lung selected 25x25 external validation")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose: external synthetic validation only, not training.")
    lines.append("Mixture law: M = H + L with SNR scaling and shared triplet peak normalization.")
    lines.append("HS side: PhysioNet selected by quality/selected CSV.")
    lines.append("LS side: HF_Lung selected from corrected similarity ranking vs EXP_H/ICBHI.")
    lines.append("")
    lines.append(json.dumps(actual, indent=2, sort_keys=True))
    lines.append("")
    lines.append("HF_Lung selected file type distribution:")
    lines.append(str(ls_manifest["file_type"].value_counts(dropna=False)))
    lines.append("")
    lines.append("HF_Lung selected location distribution:")
    lines.append(str(ls_manifest["location"].value_counts(dropna=False)))
    if "has_any_adventitious" in ls_manifest.columns:
        lines.append("")
        lines.append("HF_Lung selected adventitious distribution:")
        lines.append(str(ls_manifest["has_any_adventitious"].value_counts(dropna=False)))
    lines.append("")
    lines.append("model_config.py evaluation reminder:")
    lines.append(f'EXPERIMENT_NAME = "eval_hflung_selected_25x25_mixed_ssl"')
    lines.append(f'SUPERVISED_DIR = PROJECT_ROOT + "/dataset/processed/{processed_dir.name}"')
    lines.append('SYNTH_SUPERVISED_DIR = SUPERVISED_DIR')
    lines.append('USE_SOURCE_DISJOINT_SPLIT = True')
    lines.append('SOURCE_DISJOINT_SPLIT_CSV = SUPERVISED_DIR + "/source_disjoint_split_smoke.csv"')
    lines.append('N_FOLDS = 1')
    lines.append('ONLY_FOLD = 1')
    lines.append('EVAL_CKPT = CKPT_DIR + "/finetune_fold1_best.pt"  # replace with Mixed NOAUG + SSL fold/checkpoint')

    for p in [selected_root / "summary.txt", mix_root / "summary.txt", processed_dir / "summary.txt"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
    for p in [selected_root / "summary.json", mix_root / "summary.json", processed_dir / "summary.json"]:
        p.write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root)
    out_upper = safe_id(args.out_name).upper()

    selected_root = project_root / "dataset" / f"{out_upper}_SELECTED"
    mix_root = project_root / "dataset" / f"MIX_{out_upper}_4K"
    processed_dir = project_root / "dataset" / "processed" / safe_id(args.out_name).lower()

    print("=" * 100)
    print("BUILDING HF_LUNG SELECTED 25x25 EXTERNAL VALIDATION")
    print("=" * 100)
    print(f"Project root     : {project_root}")
    print(f"HS quality CSV   : {args.hs_quality_csv}")
    print(f"HS selected CSV  : {args.hs_selected_csv}")
    print(f"HF_Lung rank CSV : {args.hflung_rank_csv}")
    print(f"HF_Lung label CSV: {args.hflung_label_csv}")
    print(f"HF_Lung root     : {args.hflung_root}")
    print(f"Selection        : {args.selection}")
    print(f"Prefer trunc     : {args.prefer_trunc}")
    print(f"Allowed locations: {args.allowed_locations}")
    print(f"n_hs/n_ls        : {args.n_hs}/{args.n_ls}")
    print(f"SNRs             : {args.snrs}")
    print(f"Processed dir    : {processed_dir}")

    hs_df = read_hs_sources(args)
    ls_df = read_hflung_sources(args)

    print("\nSelected HS sources:")
    print(hs_df[["source_id", "target_class", "source_path"]].head(10).to_string(index=False))
    print("\nSelected HF_Lung LS sources:")
    print(ls_df[["source_id", "file_type", "location", "similarity_label", "source_path"]].head(25).to_string(index=False))

    hs_manifest, ls_manifest = write_selected_sources(hs_df, ls_df, selected_root, bool(args.overwrite))
    metadata, _split = create_mixtures(hs_manifest, ls_manifest, mix_root, args.snrs, bool(args.overwrite))
    manifest = segment_processed_dataset(mix_root, processed_dir, bool(args.overwrite))
    write_summary(args, selected_root, mix_root, processed_dir, hs_manifest, ls_manifest, metadata, manifest)

    index = {
        "processed_dir": str(processed_dir),
        "selected_root": str(selected_root),
        "mix_root": str(mix_root),
        "n_hs": int(args.n_hs),
        "n_ls": int(args.n_ls),
        "snrs": list(args.snrs),
        "base_triplets": int(len(metadata)),
        "segments": int(len(manifest)),
    }
    index_path = project_root / "outputs" / f"{safe_id(args.out_name)}_dataset_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([index]).to_csv(index_path, index=False)

    print("\n" + "=" * 100)
    print("DATASET CREATED")
    print("=" * 100)
    print(f"Selected root : {selected_root}")
    print(f"Mix root      : {mix_root}")
    print(f"Processed dir : {processed_dir}")
    print(f"Summary       : {processed_dir / 'summary.txt'}")
    print(f"Index         : {index_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    p.add_argument("--hs-quality-csv", type=Path, default=None, help="PhysioNet candidate quality CSV. Used if --hs-selected-csv is missing/unavailable.")
    p.add_argument("--hs-selected-csv", type=Path, default=None)
    p.add_argument("--hs-split", type=str, default="all", choices=["all", "train", "val"], help="Filter HS selected manifest by split. Ignored for quality CSV unless split column exists and selected CSV is used.")

    p.add_argument("--hflung-rank-csv", type=Path, default=None)
    p.add_argument("--hflung-label-csv", type=Path, default=None)
    p.add_argument("--hflung-root", type=Path, default=None)

    p.add_argument("--selection", type=str, default="top25", choices=["top25", "top50_first25", "top50", "very_close", "very_close_p95_1", "close"])
    p.add_argument("--prefer-trunc", action="store_true", help="Keep only trunc_ HF_Lung files before taking n_ls.")
    p.add_argument("--allowed-locations", type=str, default="", help="Optional comma list, e.g. L1,L2,L5,L6,L8")

    p.add_argument("--n-hs", type=int, default=25)
    p.add_argument("--n-ls", type=int, default=25)
    p.add_argument("--snrs", nargs="+", default=[str(int(x)) for x in DEFAULT_SNRS])
    p.add_argument("--out-name", type=str, default="hflung_selected_25x25_external_val")
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()

    project_root = Path(args.project_root)

    if args.hs_selected_csv is None:
        args.hs_selected_csv = (
            project_root
            / "dataset"
            / "EXP_H_FULL_BOTH_SELECTED"
            / "selected_hs_EXP_H_FULL_BOTH.csv"
        )

    if args.hflung_rank_csv is None:
        args.hflung_rank_csv = (
            project_root
            / "outputs"
            / "domain_gap"
            / "HFLUNG_trend_vs_SELECTED_ICBHI"
            / "hflung_files_ranked_by_similarity_to_selected_icbhi.csv"
        )

    if args.hflung_label_csv is None:
        args.hflung_label_csv = (
            project_root
            / "outputs"
            / "domain_gap"
            / "HFLUNG_SELECTED_LABEL_SUMMARY"
            / "hflung_selected_files_with_labels.csv"
        )

    if args.hflung_root is None:
        args.hflung_root = project_root / "dataset" / "raw" / "HF_Lung_V1"

    args.snrs = parse_snr_values(args.snrs)
    return args


def main() -> None:
    args = parse_args()
    build(args)


if __name__ == "__main__":
    main()

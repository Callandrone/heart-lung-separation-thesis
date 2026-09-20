"""Shared utilities for constructing the controlled EXP_G / EXP_H benchmarks.

These functions are extracted from the experimental dataset-building code used
for the thesis and preserve its waveform processing, additive-mixture
construction, source-disjoint checks, and 2 s / 0.5 s segmentation protocol.
"""

from __future__ import annotations

import shutil
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

TARGET_SR = 4000
DURATION_SEC = 15.0
TARGET_LEN = int(TARGET_SR * DURATION_SEC)
SEG_SECONDS = 2.0
HOP_SECONDS = 0.5
SEG_SAMPLES = int(TARGET_SR * SEG_SECONDS)
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)
EXPECTED_SEGMENTS = 27
PEAK_VALUE = 0.95
EPS = 1e-10

def safe_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + EPS))

def load_any_mono(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if orig_sr != sr:
        g = gcd(sr, int(orig_sr))
        y = resample_poly(y, sr // g, int(orig_sr) // g).astype(np.float32)
    return y

def make_fixed_15s_4k(path: Path) -> Tuple[np.ndarray, bool]:
    """Mono, 4 kHz, DC-removed, fixed 15 s. No standalone peak normalization."""
    y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = y - np.mean(y)

    if len(y) == 0 or float(np.max(np.abs(y))) < EPS:
        raise ValueError(f"Silent or near-silent file after loading: {path}")

    if len(y) < TARGET_LEN:
        reps = int(np.ceil(TARGET_LEN / max(len(y), 1)))
        y = np.tile(y, reps)[:TARGET_LEN]
        was_repeated = True
    else:
        y = y[:TARGET_LEN]
        was_repeated = False

    return y.astype(np.float32), was_repeated

def write_fixed_copy(src_path: Path, dst_path: Path) -> Tuple[float, bool]:
    y, was_repeated = make_fixed_15s_4k(src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), y, TARGET_SR)
    return safe_rms(y), was_repeated

def load_mono_4k_fixed(path: Path) -> np.ndarray:
    y = load_any_mono(path, TARGET_SR)
    y = y - np.mean(y)
    if len(y) < TARGET_LEN:
        reps = int(np.ceil(TARGET_LEN / max(len(y), 1)))
        y = np.tile(y, reps)[:TARGET_LEN]
    else:
        y = y[:TARGET_LEN]
    if float(np.max(np.abs(y))) < EPS:
        raise ValueError(f"Near-silent file: {path}")
    return y.astype(np.float32)

def rms_power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))

def make_mixture(h: np.ndarray, l: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    SNR convention:
        SNR_HL = 10 * log10(P_H / P_L)

    positive SNR => heart stronger
    negative SNR => lung stronger
    """
    ph = rms_power(h)
    pl = rms_power(l)
    if ph < EPS or pl < EPS:
        raise ValueError("Near-zero source power.")

    alpha = 1.0
    beta = np.sqrt(ph / pl) * (10.0 ** (-float(snr_db) / 20.0))

    h_scaled = alpha * h
    l_scaled = beta * l
    m = h_scaled + l_scaled

    shared_peak = max(
        float(np.max(np.abs(m))),
        float(np.max(np.abs(h_scaled))),
        float(np.max(np.abs(l_scaled))),
        EPS,
    )
    norm_scale = PEAK_VALUE / shared_peak

    m = (m * norm_scale).astype(np.float32)
    h_scaled = (h_scaled * norm_scale).astype(np.float32)
    l_scaled = (l_scaled * norm_scale).astype(np.float32)

    return m, h_scaled, l_scaled, float(alpha), float(beta), float(norm_scale)

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

def match_length(*signals: np.ndarray) -> List[np.ndarray]:
    n = min(len(x) for x in signals)
    return [np.asarray(x[:n], dtype=np.float32) for x in signals]

def segment_signal(signal: np.ndarray) -> List[np.ndarray]:
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
    sf.write(str(m_path), M, TARGET_SR)
    sf.write(str(h_path), H, TARGET_SR)
    sf.write(str(l_path), L, TARGET_SR)
    return float(factor)

def snr_label(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr).replace(".", "p")

def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {name}: {sorted(missing)}")

def class_key(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip().lower().replace(" ", "_")

def check_source_disjoint(hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame) -> Dict[str, object]:
    for name, df in [("HS", hs_manifest), ("LS", ls_manifest)]:
        require_columns(df, ["source_id", "source_path", "split"], f"{name} manifest")

    hs_train = set(hs_manifest.loc[hs_manifest["split"].eq("train"), "source_path"].astype(str))
    hs_val = set(hs_manifest.loc[hs_manifest["split"].eq("val"), "source_path"].astype(str))
    ls_train = set(ls_manifest.loc[ls_manifest["split"].eq("train"), "source_path"].astype(str))
    ls_val = set(ls_manifest.loc[ls_manifest["split"].eq("val"), "source_path"].astype(str))

    h_overlap = hs_train & hs_val
    l_overlap = ls_train & ls_val
    if h_overlap:
        raise RuntimeError(f"HS source_path leakage: {sorted(h_overlap)[:5]}")
    if l_overlap:
        raise RuntimeError(f"LS source_path leakage: {sorted(l_overlap)[:5]}")

    patient_overlap = set()
    session_overlap = set()
    if "patient_id" in ls_manifest.columns:
        p_train = set(ls_manifest.loc[ls_manifest["split"].eq("train"), "patient_id"].dropna().astype(str))
        p_val = set(ls_manifest.loc[ls_manifest["split"].eq("val"), "patient_id"].dropna().astype(str))
        patient_overlap = p_train & p_val
    if "session_id" in ls_manifest.columns:
        s_train = set(ls_manifest.loc[ls_manifest["split"].eq("train"), "session_id"].dropna().astype(str))
        s_val = set(ls_manifest.loc[ls_manifest["split"].eq("val"), "session_id"].dropna().astype(str))
        session_overlap = s_train & s_val

    return {
        "hs_train_sources": len(hs_train),
        "hs_val_sources": len(hs_val),
        "hs_source_overlap": len(h_overlap),
        "ls_train_sources": len(ls_train),
        "ls_val_sources": len(ls_val),
        "ls_source_overlap": len(l_overlap),
        "icbhi_patient_overlap": len(patient_overlap),
        "icbhi_session_overlap": len(session_overlap),
    }

def get_row_value(row: pd.Series, col: str):
    if col not in row.index:
        return None
    value = row[col]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value

def create_mixtures_for_cluster(
    cluster_id: int,
    hs_manifest: pd.DataFrame,
    ls_manifest: pd.DataFrame,
    mix_root: Path,
    snrs: List[float],
    overwrite: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mix_root.exists() and overwrite:
        print(f"Deleting previous mix root: {mix_root}")
        shutil.rmtree(mix_root)
    mix_dir = mix_root / "Mix"
    mix_dir.mkdir(parents=True, exist_ok=True)

    hs_by_split = {s: hs_manifest[hs_manifest["split"].eq(s)].copy().reset_index(drop=True) for s in ["train", "val"]}
    ls_by_split = {s: ls_manifest[ls_manifest["split"].eq(s)].copy().reset_index(drop=True) for s in ["train", "val"]}

    expected_train = len(hs_by_split["train"]) * len(ls_by_split["train"]) * len(snrs)
    expected_val = len(hs_by_split["val"]) * len(ls_by_split["val"]) * len(snrs)
    print(f"Cluster {cluster_id}: expected train base triplets = {expected_train}")
    print(f"Cluster {cluster_id}: expected val base triplets   = {expected_val}")

    rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    case_id = 1

    for split in ["train", "val"]:
        hdf = hs_by_split[split]
        ldf = ls_by_split[split]
        for _, h_row in tqdm(hdf.iterrows(), total=len(hdf), desc=f"Cluster {cluster_id} mixtures HS {split}"):
            h = load_mono_4k_fixed(Path(str(h_row["output_path"])))
            for _, l_row in ldf.iterrows():
                l = load_mono_4k_fixed(Path(str(l_row["output_path"])))
                for snr_db in snrs:
                    m, h_tgt, l_tgt, alpha, beta, norm_scale = make_mixture(h, l, snr_db)
                    base_id = f"{case_id:06d}"
                    m_name = f"M_{base_id}.wav"
                    h_name = f"H_{base_id}.wav"
                    l_name = f"L_{base_id}.wav"
                    sf.write(str(mix_dir / m_name), m, TARGET_SR)
                    sf.write(str(mix_dir / h_name), h_tgt, TARGET_SR)
                    sf.write(str(mix_dir / l_name), l_tgt, TARGET_SR)

                    add = h_tgt + l_tgt
                    slabel = snr_label(snr_db)
                    ls_class = get_row_value(l_row, "crop_lung_class") or get_row_value(l_row, "target_class") or "unknown"
                    row = {
                        "base_id": base_id,
                        "case_id": int(case_id),
                        "split": split,
                        "m_file": m_name,
                        "h_file": h_name,
                        "l_file": l_name,
                        "hs_source_id": str(h_row["source_id"]),
                        "ls_source_id": str(l_row["source_id"]),
                        "hs_output_file": h_row.get("output_file"),
                        "ls_output_file": l_row.get("output_file"),
                        "hs_source_path": h_row.get("source_path"),
                        "ls_source_path": l_row.get("source_path"),
                        "hs_record_id": get_row_value(h_row, "record_id"),
                        "ls_record_id": get_row_value(l_row, "record_id"),
                        "hs_label_name": get_row_value(h_row, "label_name"),
                        "hs_target_class": get_row_value(h_row, "target_class"),
                        "hs_cluster_id": get_row_value(h_row, "cluster_id"),
                        "hs_cluster_acoustic_profile": get_row_value(h_row, "cluster_acoustic_profile"),
                        "ls_lung_class": class_key(ls_class),
                        "ls_target_class": class_key(get_row_value(l_row, "target_class") or ls_class),
                        "ls_patient_id": get_row_value(l_row, "patient_id"),
                        "ls_session_id": get_row_value(l_row, "session_id"),
                        "ls_device": get_row_value(l_row, "device"),
                        "snr_label": slabel,
                        "effective_snr_db": float(snr_db),
                        "alpha": alpha,
                        "beta": beta,
                        "norm_scale": norm_scale,
                        "corr_m_h_plus_l": corrcoef_safe(m, add),
                        "residual_snr_m_vs_h_plus_l_db": residual_snr_db(m, add),
                    }
                    rows.append(row)

                    split_rows.append({
                        "fold_no": 1,
                        "base_id": base_id,
                        "split": split,
                        "hs_source_id": str(h_row["source_id"]),
                        "ls_source_id": str(l_row["source_id"]),
                        "hs_label_name": get_row_value(h_row, "label_name"),
                        "hs_target_class": get_row_value(h_row, "target_class"),
                        "hs_cluster_id": get_row_value(h_row, "cluster_id"),
                        "hs_cluster_acoustic_profile": get_row_value(h_row, "cluster_acoustic_profile"),
                        "ls_lung_class": class_key(ls_class),
                        "ls_target_class": class_key(get_row_value(l_row, "target_class") or ls_class),
                        "snr_label": slabel,
                    })
                    case_id += 1

    metadata = pd.DataFrame(rows)
    split_df = pd.DataFrame(split_rows)

    metadata_path = mix_root / "metadata.csv"
    split_path = mix_root / "source_disjoint_split_smoke.csv"
    metadata.to_csv(metadata_path, index=False)
    split_df.to_csv(split_path, index=False)

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

def segment_processed_dataset(
    cluster_id: int,
    mix_root: Path,
    processed_dir: Path,
    overwrite: bool,
) -> pd.DataFrame:
    if processed_dir.exists() and overwrite:
        print(f"Deleting previous processed dir: {processed_dir}")
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = mix_root / "metadata.csv"
    split_path = mix_root / "source_disjoint_split_smoke.csv"
    if not metadata_path.exists():
        raise RuntimeError(f"metadata.csv not found: {metadata_path}")
    if not split_path.exists():
        raise RuntimeError(f"source_disjoint_split_smoke.csv not found: {split_path}")

    metadata = pd.read_csv(metadata_path)
    metadata["base_id"] = metadata["base_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    meta_by_base = metadata.set_index("base_id", drop=False)

    triplets = discover_full_triplets(mix_root)
    manifest_rows: List[Dict[str, object]] = []
    seg_add_snr_values: List[float] = []
    full_add_snr_values: List[float] = []

    for case in tqdm(triplets, desc=f"Cluster {cluster_id} segmenting"):
        base_id = str(case["base_id"]).zfill(6)
        if base_id not in meta_by_base.index:
            raise RuntimeError(f"base_id={base_id} missing from metadata.csv")
        meta = meta_by_base.loc[base_id]

        M_full = load_any_mono(case["M"], TARGET_SR)
        H_full = load_any_mono(case["H"], TARGET_SR)
        L_full = load_any_mono(case["L"], TARGET_SR)
        M_full, H_full, L_full = match_length(M_full, H_full, L_full)
        full_add_snr = residual_snr_db(M_full, H_full + L_full)
        full_add_snr_values.append(full_add_snr)

        M_segs = segment_signal(M_full)
        H_segs = segment_signal(H_full)
        L_segs = segment_signal(L_full)

        for seg_idx, (M_seg, H_seg, L_seg) in enumerate(zip(M_segs, H_segs, L_segs)):
            name = f"{base_id}_s{seg_idx:03d}_orig"
            m_out = processed_dir / f"M_{name}.wav"
            h_out = processed_dir / f"H_{name}.wav"
            l_out = processed_dir / f"L_{name}.wav"
            save_factor = save_triplet_segment(m_out, h_out, l_out, M_seg, H_seg, L_seg)
            seg_add_snr = residual_snr_db(M_seg, H_seg + L_seg)
            seg_add_snr_values.append(seg_add_snr)

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
                "shared_save_factor": save_factor,
                "full_additivity_snr_db": full_add_snr,
                "segment_additivity_snr_db": seg_add_snr,
            }
            for col in [
                "hs_source_id", "ls_source_id", "hs_label_name", "hs_target_class",
                "hs_cluster_id", "hs_cluster_acoustic_profile", "ls_lung_class",
                "ls_target_class", "ls_patient_id", "ls_session_id", "ls_device",
                "snr_label", "effective_snr_db",
            ]:
                if col in metadata.columns:
                    row[col] = meta.get(col)
            manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "manifest_synth_supervised.csv"
    manifest.to_csv(manifest_path, index=False)
    shutil.copy2(split_path, processed_dir / "source_disjoint_split_smoke.csv")

    return manifest

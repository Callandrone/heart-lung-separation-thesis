#!/usr/bin/env python3
"""
01_build_hlscmds_folds.py

Build HLS-CMDS standalone-source synthetic datasets using only isolated HS and LS files.
The real paired Mix.zip / 145 mixtures are NOT used.

Main design
-----------
- Source pool: 50 HS standalone + 50 LS standalone.
- Per generated fold:
    train = 40 HS sources x 40 LS sources x SNRs
    val   = 10 HS sources x 10 LS sources x SNRs
- No unused base triplets are written inside each generated processed folder.
- Each processed folder has fold_no=1 in source_disjoint_split_smoke.csv.
  This is deliberate: train_disjoint.py can run it as a clean single-fold dataset.
- If --n-folds 5 is used, the script creates 5 independent processed folders,
  one per source split. Across the 5 folders each source is used once in validation.

Why this instead of generating 50x50 once?
-----------------------------------------
If you generate all 50x50 pairs and require source-disjoint train/val, the cross pairs
train-HS x val-LS and val-HS x train-LS must be marked as unused. Since the goal here
is explicitly "40x40 + 10x10 without unused", the correct construction is fold-specific:
only trainxtrain and valxval pairs are generated for that fold.

Expected counts with 5 SNRs and 2 s / 0.5 s segmentation:
- train base triplets = 40*40*5 = 8,000  -> 216,000 segments
- val base triplets   = 10*10*5 =   500  ->  13,500 segments
- total per fold      = 8,500 base       -> 229,500 segments

Example
-------
python 01_build_hlscmds_folds.py --overwrite
python 01_build_hlscmds_folds.py --n-folds 1 --seed 42 --overwrite
python 01_build_hlscmds_folds.py --hs-root /path/HS --ls-root /path/LS --overwrite
"""

from __future__ import annotations

import argparse
import os
import json
import re
import shutil
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
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
    """SNR convention: 10log10(P_H/P_L). Positive SNR means heart is stronger."""
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


def discover_default_source_root(project_root: Path, modality: str) -> Path:
    names = [modality, modality.upper(), modality.lower()]
    candidates: List[Path] = []
    for n in names:
        candidates.extend([
            project_root / "dataset" / n,
            project_root / "HLS_CMDS_ALIGNED" / n,
            project_root / "dataset" / "HLS_CMDS_ALIGNED" / n,
            project_root / "dataset" / "HLS_CMDS" / n,
            project_root / "dataset" / "HLS-CMDS-main" / n,
            project_root / "dataset" / "HLS-CMDS" / f"{n}.zip",  # reported only; not read directly
        ])
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise RuntimeError(
        f"Could not auto-discover {modality} standalone root. Pass --{modality.lower()}-root explicitly.\n"
        + "Checked:\n" + "\n".join(str(c) for c in candidates)
    )


def infer_class_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        parts = rel.parts
    except Exception:
        parts = path.parts
    # Prefer the immediate parent if files are class-organized. Otherwise use filename prefix.
    if len(parts) >= 2 and path.parent != root:
        return safe_id(parts[-2]).lower()
    stem = path.stem.lower()
    # Remove common standalone prefixes / numbers and keep a readable class proxy.
    stem = re.sub(r"^(hs|ls|heart|lung|standalone)[_\-]*", "", stem)
    stem = re.sub(r"[_\-]*\d+$", "", stem)
    stem = stem or path.parent.name.lower()
    return safe_id(stem).lower()


def discover_sources(root: Path, modality: str, limit: int | None = 50) -> pd.DataFrame:
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"{modality} root not found or not a directory: {root}")
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise RuntimeError(f"No audio files found under {root}")

    rows = []
    for i, p in enumerate(files, start=1):
        cls = infer_class_from_path(p, root)
        rows.append({
            "source_id": f"{modality.lower()}_{i:03d}_{safe_id(p.stem)}",
            "source_kind": modality.lower(),
            "source_path": str(p),
            "original_file": p.name,
            "class_name": cls,
            "target_class": cls,
        })
    df = pd.DataFrame(rows)
    if limit is not None:
        if len(df) < limit:
            raise RuntimeError(f"Need at least {limit} {modality} sources, found {len(df)} in {root}")
        if len(df) > limit:
            print(f"WARNING: found {len(df)} {modality} files; using the first {limit} sorted files. Pass --source-limit 0 to use all.")
            df = df.head(limit).copy()
    return df.reset_index(drop=True)


def make_stratified_folds(df: pd.DataFrame, n_folds: int, seed: int, label: str) -> List[np.ndarray]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if len(df) % n_folds != 0:
        raise RuntimeError(f"{label}: number of sources {len(df)} must be divisible by n_folds={n_folds}")

    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    if "class_name" not in df.columns:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        chunks = np.array_split(idx, n_folds)
        return [np.array(c, dtype=int) for c in chunks]

    for _, g in df.groupby("class_name", sort=True):
        idx = g.index.to_numpy(dtype=int).copy()
        rng.shuffle(idx)
        chunks = np.array_split(idx, n_folds)
        for k, ch in enumerate(chunks):
            folds[k].extend(ch.tolist())

    expected = len(df) // n_folds
    # If class sizes are not perfectly divisible, rebalance deterministically.
    all_idx = set(df.index.to_list())
    used = set(i for fold in folds for i in fold)
    if used != all_idx:
        raise RuntimeError(f"{label}: fold assignment lost indices")

    too_big = [k for k, f in enumerate(folds) if len(f) > expected]
    too_small = [k for k, f in enumerate(folds) if len(f) < expected]
    for small in too_small:
        while len(folds[small]) < expected:
            big = next(k for k in too_big if len(folds[k]) > expected)
            moved = folds[big].pop()
            folds[small].append(moved)
            too_big = [k for k, f in enumerate(folds) if len(f) > expected]

    return [np.array(sorted(f), dtype=int) for f in folds]


def write_selected_sources(hs_all: pd.DataFrame, ls_all: pd.DataFrame, hs_train_idx: np.ndarray, hs_val_idx: np.ndarray, ls_train_idx: np.ndarray, ls_val_idx: np.ndarray, selected_root: Path, overwrite: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected_root.exists() and overwrite:
        shutil.rmtree(selected_root)
    hs_dir = selected_root / "HS_HLSCMDS_STANDALONE"
    ls_dir = selected_root / "LS_HLSCMDS_STANDALONE"
    hs_dir.mkdir(parents=True, exist_ok=True)
    ls_dir.mkdir(parents=True, exist_ok=True)

    def enrich(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> pd.DataFrame:
        out = df.copy()
        out["split"] = "unused"
        out.loc[train_idx, "split"] = "train"
        out.loc[val_idx, "split"] = "val"
        if set(out["split"]) - {"train", "val"}:
            raise RuntimeError("Internal split error: unused sources remained")
        return out.reset_index(drop=True)

    hs_split = enrich(hs_all, hs_train_idx, hs_val_idx)
    ls_split = enrich(ls_all, ls_train_idx, ls_val_idx)

    hs_rows = []
    for i, (_, row) in enumerate(tqdm(hs_split.iterrows(), total=len(hs_split), desc="Writing fixed HLS-CMDS HS"), start=1):
        src = Path(str(row["source_path"]))
        out_name = f"hs_{row['split']}_{i:03d}_{safe_id(row['class_name'])}_{safe_id(src.stem)}.wav"
        out_path = hs_dir / out_name
        fixed_rms, repeated = write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "output_file": out_name,
            "output_path": str(out_path),
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(repeated),
        })
        hs_rows.append(d)

    ls_rows = []
    for i, (_, row) in enumerate(tqdm(ls_split.iterrows(), total=len(ls_split), desc="Writing fixed HLS-CMDS LS"), start=1):
        src = Path(str(row["source_path"]))
        out_name = f"ls_{row['split']}_{i:03d}_{safe_id(row['class_name'])}_{safe_id(src.stem)}.wav"
        out_path = ls_dir / out_name
        fixed_rms, repeated = write_fixed_copy(src, out_path)
        d = row.to_dict()
        d.update({
            "selected_id": i,
            "output_file": out_name,
            "output_path": str(out_path),
            "fixed_rms_after_write": fixed_rms,
            "was_repeated_to_15s_after_write": bool(repeated),
        })
        ls_rows.append(d)

    hs_manifest = pd.DataFrame(hs_rows)
    ls_manifest = pd.DataFrame(ls_rows)
    hs_manifest.to_csv(selected_root / "selected_hs_sources.csv", index=False)
    ls_manifest.to_csv(selected_root / "selected_ls_sources.csv", index=False)
    pd.concat([hs_manifest.assign(modality="HS"), ls_manifest.assign(modality="LS")], ignore_index=True, sort=False).to_csv(
        selected_root / "selected_sources_combined.csv", index=False
    )
    return hs_manifest, ls_manifest


def create_mixtures(hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame, mix_root: Path, snrs: List[float], overwrite: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mix_root.exists() and overwrite:
        shutil.rmtree(mix_root)
    mix_dir = mix_root / "Mix"
    mix_dir.mkdir(parents=True, exist_ok=True)

    hs_by_split = {s: hs_manifest[hs_manifest["split"].eq(s)].copy().reset_index(drop=True) for s in ["train", "val"]}
    ls_by_split = {s: ls_manifest[ls_manifest["split"].eq(s)].copy().reset_index(drop=True) for s in ["train", "val"]}

    rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    case_id = 1

    for split in ["train", "val"]:
        hdf = hs_by_split[split]
        ldf = ls_by_split[split]
        for _, h_row in tqdm(hdf.iterrows(), total=len(hdf), desc=f"Creating {split} HLS-CMDS mixtures"):
            h = make_fixed_15s_4k(Path(str(h_row["output_path"])))[0]
            for _, l_row in ldf.iterrows():
                l = make_fixed_15s_4k(Path(str(l_row["output_path"])))[0]
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
                        "split": split,
                        "m_file": m_name,
                        "h_file": h_name,
                        "l_file": l_name,
                        "hs_source_id": str(h_row["source_id"]),
                        "ls_source_id": str(l_row["source_id"]),
                        "hs_source_path": h_row.get("source_path"),
                        "ls_source_path": l_row.get("source_path"),
                        "hs_target_class": h_row.get("target_class"),
                        "ls_target_class": l_row.get("target_class"),
                        "hs_class_name": h_row.get("class_name"),
                        "ls_class_name": l_row.get("class_name"),
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
                        "split": split,
                        "hs_source_id": str(h_row["source_id"]),
                        "ls_source_id": str(l_row["source_id"]),
                        "hs_target_class": h_row.get("target_class"),
                        "ls_target_class": l_row.get("target_class"),
                        "snr_label": snr_label(snr_db),
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


def validate_no_unused_no_leakage(split_df: pd.DataFrame) -> None:
    if set(split_df["split"].astype(str)) != {"train", "val"}:
        raise RuntimeError(f"Unexpected split labels: {sorted(set(split_df['split'].astype(str)))}")
    train = split_df[split_df["split"].eq("train")]
    val = split_df[split_df["split"].eq("val")]
    hs_overlap = set(train["hs_source_id"].astype(str)) & set(val["hs_source_id"].astype(str))
    ls_overlap = set(train["ls_source_id"].astype(str)) & set(val["ls_source_id"].astype(str))
    if hs_overlap or ls_overlap:
        raise RuntimeError(f"SOURCE LEAKAGE DETECTED: HS={sorted(hs_overlap)} LS={sorted(ls_overlap)}")


def write_summary(fold_no: int, selected_root: Path, mix_root: Path, processed_dir: Path, hs_manifest: pd.DataFrame, ls_manifest: pd.DataFrame, metadata: pd.DataFrame, split_df: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    train = split_df[split_df["split"].eq("train")]
    val = split_df[split_df["split"].eq("val")]
    summary = {
        "fold_generated": int(fold_no),
        "processed_dir": str(processed_dir),
        "split_csv": str(processed_dir / "source_disjoint_split_smoke.csv"),
        "selected_root": str(selected_root),
        "mix_root": str(mix_root),
        "hs_train_sources": int(train["hs_source_id"].nunique()),
        "hs_val_sources": int(val["hs_source_id"].nunique()),
        "ls_train_sources": int(train["ls_source_id"].nunique()),
        "ls_val_sources": int(val["ls_source_id"].nunique()),
        "train_base_triplets": int((metadata["split"] == "train").sum()),
        "val_base_triplets": int((metadata["split"] == "val").sum()),
        "train_segments": int((manifest["split"] == "train").sum()),
        "val_segments": int((manifest["split"] == "val").sum()),
        "snrs": list(args.snrs),
    }
    lines = []
    lines.append("HLS-CMDS standalone full-source 40x40 + 10x10 baseline, no unused")
    lines.append("=" * 100)
    lines.append("Only isolated HS and LS standalone files are used. Mix.zip / 145 real paired mixtures are not used.")
    lines.append("This processed folder is a single clean fold: fold_no=1, split in {train,val}, no unused rows.")
    lines.append("")
    lines.append(json.dumps(summary, indent=2, sort_keys=True))
    lines.append("")
    lines.append("HS source distribution:")
    lines.append(str(pd.crosstab(hs_manifest["target_class"], hs_manifest["split"])))
    lines.append("")
    lines.append("LS source distribution:")
    lines.append(str(pd.crosstab(ls_manifest["target_class"], ls_manifest["split"])))
    lines.append("")
    lines.append("Base triplets:")
    lines.append(str(metadata["split"].value_counts(dropna=False)))
    lines.append("")
    lines.append("Segments:")
    lines.append(str(manifest["split"].value_counts(dropna=False)))
    lines.append("")
    lines.append("Full additivity:")
    lines.append(str(metadata[["corr_m_h_plus_l", "residual_snr_m_vs_h_plus_l_db"]].describe().T))
    lines.append("")
    lines.append("Segment additivity:")
    lines.append(str(manifest[["segment_additivity_snr_db"]].describe().T))
    lines.append("")
    lines.append("model_config.py reminder:")
    lines.append(f'SUPERVISED_DIR = PROJECT_ROOT + "/dataset/processed/{processed_dir.name}"')
    lines.append("USE_SOURCE_DISJOINT_SPLIT = True")
    lines.append('SOURCE_DISJOINT_SPLIT_CSV = SUPERVISED_DIR + "/source_disjoint_split_smoke.csv"')
    lines.append("N_FOLDS = 1")
    lines.append("ONLY_FOLD = 1")

    for p in [selected_root / "summary.txt", mix_root / "summary.txt", processed_dir / "summary.txt"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
    for p in [selected_root / "summary.json", mix_root / "summary.json", processed_dir / "summary.json"]:
        p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_one_fold(args: argparse.Namespace, fold_idx: int, hs_all: pd.DataFrame, ls_all: pd.DataFrame, hs_folds: List[np.ndarray], ls_folds: List[np.ndarray]) -> Dict[str, object]:
    fold_no = fold_idx + 1
    hs_val_idx = hs_folds[fold_idx]
    ls_val_idx = ls_folds[fold_idx]
    hs_train_idx = np.array(sorted(set(hs_all.index.to_list()) - set(hs_val_idx.tolist())), dtype=int)
    ls_train_idx = np.array(sorted(set(ls_all.index.to_list()) - set(ls_val_idx.tolist())), dtype=int)

    if len(hs_train_idx) != args.train_hs or len(hs_val_idx) != args.val_hs:
        raise RuntimeError(f"HS fold counts mismatch: train={len(hs_train_idx)}, val={len(hs_val_idx)}")
    if len(ls_train_idx) != args.train_ls or len(ls_val_idx) != args.val_ls:
        raise RuntimeError(f"LS fold counts mismatch: train={len(ls_train_idx)}, val={len(ls_val_idx)}")

    out_name = f"{safe_id(args.out_name)}_fold{fold_no}"
    selected_root = args.project_root / "dataset" / f"{out_name.upper()}_SELECTED"
    mix_root = args.project_root / "dataset" / f"MIX_{out_name.upper()}_4K"
    processed_dir = args.project_root / "dataset" / "processed" / out_name

    print("\n" + "=" * 100)
    print(f"BUILDING HLS-CMDS FULL BASELINE FOLD {fold_no}/{args.n_folds}")
    print("=" * 100)
    print(f"Selected root : {selected_root}")
    print(f"Mix root      : {mix_root}")
    print(f"Processed dir : {processed_dir}")
    print(f"HS train/val  : {len(hs_train_idx)} / {len(hs_val_idx)}")
    print(f"LS train/val  : {len(ls_train_idx)} / {len(ls_val_idx)}")

    hs_manifest, ls_manifest = write_selected_sources(hs_all, ls_all, hs_train_idx, hs_val_idx, ls_train_idx, ls_val_idx, selected_root, args.overwrite)
    metadata, split_df = create_mixtures(hs_manifest, ls_manifest, mix_root, args.snrs, args.overwrite)
    manifest = segment_processed_dataset(mix_root, processed_dir, args.overwrite)
    validate_no_unused_no_leakage(split_df)

    expected_train_bases = args.train_hs * args.train_ls * len(args.snrs)
    expected_val_bases = args.val_hs * args.val_ls * len(args.snrs)
    expected_segments = (expected_train_bases + expected_val_bases) * EXPECTED_SEGMENTS
    if len(metadata) != expected_train_bases + expected_val_bases:
        raise RuntimeError(f"Base count mismatch: expected={expected_train_bases + expected_val_bases}, actual={len(metadata)}")
    if len(manifest) != expected_segments:
        raise RuntimeError(f"Segment count mismatch: expected={expected_segments}, actual={len(manifest)}")
    if float(metadata["residual_snr_m_vs_h_plus_l_db"].min()) < 50.0:
        raise RuntimeError("Generated full-length additivity below 50 dB.")
    if float(manifest["segment_additivity_snr_db"].min()) < 50.0:
        raise RuntimeError("Generated segment additivity below 50 dB.")

    return write_summary(fold_no, selected_root, mix_root, processed_dir, hs_manifest, ls_manifest, metadata, split_df, manifest, args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--hs-root", type=Path, default=None)
    p.add_argument("--ls-root", type=Path, default=None)
    p.add_argument("--out-name", type=str, default="hlscmds_full_40x40_10x10_no_unused")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=None, help="Generate only this external fold number, e.g. 1..5.")
    p.add_argument("--train-hs", type=int, default=40)
    p.add_argument("--val-hs", type=int, default=10)
    p.add_argument("--train-ls", type=int, default=40)
    p.add_argument("--val-ls", type=int, default=10)
    p.add_argument("--source-limit", type=int, default=50, help="Use first N sorted standalone sources per modality. Use 0 for all.")
    p.add_argument("--snrs", nargs="+", default=[str(int(x)) for x in DEFAULT_SNRS])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.snrs = parse_snr_values(args.snrs)
    source_limit = None if int(args.source_limit) == 0 else int(args.source_limit)

    hs_root = args.hs_root or discover_default_source_root(args.project_root, "HS")
    ls_root = args.ls_root or discover_default_source_root(args.project_root, "LS")
    print(f"HS root: {hs_root}")
    print(f"LS root: {ls_root}")

    hs_all = discover_sources(hs_root, "hs", limit=source_limit)
    ls_all = discover_sources(ls_root, "ls", limit=source_limit)

    expected_hs_total = args.train_hs + args.val_hs
    expected_ls_total = args.train_ls + args.val_ls
    if len(hs_all) != expected_hs_total:
        raise RuntimeError(f"Expected {expected_hs_total} HS sources for {args.train_hs}+{args.val_hs}, got {len(hs_all)}")
    if len(ls_all) != expected_ls_total:
        raise RuntimeError(f"Expected {expected_ls_total} LS sources for {args.train_ls}+{args.val_ls}, got {len(ls_all)}")

    hs_folds = make_stratified_folds(hs_all, args.n_folds, args.seed + 11, "HS")
    ls_folds = make_stratified_folds(ls_all, args.n_folds, args.seed + 29, "LS")

    if len(hs_folds[0]) != args.val_hs or len(ls_folds[0]) != args.val_ls:
        raise RuntimeError("Fold size is not equal to requested validation source count. Check n-folds and counts.")

    fold_indices = range(args.n_folds)
    if args.only_fold is not None:
        if args.only_fold < 1 or args.only_fold > args.n_folds:
            raise RuntimeError(f"--only-fold must be between 1 and {args.n_folds}")
        fold_indices = [args.only_fold - 1]

    results = []
    for fold_idx in fold_indices:
        results.append(build_one_fold(args, fold_idx, hs_all, ls_all, hs_folds, ls_folds))

    index_path = args.project_root / "outputs" / f"{safe_id(args.out_name)}_dataset_indices.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(index_path, index=False)
    print("\n" + "=" * 100)
    print("HLS-CMDS FULL BASELINE DATASET(S) CREATED")
    print("=" * 100)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nCombined index: {index_path}")


if __name__ == "__main__":
    main()

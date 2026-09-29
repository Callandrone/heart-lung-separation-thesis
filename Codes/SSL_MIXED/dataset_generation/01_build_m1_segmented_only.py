#!/usr/bin/env python3
"""
01_build_m1_segmented_only.py

STEP 1 SSL / V1 real mixtures, M-only.

This script builds the unlabeled M-only dataset used before pseudo-label generation.
It takes the 110 V1 real mixture files from HLS_CMDS_ALIGNED/Mix, i.e. files whose
name starts with M and whose numeric id is in 0001..0110, and creates:

    PROJECT_ROOT/dataset/processed/M1_segmented_only/
        M_<base_id>_s000_orig.wav
        M_<base_id>_s001_orig.wav
        ...
        manifest_m1_segmented_only.csv
        summary.json
        summary.txt

Important methodological choice:
- This is NOT a supervised dataset.
- H/L files are ignored on purpose.
- The output contains only M segments, so it can be used for teacher inference
  and pseudo-label generation.

Preprocessing order, aligned with the HLS-CMDS dataset-generation pipeline:
1. load audio
2. convert to mono
3. resample to 4 kHz
4. remove DC
5. force 15 s duration by crop/tile
6. full-recording peak normalization with one global gain
7. segment into 2 s windows with 0.5 s hop -> 27 segments/file
8. save M-only segments and manifest

Default run:
    python 01_build_m1_segmented_only.py --overwrite

Explicit paths:
    python 01_build_m1_segmented_only.py --mix-root /path/to/HLS_CMDS_ALIGNED/Mix --overwrite

After this step, the next step is teacher pseudo-label generation:
    teacher(M_segment) -> H_pseudo, L_pseudo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


# Repository/project root. Override with ESD_JASSNET_ROOT when datasets live elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(REPO_ROOT))).expanduser().resolve()
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
AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff"}


@dataclass
class AudioStats:
    original_sr: int
    original_samples: int
    mono_samples: int
    resampled_samples: int
    fixed_samples: int
    was_repeated_to_15s: bool
    was_cropped_to_15s: bool
    dc_mean_removed: float
    peak_before_full_norm: float
    full_norm_scale: float
    rms_after_full_norm: float
    peak_after_full_norm: float


def safe_id(x: object) -> str:
    s = "" if x is None else str(x).strip()
    s = re.sub(r"[^A-Za-z0-9_\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def parse_numeric_id(path: Path) -> Optional[int]:
    """Extract the first numeric id after the initial M/m prefix: M0001.wav -> 1."""
    m = re.match(r"^[mM][_\- ]*(\d+)", path.stem)
    if not m:
        return None
    return int(m.group(1))


def is_mixture_file(path: Path) -> bool:
    """Keep only files that are actual mixture files: M0001.wav / M_0001.wav / etc."""
    if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
        return False
    return parse_numeric_id(path) is not None


def discover_default_mix_root(project_root: Path) -> Path:
    candidates = [
        project_root / "HLS_CMDS_ALIGNED" / "Mix",
        project_root / "dataset" / "HLS_CMDS_ALIGNED" / "Mix",
        project_root / "dataset" / "HLS-CMDS_ALIGNED" / "Mix",
        project_root / "dataset" / "HLS_CMDS" / "Mix",
        project_root / "dataset" / "HLS-CMDS" / "Mix",
        project_root / "dataset" / "Torabi" / "Mix",  # legacy local layout
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise RuntimeError(
        "Could not auto-discover the V1 Mix folder. Pass --mix-root explicitly.\n"
        + "Checked:\n"
        + "\n".join(str(c) for c in candidates)
    )


def discover_v1_mixtures(
    mix_root: Path,
    v1_min_index: int,
    v1_max_index: int,
    expected_count: int,
    strict_count: bool,
) -> List[Path]:
    if not mix_root.exists() or not mix_root.is_dir():
        raise RuntimeError(f"Mix root not found or not a directory: {mix_root}")

    all_m = [p for p in sorted(mix_root.rglob("*")) if is_mixture_file(p)]
    selected = []
    for p in all_m:
        idx = parse_numeric_id(p)
        if idx is not None and v1_min_index <= idx <= v1_max_index:
            selected.append(p)

    selected = sorted(selected, key=lambda p: (parse_numeric_id(p) or 10**9, p.name.lower()))

    if strict_count and len(selected) != expected_count:
        preview = "\n".join(f"  - {p}" for p in selected[:20])
        raise RuntimeError(
            f"Expected exactly {expected_count} V1 mixture files in index range "
            f"{v1_min_index:04d}..{v1_max_index:04d}, found {len(selected)}.\n"
            f"Mix root: {mix_root}\n"
            f"First selected files:\n{preview}\n\n"
            "If this is intentional, rerun with --no-strict-count."
        )

    if not selected:
        raise RuntimeError(f"No V1 mixture files found under {mix_root}")

    return selected


def load_audio_mono_resampled(path: Path, target_sr: int) -> Tuple[np.ndarray, int, int, int]:
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
    original_samples = int(y.shape[0])

    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)

    y = np.asarray(y, dtype=np.float32)
    mono_samples = int(len(y))

    if int(orig_sr) != int(target_sr):
        g = gcd(int(target_sr), int(orig_sr))
        y = resample_poly(y, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)

    return y.astype(np.float32), int(orig_sr), original_samples, mono_samples


def preprocess_full_mixture(path: Path, target_sr: int, peak_value: float) -> Tuple[np.ndarray, AudioStats]:
    y, original_sr, original_samples, mono_samples = load_audio_mono_resampled(path, target_sr)
    resampled_samples = int(len(y))

    if len(y) == 0:
        raise ValueError(f"Empty audio file: {path}")

    dc = float(np.mean(y))
    y = y - dc

    if float(np.max(np.abs(y))) < EPS:
        raise ValueError(f"Silent or near-silent file after DC removal: {path}")

    was_repeated = False
    was_cropped = False
    if len(y) < TARGET_LEN:
        reps = int(np.ceil(TARGET_LEN / max(len(y), 1)))
        y = np.tile(y, reps)[:TARGET_LEN]
        was_repeated = True
    elif len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
        was_cropped = True

    peak_before = float(np.max(np.abs(y)))
    full_norm_scale = float(peak_value / max(peak_before, EPS))
    y = (y * full_norm_scale).astype(np.float32)

    stats = AudioStats(
        original_sr=original_sr,
        original_samples=original_samples,
        mono_samples=mono_samples,
        resampled_samples=resampled_samples,
        fixed_samples=int(len(y)),
        was_repeated_to_15s=bool(was_repeated),
        was_cropped_to_15s=bool(was_cropped),
        dc_mean_removed=dc,
        peak_before_full_norm=peak_before,
        full_norm_scale=full_norm_scale,
        rms_after_full_norm=float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + EPS)),
        peak_after_full_norm=float(np.max(np.abs(y))),
    )
    return y, stats


def segment_signal(signal: np.ndarray) -> List[np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    segments: List[np.ndarray] = []
    start = 0
    while start + SEG_SAMPLES <= len(signal):
        segments.append(signal[start : start + SEG_SAMPLES].astype(np.float32))
        start += HOP_SAMPLES
    if len(segments) != EXPECTED_SEGMENTS:
        raise RuntimeError(f"Expected {EXPECTED_SEGMENTS} segments, got {len(segments)}")
    return segments


def maybe_write_segment(
    out_path: Path,
    seg: np.ndarray,
    target_sr: int,
    segment_peak_normalize: bool,
    peak_value: float,
) -> Tuple[float, float, float]:
    seg = np.asarray(seg, dtype=np.float32)
    peak_before = float(np.max(np.abs(seg)))
    save_factor = 1.0

    if segment_peak_normalize:
        save_factor = float(peak_value / max(peak_before, EPS))
        seg = (seg * save_factor).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), seg, target_sr, subtype=WAV_SUBTYPE)

    return (
        peak_before,
        save_factor,
        float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + EPS)),
    )


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    mix_root = args.mix_root or discover_default_mix_root(args.project_root)
    out_dir = args.out_dir or (args.project_root / "dataset" / "processed" / args.out_name)
    full_fixed_dir = out_dir / "full_fixed_15s"

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_fixed_dir.mkdir(parents=True, exist_ok=True)

    v1_files = discover_v1_mixtures(
        mix_root=mix_root,
        v1_min_index=args.v1_min_index,
        v1_max_index=args.v1_max_index,
        expected_count=args.expected_count,
        strict_count=not args.no_strict_count,
    )

    rows: List[Dict[str, object]] = []
    full_rows: List[Dict[str, object]] = []

    for src_path in tqdm(v1_files, desc="Preprocessing V1 real mixtures M-only"):
        numeric_id = parse_numeric_id(src_path)
        if numeric_id is None:
            raise RuntimeError(f"Internal error: selected non-M file {src_path}")

        base_id = f"M{numeric_id:04d}"
        y_full, stats = preprocess_full_mixture(src_path, args.target_sr, args.peak_value)
        stats_dict = asdict(stats)

        full_out = full_fixed_dir / f"{base_id}_fixed15s_4k.wav"
        sf.write(str(full_out), y_full, args.target_sr, subtype=WAV_SUBTYPE)

        full_rows.append(
            {
                "base_id": base_id,
                "numeric_id": int(numeric_id),
                "source_file": src_path.name,
                "source_path": str(src_path),
                "full_fixed_path": str(full_out),
                **stats_dict,
            }
        )

        segments = segment_signal(y_full)
        for seg_idx, seg in enumerate(segments):
            name = f"{base_id}_s{seg_idx:03d}_orig"
            m_out = out_dir / f"M_{name}.wav"
            peak_before, save_factor, rms_after_save = maybe_write_segment(
                out_path=m_out,
                seg=seg,
                target_sr=args.target_sr,
                segment_peak_normalize=args.segment_peak_normalize,
                peak_value=args.peak_value,
            )

            rows.append(
                {
                    "base_id": base_id,
                    "numeric_id": int(numeric_id),
                    "segment_index": int(seg_idx),
                    "name": name,
                    "split": "ssl_unlabeled",
                    "m_path": str(m_out),
                    "source_file": src_path.name,
                    "source_path": str(src_path),
                    "full_fixed_path": str(full_out),
                    "sr": int(args.target_sr),
                    "segment_samples": int(SEG_SAMPLES),
                    "hop_samples": int(HOP_SAMPLES),
                    "segment_seconds": float(SEG_SECONDS),
                    "hop_seconds": float(HOP_SECONDS),
                    "used_for_ssl": True,
                    "has_reference_hl": False,
                    "segment_peak_before_save": peak_before,
                    "segment_save_factor": save_factor,
                    "segment_rms_after_save": rms_after_save,
                    "segment_peak_normalize": bool(args.segment_peak_normalize),
                    **stats_dict,
                }
            )

    manifest = pd.DataFrame(rows)
    full_manifest = pd.DataFrame(full_rows)

    manifest_path = out_dir / "manifest_m1_segmented_only.csv"
    full_manifest_path = out_dir / "manifest_m1_full_fixed.csv"
    manifest.to_csv(manifest_path, index=False)
    full_manifest.to_csv(full_manifest_path, index=False)

    # Small compatibility copy: many project scripts expect a manifest_*.csv in the processed folder.
    manifest.to_csv(out_dir / "manifest_ssl_unlabeled.csv", index=False)

    expected_segments = len(v1_files) * EXPECTED_SEGMENTS
    if len(manifest) != expected_segments:
        raise RuntimeError(f"Segment count mismatch: expected={expected_segments}, actual={len(manifest)}")

    summary: Dict[str, object] = {
        "dataset_name": args.out_name,
        "methodological_role": "SSL unlabeled M-only dataset for teacher pseudo-label generation",
        "mix_root": str(mix_root),
        "out_dir": str(out_dir),
        "manifest_m1_segmented_only": str(manifest_path),
        "manifest_m1_full_fixed": str(full_manifest_path),
        "n_input_mixtures": int(len(v1_files)),
        "n_segments": int(len(manifest)),
        "segments_per_mixture": int(EXPECTED_SEGMENTS),
        "target_sr": int(args.target_sr),
        "duration_sec": float(DURATION_SEC),
        "segment_sec": float(SEG_SECONDS),
        "hop_sec": float(HOP_SECONDS),
        "segment_peak_normalize": bool(args.segment_peak_normalize),
        "peak_value": float(args.peak_value),
        "v1_min_index": int(args.v1_min_index),
        "v1_max_index": int(args.v1_max_index),
        "ignored_h_l_files": True,
        "supervised_targets_available": False,
        "next_step": "Run teacher inference on M_*.wav to generate H_pseudo/L_pseudo, then audit pseudo-label quality.",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    summary_lines = [
        "M1_segmented_only — SSL M-only preprocessing",
        "=" * 88,
        "This folder contains ONLY mixture segments from V1 real mixtures.",
        "No H/L reference is used, copied, or generated at this step.",
        "",
        json.dumps(summary, indent=2, sort_keys=True),
        "",
        "Preprocessing order:",
        "1) load audio",
        "2) mono conversion",
        "3) resample to 4 kHz",
        "4) DC removal",
        "5) crop/tile to 15 s",
        "6) full-recording global peak normalization",
        "7) 2 s / 0.5 s segmentation -> 27 segments",
        "8) save M-only segments",
        "",
        "Sanity counts:",
        str(manifest.groupby("base_id")["segment_index"].count().describe()),
        "",
        "Use for pseudo-label generation:",
        f"M_ONLY_DIR = {out_dir}",
        f"M_ONLY_MANIFEST = {manifest_path}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    readme = """M1_segmented_only
=================

This is the first SSL preprocessing step for V1 real mixtures.

Contents:
- M_*.wav: 2-second mixture-only segments, 4 kHz, 8000 samples.
- manifest_m1_segmented_only.csv: segment-level manifest for teacher inference.
- manifest_m1_full_fixed.csv: full-recording preprocessing manifest.
- full_fixed_15s/: fixed 15 s, 4 kHz, DC-removed, peak-normalized full mixtures.

Important:
- This is not a supervised dataset.
- H/L files are intentionally ignored.
- Use this dataset only to generate pseudo-labels with the teacher checkpoint.
"""
    (out_dir / "README_M1_segmented_only.txt").write_text(readme, encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build M1_segmented_only from V1 real mixtures, M-only.")
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--mix-root", type=Path, default=None, help="Folder containing HLS_CMDS_ALIGNED/Mix files.")
    p.add_argument("--out-name", type=str, default="M1_segmented_only")
    p.add_argument("--out-dir", type=Path, default=None)

    p.add_argument("--target-sr", type=int, default=TARGET_SR)
    p.add_argument("--peak-value", type=float, default=PEAK_VALUE)
    p.add_argument("--v1-min-index", type=int, default=1)
    p.add_argument("--v1-max-index", type=int, default=110)
    p.add_argument("--expected-count", type=int, default=110)
    p.add_argument("--no-strict-count", action="store_true", help="Do not fail if selected M files are not exactly expected-count.")

    p.add_argument(
        "--segment-peak-normalize",
        action="store_true",
        help=(
            "Optional: apply an additional per-segment peak normalization. "
            "Default OFF to preserve the full-recording global dynamics inside the M-only SSL pool."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print("\n" + "=" * 88)
    print("M1 SEGMENTED ONLY DATASET CREATED")
    print("=" * 88)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nNext step: generate teacher pseudo-labels for every M_*.wav in the output folder.")


if __name__ == "__main__":
    main()

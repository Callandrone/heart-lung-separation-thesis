#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_hflung_quality_score_and_selection.py

HF_Lung_V1 quality scoring + balanced source selection for domain-gap audit.

Goal
----
- scan all HF_Lung WAV files;
- infer/attach respiratory class labels;
- convert each candidate to a comparable 15 s mono 4 kHz clip;
- compute quality features and a quality_score similar in spirit to the ICBHI
  candidate quality CSV used in EXP_H;
- select Option A for the audit:

    total selected = 70 LS sources
    normal   = 20  -> train 14, val 6
    crackles = 15  -> train 11, val 4
    wheezes  = 15  -> train 11, val 4
    other    = 20  -> train 14, val 6

Notes
-----
- No mixtures are generated.
- No training split is meant for model training here; train/val-like labels are
  only used to keep a coherent 50 + 20 selected-set structure.
- The processed WAVs are written as FLOAT without peak/RMS normalization, so raw
  amplitude information is preserved for the domain-gap audit. The later audit
  can still compute RMS-normalized features separately.

Example
-------
python Codes/BUILD_HFLUNG_RESPIRATORYTR/hflung_quality_score_and_selection_v2.py \
  --project-root . \
  --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


EPS = 1e-12


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
DEFAULT_RAW_ROOT = DEFAULT_PROJECT_ROOT / "dataset" / "raw" / "HF_Lung_V1"
DEFAULT_OUT_DIR = DEFAULT_PROJECT_ROOT / "outputs" / "hflung_v1_quality_selection"
DEFAULT_PROCESSED_DIR = DEFAULT_PROJECT_ROOT / "dataset" / "processed" / "hflung_v1_selected_70_audit"

TARGET_SEC = 15.0
TARGET_SR = 4000

# Option A selected set. Totals: train=50, val=20, all=70.
OPTION_A_COUNTS = {
    "normal": {"train": 14, "val": 6},
    "crackles": {"train": 11, "val": 4},
    "wheezes": {"train": 11, "val": 4},
    "other": {"train": 14, "val": 6},
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_db_power(x: float) -> float:
    return 10.0 * math.log10(max(float(x), EPS))


def safe_db_amp(x: float) -> float:
    return 20.0 * math.log10(max(float(x), EPS))


def clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(max(x, 0.0), 1.0))


def stable_id_from_path(path: Path, prefix: str = "HFLUNG") -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", path.stem).strip("_")[:40]
    return f"{prefix}_{stem}_{digest}"


def find_wavs(root: Path) -> List[Path]:
    patterns = ["*.wav", "*.WAV", "*.Wave", "*.WAVE"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    # Remove duplicates if filesystem is case-insensitive.
    files = sorted({p.resolve() for p in files})
    return files


# -----------------------------------------------------------------------------
# Metadata / class mapping
# -----------------------------------------------------------------------------

def norm_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def load_optional_metadata(metadata_csv: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    Optional metadata support.

    If you know HF_Lung has a metadata CSV, pass it with --metadata-csv and set
    --metadata-path-col / --metadata-label-col. This function builds a lookup by:
    - exact path string;
    - basename;
    - stem.

    The main script still works without metadata by inferring labels from the
    file path/name.
    """
    if metadata_csv is None:
        return {}
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        row_dict = {str(k): norm_text(v) for k, v in row.to_dict().items()}
        for value in row_dict.values():
            if not value:
                continue
            p = Path(value)
            keys = {value, p.name, p.stem}
            for k in keys:
                if k:
                    lookup.setdefault(k, row_dict)
    return lookup


def metadata_row_for_path(path: Path, metadata_lookup: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    if not metadata_lookup:
        return {}
    candidates = [str(path), path.name, path.stem]
    for c in candidates:
        if c in metadata_lookup:
            return metadata_lookup[c]
    return {}


def infer_raw_label_from_metadata_or_path(
    path: Path,
    metadata_row: Dict[str, str],
    label_col: str = "",
) -> str:
    # Prefer an explicit metadata label column when provided.
    if label_col and metadata_row and label_col in metadata_row and metadata_row[label_col]:
        return metadata_row[label_col]

    # Otherwise look for likely label columns.
    likely_cols = [
        "label", "class", "diagnosis", "sound", "event", "type", "category",
        "target_class", "crop_lung_class", "lung_class", "respiratory_class",
    ]
    for col in likely_cols:
        if col in metadata_row and metadata_row[col]:
            return metadata_row[col]

    # Fallback: use path components and filename.
    return " / ".join([p for p in path.parts[-5:]])


def hflung_label_path_for_wav(path: Path) -> Path:
    """Return the expected HF_Lung event-label path for a WAV file."""
    return path.with_name(f"{path.stem}_label.txt")


def parse_hflung_event_label_file(path: Path) -> Dict[str, object]:
    """
    Parse HF_Lung *_label.txt files.

    Observed format examples:
        I       00:00:04.991 00:00:06.149
        E       00:00:07.540 00:00:08.566
        D       00:00:07.540 00:00:08.566
        WHEEZE  ...
        RHONCHI ...
        STRIDOR ...

    For Option A selection:
        - D only                         -> crackles
        - WHEEZE only                    -> wheezes
        - D + WHEEZE, RHONCHI, STRIDOR   -> other
        - only I/E                       -> normal
    """
    label_path = hflung_label_path_for_wav(path)

    info: Dict[str, object] = {
        "label_path": str(label_path),
        "label_file_exists": bool(label_path.exists()),
        "event_labels_raw": "",
        "event_labels_unique": "",
        "n_events_total": 0,
        "n_events_I": 0,
        "n_events_E": 0,
        "n_events_D": 0,
        "n_events_WHEEZE": 0,
        "n_events_RHONCHI": 0,
        "n_events_STRIDOR": 0,
        "has_inspiration": False,
        "has_expiration": False,
        "has_crackles": False,
        "has_wheezes": False,
        "has_rhonchi": False,
        "has_stridor": False,
        "has_any_adventitious": False,
        "hflung_event_mapped_class": "unknown",
    }

    if not label_path.exists():
        return info

    labels: List[str] = []
    try:
        for line in label_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            lab = line.split()[0].strip().upper()
            if not lab:
                continue
            labels.append(lab)
    except Exception:
        return info

    unique = sorted(set(labels))
    label_set = set(unique)

    info["event_labels_raw"] = ";".join(labels)
    info["event_labels_unique"] = ";".join(unique)
    info["n_events_total"] = int(len(labels))

    for lab in ["I", "E", "D", "WHEEZE", "RHONCHI", "STRIDOR"]:
        info[f"n_events_{lab}"] = int(sum(1 for x in labels if x == lab))

    has_i = "I" in label_set
    has_e = "E" in label_set
    has_d = "D" in label_set or "DAS" in label_set
    has_w = "W" in label_set or "WHEEZE" in label_set or "WHEEZES" in label_set
    has_r = "R" in label_set or "RHONCHI" in label_set or "RHONCHUS" in label_set
    has_s = "S" in label_set or "STRIDOR" in label_set

    info["has_inspiration"] = bool(has_i)
    info["has_expiration"] = bool(has_e)
    info["has_crackles"] = bool(has_d)
    info["has_wheezes"] = bool(has_w)
    info["has_rhonchi"] = bool(has_r)
    info["has_stridor"] = bool(has_s)
    info["has_any_adventitious"] = bool(has_d or has_w or has_r or has_s)

    # Option A mapping. Combined D+W and non-ICBHI classes are intentionally
    # routed to "other".
    if has_r or has_s or (has_d and has_w):
        mapped = "other"
    elif has_d:
        mapped = "crackles"
    elif has_w:
        mapped = "wheezes"
    elif has_i or has_e:
        mapped = "normal"
    else:
        mapped = "unknown"

    info["hflung_event_mapped_class"] = mapped
    return info


def map_hflung_class_from_events_or_text(
    event_info: Dict[str, object],
    raw_label: str,
    path: Path,
) -> str:
    """Prefer HF_Lung event-label files; fall back to path/metadata text."""
    event_mapped = str(event_info.get("hflung_event_mapped_class", "unknown"))
    if bool(event_info.get("label_file_exists", False)) and event_mapped != "unknown":
        return event_mapped
    return map_hflung_class(raw_label, path)


def map_hflung_class(raw_label: str, path: Path) -> str:
    """
    Map an HF_Lung label/path to the audit classes:
    normal / crackles / wheezes / other.

    Any combined or non-ICBHI class, e.g. rhonchus/stridor/both/unknown, goes to
    'other' for Option A.
    """
    text = (raw_label + " " + " ".join(path.parts[-6:])).lower()
    text = text.replace("-", "_").replace(" ", "_")

    has_crackle = any(k in text for k in ["crackle", "crackles", "crepitation", "crepitations", "crepitant"])
    has_wheeze = any(k in text for k in ["wheeze", "wheezes", "wheezing"])
    has_normal = any(k in text for k in ["normal", "healthy", "control"])
    has_other_named = any(k in text for k in ["rhonch", "rhonchi", "rhonchus", "stridor", "stridors", "bronchial"])

    if has_crackle and has_wheeze:
        return "other"
    if has_other_named:
        return "other"
    if has_crackle:
        return "crackles"
    if has_wheeze:
        return "wheezes"
    if has_normal:
        return "normal"
    return "other"


# -----------------------------------------------------------------------------
# Audio processing
# -----------------------------------------------------------------------------

def read_audio(path: Path) -> Tuple[np.ndarray, int, int]:
    x, sr = sf.read(str(path), always_2d=False, dtype="float32")
    channels = 1
    if x.ndim == 2:
        channels = int(x.shape[1])
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    return x, int(sr), channels


def resample_to_target(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    if resample_poly is None:
        raise RuntimeError(
            "scipy is required for resampling because the input sample rate differs from target_sr. "
            "Install scipy or pre-resample the dataset."
        )
    frac = Fraction(target_sr, sr).limit_denominator(1000)
    y = resample_poly(x, frac.numerator, frac.denominator)
    return np.asarray(y, dtype=np.float32)


def frame_rms(x: np.ndarray, sr: int, frame_sec: float = 0.50, hop_sec: float = 0.25) -> np.ndarray:
    n = len(x)
    frame = max(1, int(round(frame_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    if n < frame:
        return np.array([float(np.sqrt(np.mean(x ** 2) + EPS))], dtype=np.float64)
    values = []
    for start in range(0, n - frame + 1, hop):
        seg = x[start:start + frame]
        values.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + EPS)))
    return np.asarray(values, dtype=np.float64)


def choose_best_15s_crop(x: np.ndarray, sr: int, target_sec: float) -> Tuple[np.ndarray, float, float]:
    """
    Return a 15 s crop/pad and diagnostics:
    - crop_start_s
    - crop_score

    For long files, the chosen window maximizes a simple activity/stability score:
      score = p10(frame_rms) - 0.25 * std(frame_rms)
    This avoids selecting a window with only a short loud transient.
    """
    target_n = int(round(target_sec * sr))
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return np.zeros(target_n, dtype=np.float32), 0.0, 0.0

    if len(x) <= target_n:
        out = np.zeros(target_n, dtype=np.float32)
        out[:len(x)] = x
        return out, 0.0, 0.0

    hop = int(round(1.0 * sr))
    starts = list(range(0, len(x) - target_n + 1, hop))
    if starts[-1] != len(x) - target_n:
        starts.append(len(x) - target_n)

    best_start = 0
    best_score = -float("inf")

    for start in starts:
        w = x[start:start + target_n]
        w = w - np.mean(w)
        fr = frame_rms(w, sr)
        p10 = float(np.percentile(fr, 10)) if len(fr) else 0.0
        std = float(np.std(fr)) if len(fr) else 0.0
        score = p10 - 0.25 * std
        if score > best_score:
            best_score = score
            best_start = start

    return x[best_start:best_start + target_n].astype(np.float32), best_start / sr, float(best_score)


# -----------------------------------------------------------------------------
# Feature extraction / quality scoring
# -----------------------------------------------------------------------------

def band_power(freqs: np.ndarray, power: np.ndarray, fmin: float, fmax: float) -> float:
    if fmax <= fmin:
        return 0.0
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(power[mask].sum())


def spectral_features(x: np.ndarray, sr: int) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        return {}

    window = np.hanning(n)
    spec = np.fft.rfft(x * window)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    total_power = float(power.sum() + EPS)

    bands = {
        "energy_0_50": band_power(freqs, power, 0, 50),
        "energy_20_50": band_power(freqs, power, 20, 50),
        "energy_50_150": band_power(freqs, power, 50, 150),
        "energy_150_500": band_power(freqs, power, 150, 500),
        "energy_200_500": band_power(freqs, power, 200, 500),
        "energy_500_1000": band_power(freqs, power, 500, 1000),
        "energy_1000_1800": band_power(freqs, power, 1000, 1800),
        "energy_1800_2000": band_power(freqs, power, 1800, min(2000, sr / 2)),
        "energy_50_1800": band_power(freqs, power, 50, min(1800, sr / 2)),
        "energy_1000_2000": band_power(freqs, power, 1000, min(2000, sr / 2)),
        "energy_20_500": band_power(freqs, power, 20, 500),
    }

    centroid = float((freqs * power).sum() / total_power)
    bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * power).sum() / total_power))
    p_nonzero = power[power > 0]
    flatness = float(np.exp(np.mean(np.log(p_nonzero + EPS))) / (np.mean(p_nonzero) + EPS))

    out: Dict[str, float] = {
        "total_psd_power": total_power,
        "total_psd_power_db": safe_db_power(total_power),
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_flatness": flatness,
    }

    for k, v in bands.items():
        out[f"{k}_power"] = float(v)
        out[f"{k}_ratio"] = float(v / total_power)
        out[f"{k}_db"] = safe_db_power(v)

    # Convenient aliases aligned with the previous audit scripts.
    out["lung_band_ratio_50_1800"] = out.get("energy_50_1800_ratio", 0.0)
    out["drift_ratio_0_50"] = out.get("energy_0_50_ratio", 0.0)
    out["ultra_hf_ratio_1800_2000"] = out.get("energy_1800_2000_ratio", 0.0)
    out["hf_noise_ratio_1000_2000"] = out.get("energy_1000_2000_ratio", 0.0)
    out["ls_band_50_150_ratio"] = out.get("energy_50_150_ratio", 0.0)
    out["ls_band_150_500_ratio"] = out.get("energy_150_500_ratio", 0.0)
    out["ls_band_200_500_ratio"] = out.get("energy_200_500_ratio", 0.0)
    out["ls_band_500_1000_ratio"] = out.get("energy_500_1000_ratio", 0.0)
    out["ls_band_1000_1800_ratio"] = out.get("energy_1000_1800_ratio", 0.0)
    out["ls_band_20_500_ratio"] = out.get("energy_20_500_ratio", 0.0)

    return out


def compute_quality_score(features: Dict[str, float], min_duration_s: float) -> Tuple[float, bool, Dict[str, float]]:
    duration_s = float(features.get("duration_s", 0.0))
    rms_db = float(features.get("rms_db", -120.0))
    clipping_ratio = float(features.get("clipping_ratio", 1.0))
    active_ratio = float(features.get("active_ratio", 0.0))
    seg_cv = float(features.get("seg_rms_cv", 999.0))
    lung_band = float(features.get("lung_band_ratio_50_1800", 0.0))
    drift = float(features.get("drift_ratio_0_50", 1.0))
    ultra = float(features.get("ultra_hf_ratio_1800_2000", 1.0))

    # Soft components, each 0..1.
    duration_score = clip01(duration_s / TARGET_SEC)
    rms_score = clip01((rms_db - (-55.0)) / 30.0)  # -55 dB -> 0, -25 dB -> 1
    non_clipping_score = clip01(1.0 - clipping_ratio / 0.005)
    active_score = clip01(active_ratio)
    stability_score = clip01(1.0 - seg_cv / 1.50)
    lung_band_score = clip01(lung_band / 0.65)
    drift_score = clip01(1.0 - drift / 0.55)
    ultra_score = clip01(1.0 - ultra / 0.35)

    quality_score = (
        0.20 * rms_score
        + 0.20 * active_score
        + 0.20 * lung_band_score
        + 0.15 * stability_score
        + 0.15 * non_clipping_score
        + 0.05 * drift_score
        + 0.05 * ultra_score
    )

    passes_hard = bool(
        duration_s >= min_duration_s
        and rms_db >= -55.0
        and clipping_ratio <= 0.005
        and active_ratio >= 0.40
        and lung_band >= 0.35
        and drift <= 0.65
        and ultra <= 0.45
    )

    components = {
        "duration_score": duration_score,
        "rms_score": rms_score,
        "non_clipping_score": non_clipping_score,
        "active_score": active_score,
        "stability_score": stability_score,
        "lung_band_score": lung_band_score,
        "drift_score": drift_score,
        "ultra_score": ultra_score,
    }

    return float(quality_score), passes_hard, components


def analyse_one_file(
    path: Path,
    raw_root: Path,
    metadata_lookup: Dict[str, Dict[str, str]],
    label_col: str,
    target_sr: int,
    target_sec: float,
    min_duration_s: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "original_path": str(path),
        "relative_path": str(path.relative_to(raw_root)) if path.is_relative_to(raw_root) else path.name,
        "filename": path.name,
        "source_id": stable_id_from_path(path),
        "ok_read": False,
        "error": "",
    }

    metadata_row = metadata_row_for_path(path, metadata_lookup)
    raw_label = infer_raw_label_from_metadata_or_path(path, metadata_row, label_col=label_col)
    event_info = parse_hflung_event_label_file(path)
    mapped_class = map_hflung_class_from_events_or_text(event_info, raw_label, path)

    row["raw_label"] = raw_label
    row.update(event_info)
    row["mapped_class"] = mapped_class

    try:
        x_raw, sr_raw, channels = read_audio(path)
        original_duration_s = float(len(x_raw) / sr_raw) if sr_raw > 0 else 0.0

        x_rs = resample_to_target(x_raw, sr_raw, target_sr)
        x_crop, crop_start_s, crop_score = choose_best_15s_crop(x_rs, target_sr, target_sec)

        # DC removal only. No RMS/peak normalization.
        x_proc = x_crop.astype(np.float64)
        x_proc = x_proc - np.mean(x_proc)
        x_proc = x_proc.astype(np.float32)

        rms = float(np.sqrt(np.mean(x_proc.astype(np.float64) ** 2) + EPS))
        peak_abs = float(np.max(np.abs(x_proc)) + EPS)
        crest_factor = float(peak_abs / max(rms, EPS))
        rms_db = safe_db_amp(rms)
        clipping_ratio = float(np.mean(np.abs(x_proc) >= 0.999))

        fr = frame_rms(x_proc, target_sr)
        fr_p10 = float(np.percentile(fr, 10)) if len(fr) else 0.0
        fr_p50 = float(np.percentile(fr, 50)) if len(fr) else 0.0
        fr_p90 = float(np.percentile(fr, 90)) if len(fr) else 0.0
        fr_mean = float(np.mean(fr)) if len(fr) else 0.0
        fr_std = float(np.std(fr)) if len(fr) else 0.0
        seg_cv = float(fr_std / (fr_mean + EPS))

        # Relative silence threshold: robust across acquisition gains.
        silence_thr = max(1e-5, 0.05 * fr_p90)
        silence_ratio = float(np.mean(fr < silence_thr)) if len(fr) else 1.0
        active_ratio = float(1.0 - silence_ratio)

        # ZCR after DC removal.
        zcr = float(np.mean(np.abs(np.diff(np.signbit(x_proc).astype(np.int8))))) if len(x_proc) > 1 else 0.0

        spec = spectral_features(x_proc, target_sr)

        features: Dict[str, float] = {
            "original_sr": float(sr_raw),
            "channels": float(channels),
            "original_duration_s": original_duration_s,
            "target_sr": float(target_sr),
            "duration_s": float(len(x_proc) / target_sr),
            "crop_start_s": float(crop_start_s),
            "crop_score": float(crop_score),
            "rms": rms,
            "fixed_rms": rms,  # compatibility alias with earlier quality tables
            "rms_db": rms_db,
            "peak_abs": peak_abs,
            "crest_factor": crest_factor,
            "clipping_ratio": clipping_ratio,
            "seg_rms_p10": fr_p10,
            "seg_rms_p50": fr_p50,
            "seg_rms_p90": fr_p90,
            "seg_rms_mean": fr_mean,
            "seg_rms_std": fr_std,
            "seg_rms_cv": seg_cv,
            "silence_ratio": silence_ratio,
            "active_ratio": active_ratio,
            "zcr": zcr,
        }
        features.update(spec)

        q, passes, comps = compute_quality_score(features, min_duration_s=min_duration_s)
        row.update(features)
        row.update(comps)
        row["quality_score"] = q
        row["passes_hard_quality"] = passes
        row["ok_read"] = True

    except Exception as exc:
        row["error"] = str(exc)
        row["quality_score"] = 0.0
        row["passes_hard_quality"] = False

    return row


# -----------------------------------------------------------------------------
# Selection and writing
# -----------------------------------------------------------------------------

def select_balanced_option_a(
    df: pd.DataFrame,
    seed: int,
    min_quality_score: float,
) -> pd.DataFrame:
    """
    Select Option A. Prefer hard-quality candidates. If a class has too few
    hard-quality rows, fall back to ranked rows from the same class and mark it
    in selection_mode.
    """
    rng = np.random.default_rng(seed)
    selected_parts: List[pd.DataFrame] = []
    used_paths: set[str] = set()

    candidates = df[df["ok_read"].astype(bool)].copy()
    candidates = candidates[candidates["quality_score"].astype(float) >= float(min_quality_score)].copy()

    if len(candidates) == 0:
        raise RuntimeError("No candidates left after ok_read and min_quality_score filtering.")

    candidates["_rand"] = rng.random(len(candidates))

    for cls, split_counts in OPTION_A_COUNTS.items():
        class_pool_all = candidates[candidates["mapped_class"] == cls].copy()
        class_pool_all = class_pool_all[~class_pool_all["original_path"].astype(str).isin(used_paths)].copy()

        n_need = int(split_counts["train"] + split_counts["val"])
        hard_pool = class_pool_all[class_pool_all["passes_hard_quality"].astype(bool)].copy()

        if len(hard_pool) >= n_need:
            pool = hard_pool
            mode = "hard_quality"
        else:
            pool = class_pool_all
            mode = "fallback_ranked_same_class"

        if len(pool) < n_need:
            print(
                f"[WARNING] class={cls}: need={n_need}, available={len(pool)}. "
                "Will select available rows and fill missing from global pool later."
            )
            n_take = len(pool)
        else:
            n_take = n_need

        # Rank primarily by quality, secondarily random for stable tie-breaking.
        pool = pool.sort_values(["quality_score", "seg_rms_p10", "lung_band_ratio_50_1800", "_rand"], ascending=[False, False, False, True])
        chosen = pool.head(n_take).copy()
        chosen["selection_class_target"] = cls
        chosen["selection_mode"] = mode

        # Assign train/val counts inside the selected class.
        chosen = chosen.reset_index(drop=True)
        chosen["split"] = "unused"
        n_val = min(int(split_counts["val"]), len(chosen))
        n_train = min(int(split_counts["train"]), max(0, len(chosen) - n_val))

        # Put highest quality in train first and next in val; this keeps both sets clean.
        chosen.loc[:n_train - 1, "split"] = "train"
        chosen.loc[n_train:n_train + n_val - 1, "split"] = "val"

        selected_parts.append(chosen)
        used_paths |= set(chosen["original_path"].astype(str).tolist())

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()

    # Fill missing train/val totals, if a class was under-populated.
    target_train = sum(v["train"] for v in OPTION_A_COUNTS.values())
    target_val = sum(v["val"] for v in OPTION_A_COUNTS.values())
    current_train = int((selected.get("split", pd.Series(dtype=str)) == "train").sum()) if len(selected) else 0
    current_val = int((selected.get("split", pd.Series(dtype=str)) == "val").sum()) if len(selected) else 0

    missing_train = target_train - current_train
    missing_val = target_val - current_val

    if missing_train > 0 or missing_val > 0:
        filler = candidates[~candidates["original_path"].astype(str).isin(used_paths)].copy()
        filler = filler.sort_values(["passes_hard_quality", "quality_score", "seg_rms_p10", "_rand"], ascending=[False, False, False, True])
        fill_rows = []

        if missing_train > 0:
            take = filler.head(missing_train).copy()
            take["split"] = "train"
            take["selection_class_target"] = "global_fill"
            take["selection_mode"] = "global_fill_missing_train"
            fill_rows.append(take)
            used_paths |= set(take["original_path"].astype(str).tolist())
            filler = filler[~filler["original_path"].astype(str).isin(used_paths)].copy()

        if missing_val > 0:
            take = filler.head(missing_val).copy()
            take["split"] = "val"
            take["selection_class_target"] = "global_fill"
            take["selection_mode"] = "global_fill_missing_val"
            fill_rows.append(take)

        if fill_rows:
            selected = pd.concat([selected] + fill_rows, ignore_index=True, sort=False)

    selected = selected.drop(columns=["_rand"], errors="ignore")
    selected = selected.sort_values(["split", "selection_class_target", "quality_score"], ascending=[True, True, False]).reset_index(drop=True)

    # Stable selected IDs.
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    selected["selected_source_id"] = [f"HFLUNG_LS_{i:04d}" for i in range(1, len(selected) + 1)]

    return selected


def write_processed_selected_wavs(
    selected: pd.DataFrame,
    raw_root: Path,
    processed_dir: Path,
    target_sr: int,
    target_sec: float,
    overwrite: bool,
) -> pd.DataFrame:
    if processed_dir.exists() and overwrite:
        shutil.rmtree(processed_dir)
    ensure_dir(processed_dir)
    wav_dir = ensure_dir(processed_dir / "LS")

    rows = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Writing selected processed WAVs"):
        src = Path(str(row["original_path"]))
        selected_source_id = str(row["selected_source_id"])
        cls = str(row.get("mapped_class", "other"))
        split = str(row.get("split", "unused"))
        out_name = f"L_{selected_source_id}_{split}_{cls}.wav"
        out_path = wav_dir / out_name

        x_raw, sr_raw, _ = read_audio(src)
        x_rs = resample_to_target(x_raw, sr_raw, target_sr)
        x_crop, _, _ = choose_best_15s_crop(x_rs, target_sr, target_sec)
        x_proc = x_crop.astype(np.float64)
        x_proc = x_proc - np.mean(x_proc)
        x_proc = x_proc.astype(np.float32)

        # FLOAT subtype preserves amplitudes without integer clipping.
        sf.write(str(out_path), x_proc, target_sr, subtype="FLOAT")

        d = row.to_dict()
        d["processed_path"] = str(out_path)
        d["processed_filename"] = out_name
        rows.append(d)

    out = pd.DataFrame(rows)
    return out


def write_selection_summary(all_df: pd.DataFrame, selected_df: pd.DataFrame, out_dir: Path, processed_dir: Path) -> None:
    lines = []
    lines.append("HF_Lung_V1 quality score and Option A selection")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"All candidates: {len(all_df)}")
    lines.append(f"Readable candidates: {int(all_df['ok_read'].astype(bool).sum()) if len(all_df) else 0}")
    lines.append(f"Hard-quality candidates: {int(all_df['passes_hard_quality'].astype(bool).sum()) if len(all_df) else 0}")
    lines.append(f"Selected candidates: {len(selected_df)}")
    lines.append(f"Processed dir: {processed_dir}")
    lines.append("")

    if len(all_df):
        lines.append("All candidates by mapped_class:")
        lines.append(str(all_df["mapped_class"].value_counts(dropna=False)))
        lines.append("")
        lines.append("Hard-quality candidates by mapped_class:")
        hard = all_df[all_df["passes_hard_quality"].astype(bool)]
        lines.append(str(hard["mapped_class"].value_counts(dropna=False)))
        lines.append("")

    if len(selected_df):
        lines.append("Selected by mapped_class and split:")
        lines.append(str(pd.crosstab(selected_df["mapped_class"], selected_df["split"])))
        lines.append("")
        lines.append("Selected by selection target and split:")
        lines.append(str(pd.crosstab(selected_df["selection_class_target"], selected_df["split"])))
        lines.append("")
        lines.append("Selected quality summary:")
        lines.append(str(selected_df.groupby(["mapped_class", "split"])["quality_score"].describe()))
        lines.append("")
        lines.append("Selection modes:")
        lines.append(str(selected_df["selection_mode"].value_counts(dropna=False)))
        lines.append("")

    (out_dir / "summary_hflung_quality_selection.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--raw-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--metadata-label-col", type=str, default="", help="Optional explicit label column in metadata CSV.")
    p.add_argument("--target-sr", type=int, default=TARGET_SR)
    p.add_argument("--target-sec", type=float, default=TARGET_SEC)
    p.add_argument("--min-duration-s", type=float, default=14.5)
    p.add_argument("--min-quality-score", type=float, default=0.0, help="Soft pre-filter before selection. Keep 0.0 initially.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-write-wavs", action="store_true", help="Only compute CSVs; do not write processed selected WAVs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)

    raw_root: Path = (
        args.raw_root
        if args.raw_root is not None
        else project_root / "dataset" / "raw" / "HF_Lung_V1"
    )
    out_dir: Path = ensure_dir(
        args.out_dir
        if args.out_dir is not None
        else project_root / "outputs" / "hflung_v1_quality_selection"
    )
    processed_dir: Path = (
        args.processed_dir
        if args.processed_dir is not None
        else project_root / "dataset" / "processed" / "hflung_v1_selected_70_audit"
    )

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw HF_Lung root not found: {raw_root}")

    print("=" * 100)
    print("HF_Lung_V1 quality score + Option A selection")
    print("=" * 100)
    print(f"Raw root      : {raw_root}")
    print(f"Output dir    : {out_dir}")
    print(f"Processed dir : {processed_dir}")
    print(f"Target        : {args.target_sr} Hz | {args.target_sec:.1f} s | mono")
    print(f"Seed          : {args.seed}")
    print()

    wav_files = find_wavs(raw_root)
    if len(wav_files) == 0:
        raise RuntimeError(f"No WAV files found under: {raw_root}")
    print(f"WAV files found: {len(wav_files)}")

    metadata_lookup = load_optional_metadata(args.metadata_csv)
    if args.metadata_csv is not None:
        print(f"Metadata loaded: {args.metadata_csv} | lookup keys={len(metadata_lookup)}")

    rows = []
    for p in tqdm(wav_files, desc="Scoring all HF_Lung WAVs"):
        rows.append(
            analyse_one_file(
                path=p,
                raw_root=raw_root,
                metadata_lookup=metadata_lookup,
                label_col=args.metadata_label_col,
                target_sr=args.target_sr,
                target_sec=args.target_sec,
                min_duration_s=args.min_duration_s,
            )
        )

    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values(["mapped_class", "quality_score"], ascending=[True, False]).reset_index(drop=True)

    all_csv = out_dir / "hflung_all_candidates_quality.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\nSaved all candidates: {all_csv}")

    print("\nCandidates by mapped_class:")
    print(all_df["mapped_class"].value_counts(dropna=False).to_string())
    print("\nHard-quality candidates by mapped_class:")
    hard = all_df[all_df["passes_hard_quality"].astype(bool)]
    print(hard["mapped_class"].value_counts(dropna=False).to_string())

    selected = select_balanced_option_a(
        all_df,
        seed=args.seed,
        min_quality_score=float(args.min_quality_score),
    )

    selected_csv_pre = out_dir / "hflung_selected_70_sources_prewrite.csv"
    selected.to_csv(selected_csv_pre, index=False)

    if not args.no_write_wavs:
        selected_written = write_processed_selected_wavs(
            selected=selected,
            raw_root=raw_root,
            processed_dir=processed_dir,
            target_sr=args.target_sr,
            target_sec=args.target_sec,
            overwrite=bool(args.overwrite),
        )
    else:
        selected_written = selected.copy()
        selected_written["processed_path"] = ""

    selected_csv = out_dir / "hflung_selected_70_sources.csv"
    selected_written.to_csv(selected_csv, index=False)

    # A minimal source manifest for the later multi-domain audit.
    manifest = pd.DataFrame({
        "domain": "HFLUNG",
        "source_type": "LS",
        "source_id": selected_written["selected_source_id"],
        "class_name": selected_written["mapped_class"],
        "split": selected_written["split"],
        "original_path": selected_written["original_path"],
        "processed_path": selected_written.get("processed_path", ""),
        "quality_score": selected_written["quality_score"],
        "passes_hard_quality": selected_written["passes_hard_quality"],
    })
    manifest_csv = out_dir / "hflung_manifest_sources_for_domain_audit.csv"
    manifest.to_csv(manifest_csv, index=False)

    # Also save a copy next to the processed WAVs.
    if not args.no_write_wavs:
        ensure_dir(processed_dir)
        selected_written.to_csv(processed_dir / "hflung_selected_70_sources.csv", index=False)
        manifest.to_csv(processed_dir / "manifest_sources.csv", index=False)

    write_selection_summary(all_df, selected_written, out_dir, processed_dir)

    print("\nSelected by mapped_class and split:")
    print(pd.crosstab(selected_written["mapped_class"], selected_written["split"]).to_string())
    print("\nSelected by selection target and split:")
    print(pd.crosstab(selected_written["selection_class_target"], selected_written["split"]).to_string())

    print("\nDONE.")
    print(f"All candidates quality CSV : {all_csv}")
    print(f"Selected 70 CSV            : {selected_csv}")
    print(f"Audit manifest             : {manifest_csv}")
    if not args.no_write_wavs:
        print(f"Processed selected WAV dir : {processed_dir}")
    print(f"Summary                    : {out_dir / 'summary_hflung_quality_selection.txt'}")


if __name__ == "__main__":
    main()

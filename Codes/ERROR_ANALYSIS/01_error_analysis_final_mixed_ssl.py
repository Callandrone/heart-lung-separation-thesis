#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import stft as scipy_stft
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Path bootstrap
# -----------------------------------------------------------------------------


def add_code_paths(extra_code_root: Optional[str] = None) -> None:
    """Make encoder.py / separator.py / decoder.py / model_config.py importable."""
    here = Path(__file__).resolve()
    candidates = []

    if extra_code_root:
        candidates.append(Path(extra_code_root))

    # Expected layout: DeepLearning/codes/ERROR_ANALYSIS/this_script.py
    candidates.extend([
        here.parent,
        here.parent.parent,                 # DeepLearning/codes
        here.parent.parent / "SSL_MIXED",   # optional copied subfolder
        here.parent.parent / "src",
    ])

    env_root = os.environ.get("DL_CODE_ROOT", "")
    if env_root:
        candidates.append(Path(env_root))

    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


# -----------------------------------------------------------------------------
# Basic signal helpers
# -----------------------------------------------------------------------------


def dc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x - np.mean(x)


def safe_rms(x: np.ndarray, eps: float = 1e-8) -> float:
    x = dc(x)
    return float(np.sqrt(np.mean(x ** 2) + eps))


def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = dc(a)
    b = dc(b)
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)) + eps)
    return float(np.dot(a, b) / den)


def safe_db_ratio(num: float, den: float, eps: float = 1e-8) -> float:
    return float(10.0 * np.log10((num + eps) / (den + eps)))


def si_sdr_np(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    est = dc(estimate)
    tgt = dc(target)
    alpha = float(np.dot(est, tgt) / (np.dot(tgt, tgt) + eps))
    proj = alpha * tgt
    noise = est - proj
    return safe_db_ratio(float(np.dot(proj, proj)), float(np.dot(noise, noise)), eps=eps)


def signed_projection_alpha_np(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    est = dc(estimate)
    tgt = dc(target)
    return float(np.dot(est, tgt) / (np.dot(tgt, tgt) + eps))


def nmse_raw_db_np(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    est = dc(estimate)
    tgt = dc(target)
    err = tgt - est
    nmse = float(np.dot(err, err) / (np.dot(tgt, tgt) + eps))
    return float(10.0 * np.log10(max(nmse, eps)))


def nmse_aligned_db_np(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    est = dc(estimate)
    tgt = dc(target)
    est_energy = float(np.dot(est, est))
    if est_energy < eps:
        return 0.0
    alpha = float(np.dot(est, tgt) / (est_energy + eps))
    est_aligned = alpha * est
    err = tgt - est_aligned
    nmse = float(np.dot(err, err) / (np.dot(tgt, tgt) + eps))
    return float(10.0 * np.log10(max(nmse, eps)))


def rms_gain_error_db_np(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    return float(20.0 * np.log10((safe_rms(estimate, eps) + eps) / (safe_rms(target, eps) + eps)))


def local_target_snr_db_np(H_ref: np.ndarray, L_ref: np.ndarray, eps: float = 1e-8) -> float:
    return float(20.0 * np.log10((safe_rms(H_ref, eps) + eps) / (safe_rms(L_ref, eps) + eps)))


def mixture_consistency_metrics_np(
    mixture: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    mix = dc(mixture)
    pred_mix = dc(H_hat + L_hat)
    residual = mix - pred_mix
    mix_energy = float(np.dot(mix, mix))
    pred_energy = float(np.dot(pred_mix, pred_mix))

    if mix_energy < eps:
        return {
            "mix_nmse_db": float("nan"),
            "mix_corr": float("nan"),
            "mix_gain_error_db": float("nan"),
            "mix_gamma": float("nan"),
            "negative_mix_gamma": float("nan"),
        }

    nmse = float(np.dot(residual, residual) / (mix_energy + eps))
    gamma = float(np.dot(mix, pred_mix) / (pred_energy + eps)) if pred_energy >= eps else 1.0
    corr = safe_corr(mix, pred_mix, eps=eps)
    gain_error_db = float(20.0 * np.log10((safe_rms(pred_mix, eps) + eps) / (safe_rms(mix, eps) + eps)))

    return {
        "mix_nmse_db": float(10.0 * np.log10(max(nmse, eps))),
        "mix_corr": corr,
        "mix_gain_error_db": gain_error_db,
        "mix_gamma": gamma,
        "negative_mix_gamma": float(gamma < 0),
    }


def apply_global_polarity_calibration_np(
    mixture: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    pre = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    should_flip = bool(pre["mix_gamma"] < 0.0)
    if should_flip:
        H_hat = -H_hat
        L_hat = -L_hat
    post = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    return H_hat, L_hat, {
        "polarity_flipped": float(should_flip),
        "mix_corr_before_polarity": float(pre["mix_corr"]),
        "mix_gamma_before_polarity": float(pre["mix_gamma"]),
        "mix_corr_after_polarity": float(post["mix_corr"]),
        "mix_gamma_after_polarity": float(post["mix_gamma"]),
    }


def apply_mixture_gain_calibration_np(
    mixture: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    pre = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    mix = dc(mixture)
    pred_mix = dc(H_hat + L_hat)
    pred_energy = float(np.dot(pred_mix, pred_mix))
    gamma = float(np.dot(mix, pred_mix) / (pred_energy + eps)) if pred_energy >= eps else 1.0
    H_hat = gamma * H_hat
    L_hat = gamma * L_hat
    post = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    return H_hat, L_hat, {
        "gain_calibration_gamma": gamma,
        "mix_corr_before_gain": float(pre["mix_corr"]),
        "mix_nmse_db_before_gain": float(pre["mix_nmse_db"]),
        "mix_gain_error_db_before_gain": float(pre["mix_gain_error_db"]),
        "mix_corr_after_gain": float(post["mix_corr"]),
        "mix_nmse_db_after_gain": float(post["mix_nmse_db"]),
        "mix_gain_error_db_after_gain": float(post["mix_gain_error_db"]),
    }


def bss_eval_sources(estimate: np.ndarray, target: np.ndarray, interference: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    def proj(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
        return onto * (np.dot(v, onto) / (np.dot(onto, onto) + eps))

    e = dc(estimate)
    t = dc(target)
    i = dc(interference)

    s_tgt = proj(e, t)
    e_interf = proj(e - s_tgt, i)
    e_artif = e - s_tgt - e_interf

    def db(num_vec: np.ndarray, den_vec: np.ndarray) -> float:
        return safe_db_ratio(float(np.dot(num_vec, num_vec)), float(np.dot(den_vec, den_vec)), eps=eps)

    return {
        "sdr": db(s_tgt, e_interf + e_artif),
        "sir": db(s_tgt, e_interf),
        "sar": db(s_tgt + e_interf, e_artif),
    }


def log_spectral_distance(estimate: np.ndarray, target: np.ndarray, sr: int, n_fft: int = 256, hop: int = 64) -> float:
    def log_mag(x: np.ndarray) -> np.ndarray:
        _, _, S = scipy_stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        return np.log10(np.abs(S) + 1e-8)

    lm_est = log_mag(estimate)
    lm_ref = log_mag(target)
    diff = lm_est - lm_ref
    return float(np.mean(np.sqrt(np.mean(diff ** 2, axis=0))))


def log_spectrogram_correlation(estimate: np.ndarray, target: np.ndarray, sr: int, n_fft: int = 256, hop: int = 64) -> float:
    def log_mag_flat(x: np.ndarray) -> np.ndarray:
        _, _, S = scipy_stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        return np.log10(np.abs(S) + 1e-8).reshape(-1)

    a = log_mag_flat(estimate)
    b = log_mag_flat(target)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# -----------------------------------------------------------------------------
# Acoustic feature extraction
# -----------------------------------------------------------------------------


def bandpower_ratio(x: np.ndarray, sr: int, lo: float, hi: float, eps: float = 1e-12) -> float:
    x = dc(x)
    n = int(max(256, 2 ** math.ceil(math.log2(max(len(x), 256)))))
    X = np.fft.rfft(x, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    power = np.abs(X) ** 2
    total = float(np.sum(power) + eps)
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sum(power[mask]) / total)


def spectral_features(x: np.ndarray, sr: int, prefix: str) -> Dict[str, float]:
    x = dc(x)
    rms = safe_rms(x)
    peak = float(np.max(np.abs(x)) + 1e-8)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)).astype(np.float32))))

    n = int(max(256, 2 ** math.ceil(math.log2(max(len(x), 256)))))
    X = np.fft.rfft(x, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(X) + 1e-12
    power = mag ** 2
    power_sum = float(np.sum(power) + 1e-12)

    centroid = float(np.sum(freqs * power) / power_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / power_sum))
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))

    out = {
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_crest_factor": float(peak / (rms + 1e-8)),
        f"{prefix}_zcr": zcr,
        f"{prefix}_spectral_centroid": centroid,
        f"{prefix}_spectral_bandwidth": bandwidth,
        f"{prefix}_spectral_flatness": flatness,
    }

    for lo, hi, name in [
        (20, 50, "band_20_50"),
        (50, 150, "band_50_150"),
        (150, 500, "band_150_500"),
        (200, 500, "band_200_500"),
        (500, 1000, "band_500_1000"),
        (1000, 1800, "band_1000_1800"),
        (20, 500, "band_20_500"),
        (50, 1800, "band_50_1800"),
    ]:
        out[f"{prefix}_{name}"] = bandpower_ratio(x, sr, lo, hi)

    return out


# -----------------------------------------------------------------------------
# Dataset and split helpers
# -----------------------------------------------------------------------------


def load_wav(path: Path, sr: int, seg_samples: int) -> np.ndarray:
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"Unexpected sample rate for {path}: expected {sr}, got {file_sr}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) > seg_samples:
        audio = audio[:seg_samples]
    elif len(audio) < seg_samples:
        audio = np.pad(audio, (0, seg_samples - len(audio)))
    return audio.astype(np.float32)


def normalise_base_id(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return ""
    s = Path(s).stem
    if s.startswith(("M_", "H_", "L_")):
        s = s[2:]
    if "_s" in s:
        s = s.split("_s")[0]
    try:
        if s.replace(".", "", 1).isdigit():
            f = float(s)
            if f.is_integer():
                return f"{int(f):06d}"
    except Exception:
        pass
    return s


def sample_name_to_base_id(name: str) -> str:
    return normalise_base_id(name)


def list_dataset_names(dataset_dir: Path) -> List[str]:
    names = sorted([p.stem[2:] for p in dataset_dir.glob("M_*.wav")])
    if not names:
        raise RuntimeError(f"No M_*.wav files found in {dataset_dir}")
    return names


def pick_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_val_bases_from_split(split_csv: Path, fold: int) -> Tuple[set[str], Dict[str, str]]:
    df = pd.read_csv(split_csv)
    fold_col = pick_existing_column(df, ["fold_no", "fold", "fold_idx"])
    split_col = pick_existing_column(df, ["split", "set", "subset"])
    base_col = pick_existing_column(df, ["base_id", "case_id", "triplet_id", "id"])
    snr_col = pick_existing_column(df, ["snr_label", "snr", "snr_db", "snr_condition"])

    missing = [name for name, col in [("fold", fold_col), ("split", split_col), ("base", base_col)] if col is None]
    if missing:
        raise RuntimeError(f"Missing split CSV columns: {missing}. Available: {list(df.columns)}")

    fold_values = pd.to_numeric(df[fold_col], errors="coerce")
    split_values = df[split_col].astype(str).str.lower().str.strip()
    val_df = df[(fold_values == int(fold)) & (split_values == "val")].copy()

    val_bases = {normalise_base_id(x) for x in val_df[base_col].tolist()}
    val_bases = {x for x in val_bases if x}
    if not val_bases:
        raise RuntimeError(f"No validation base ids found for fold={fold} in {split_csv}")

    snr_lookup: Dict[str, str] = {}
    if snr_col is not None:
        for _, row in df[[base_col, snr_col]].drop_duplicates().iterrows():
            base = normalise_base_id(row[base_col])
            if base:
                snr_lookup[base] = str(row[snr_col])

    return val_bases, snr_lookup


def build_eval_file_list(
    dataset_dir: Path,
    split_csv: Optional[Path],
    fold: Optional[int],
    only_orig: bool = True,
) -> Tuple[List[Tuple[Path, Path, Path]], Dict[str, str], Optional[set[str]]]:
    names = list_dataset_names(dataset_dir)
    snr_lookup: Dict[str, str] = {}
    val_bases: Optional[set[str]] = None

    if split_csv is not None and str(split_csv).strip() and fold is not None:
        val_bases, snr_lookup = load_val_bases_from_split(split_csv, fold)
        names = [n for n in names if sample_name_to_base_id(n) in val_bases]

    if only_orig:
        names = [n for n in names if n.endswith("_orig")]

    files = [(dataset_dir / f"M_{n}.wav", dataset_dir / f"H_{n}.wav", dataset_dir / f"L_{n}.wav") for n in names]
    missing = [str(p) for triple in files for p in triple if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files, first 10:\n" + "\n".join(missing[:10]))
    return files, snr_lookup, val_bases


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model_config() -> Any:
    import model_config as cfg  # type: ignore
    return cfg


class CardiopulmonaryNet(nn.Module):
    def __init__(self, cfg: Any, T: int):
        super().__init__()
        from encoder import CardiopulmonaryEncoder  # type: ignore
        from separator import CardiopulmonarySeparator  # type: ignore
        from decoder import CardiopulmonaryDecoder  # type: ignore

        self.encoder = CardiopulmonaryEncoder(
            in_channels=1,
            N_filters=cfg.N_FILTERS,
            N_latent=cfg.N_LATENT,
            kernel_size=cfg.KERNEL_SIZE,
            stride=cfg.STRIDE,
            R=cfg.ENCODER_R,
            dilations=cfg.DILATIONS,
            dropout_p=cfg.DROPOUT_P,
        )
        self.separator = CardiopulmonarySeparator(
            dim=cfg.N_LATENT,
            n_sources=cfg.N_SOURCES,
            n_stacks=getattr(cfg, "SEP_N_STACKS", 1),
            S=getattr(cfg, "SEP_BLOCKS_PER_STACK", getattr(cfg, "JASSNET_NUM_MODULES", 4)),
            tcn_bottleneck=getattr(cfg, "SEP_TCN_BOTTLENECK", getattr(cfg, "JASSNET_ATTN_DIM", 64)),
            attn_heads=getattr(cfg, "ATTN_HEADS", 1),
            attn_window=getattr(cfg, "ATTN_WINDOW", getattr(cfg, "JASSNET_LOCAL_CHUNK", 50)),
            n_global=getattr(cfg, "N_GLOBAL_TOKENS", 0),
            dropout_p=cfg.DROPOUT_P,
            mask_scale=getattr(cfg, "MASK_SCALE", None),
        )
        self.decoder = CardiopulmonaryDecoder(
            N_latent=cfg.N_LATENT,
            N_filters=cfg.N_FILTERS,
            kernel_size=cfg.KERNEL_SIZE,
            stride=cfg.STRIDE,
            R=cfg.DECODER_R,
            dilations=cfg.DILATIONS,
            dropout_p=cfg.DROPOUT_P,
            target_T=T,
        )

    def forward(self, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = self.encoder(M)
        Z_hs, Z_ls, _ = self.separator(Z)
        H_hat = self.decoder(Z_hs, normalise=False)
        L_hat = self.decoder(Z_ls, normalise=False)
        return H_hat, L_hat


def load_checkpoint_model(ckpt_path: Path, cfg: Any, T: int, device: torch.device) -> nn.Module:
    model = CardiopulmonaryNet(cfg, T=T).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def resolve_fold_checkpoint(ckpt_dir: Path, fold: int) -> Optional[Path]:
    candidates = [
        ckpt_dir / f"finetune_fold{fold}_best.pt",
        ckpt_dir / f"scratch_fold{fold}_best.pt",
        ckpt_dir / f"fold{fold}_best.pt",
        ckpt_dir / f"model_fold{fold}_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Evaluation and diagnostics
# -----------------------------------------------------------------------------


def get_snr_label(sample_id: str, snr_lookup: Dict[str, str]) -> str:
    base = sample_name_to_base_id(sample_id)
    if base in snr_lookup:
        return str(snr_lookup[base])
    lower = sample_id.lower()
    # Best effort for names with snr tokens.
    m = re.search(r"snr[_-]?([+-]?\d+|no)", lower)
    return m.group(1) if m else "unknown"


def local_snr_bin(local_snr_db: float) -> str:
    if local_snr_db <= -15:
        return "HS_very_weak"
    if local_snr_db <= -6:
        return "HS_weak"
    if local_snr_db < 6:
        return "balanced"
    if local_snr_db < 15:
        return "LS_weak"
    return "LS_very_weak"


def compute_metrics(
    M: np.ndarray,
    H_ref: np.ndarray,
    L_ref: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    sr: int,
) -> Dict[str, float]:
    si_h = si_sdr_np(H_hat, H_ref)
    si_l = si_sdr_np(L_hat, L_ref)
    bss_h = bss_eval_sources(H_hat, H_ref, L_ref)
    bss_l = bss_eval_sources(L_hat, L_ref, H_ref)

    local_snr = local_target_snr_db_np(H_ref, L_ref)
    mix = mixture_consistency_metrics_np(M, H_hat, L_hat)

    swap_h = si_sdr_np(H_hat, L_ref)
    swap_l = si_sdr_np(L_hat, H_ref)
    normal_mean = float(np.mean([si_h, si_l]))
    swap_mean = float(np.mean([swap_h, swap_l]))

    out = {
        "si_sdr_h": si_h,
        "si_sdr_l": si_l,
        "mean_si_sdr": normal_mean,
        "min_si_sdr": float(min(si_h, si_l)),
        "sdr_h": bss_h["sdr"],
        "sdr_l": bss_l["sdr"],
        "sir_h": bss_h["sir"],
        "sir_l": bss_l["sir"],
        "sar_h": bss_h["sar"],
        "sar_l": bss_l["sar"],
        "nmse_raw_db_h": nmse_raw_db_np(H_hat, H_ref),
        "nmse_raw_db_l": nmse_raw_db_np(L_hat, L_ref),
        "nmse_aligned_db_h": nmse_aligned_db_np(H_hat, H_ref),
        "nmse_aligned_db_l": nmse_aligned_db_np(L_hat, L_ref),
        "gain_error_db_h": rms_gain_error_db_np(H_hat, H_ref),
        "gain_error_db_l": rms_gain_error_db_np(L_hat, L_ref),
        "signed_alpha_h": signed_projection_alpha_np(H_hat, H_ref),
        "signed_alpha_l": signed_projection_alpha_np(L_hat, L_ref),
        "negative_alpha_h": float(signed_projection_alpha_np(H_hat, H_ref) < 0.0),
        "negative_alpha_l": float(signed_projection_alpha_np(L_hat, L_ref) < 0.0),
        "wave_corr_h": safe_corr(H_hat, H_ref),
        "wave_corr_l": safe_corr(L_hat, L_ref),
        "spec_corr_h": log_spectrogram_correlation(H_hat, H_ref, sr=sr),
        "spec_corr_l": log_spectrogram_correlation(L_hat, L_ref, sr=sr),
        "lsd_h": log_spectral_distance(H_hat, H_ref, sr=sr),
        "lsd_l": log_spectral_distance(L_hat, L_ref, sr=sr),
        "target_local_snr_db": local_snr,
        "local_snr_bin": local_snr_bin(local_snr),
        "hs_hat_corr_l_ref": safe_corr(H_hat, L_ref),
        "ls_hat_corr_h_ref": safe_corr(L_hat, H_ref),
        "h_l_ref_corr": safe_corr(H_ref, L_ref),
        "h_l_hat_corr": safe_corr(H_hat, L_hat),
        "swap_si_sdr_hhat_to_lref": swap_h,
        "swap_si_sdr_lhat_to_href": swap_l,
        "swap_mean_si_sdr": swap_mean,
        "swap_gain_db": swap_mean - normal_mean,
        "possible_swap": float((swap_mean - normal_mean) > 3.0),
    }
    out.update(mix)
    return out


def assign_failure_tags(row: Dict[str, Any]) -> str:
    tags = []
    if row.get("si_sdr_h", 999) < 0:
        tags.append("HS_LOW_SISDR")
    if row.get("si_sdr_l", 999) < 0:
        tags.append("LS_LOW_SISDR")
    if row.get("target_local_snr_db", 0) <= -15:
        tags.append("HS_VERY_WEAK_TARGET")
    if row.get("target_local_snr_db", 0) >= 15:
        tags.append("LS_VERY_WEAK_TARGET")
    if row.get("mix_corr", 1) < 0.95 or row.get("mix_nmse_db", -999) > -10:
        tags.append("MIX_INCONSISTENT")
    if abs(row.get("gain_error_db_h", 0)) > 6:
        tags.append("HS_GAIN_ERROR")
    if abs(row.get("gain_error_db_l", 0)) > 6:
        tags.append("LS_GAIN_ERROR")
    if row.get("possible_swap", 0) > 0:
        tags.append("POSSIBLE_SWAP")
    if row.get("ls_hat_corr_h_ref", 0) > 0.50 and row.get("si_sdr_l", 999) < 3:
        tags.append("HS_LEAK_IN_LS_OUTPUT")
    if row.get("hs_hat_corr_l_ref", 0) > 0.50 and row.get("si_sdr_h", 999) < 3:
        tags.append("LS_LEAK_IN_HS_OUTPUT")
    return ";".join(tags) if tags else "OK_OR_AMBIGUOUS"


@torch.no_grad()
def evaluate_files(
    model: nn.Module,
    files: Sequence[Tuple[Path, Path, Path]],
    fold: int,
    snr_lookup: Dict[str, str],
    sr: int,
    seg_samples: int,
    device: torch.device,
    apply_polarity_calibration: bool,
    apply_gain_calibration: bool,
) -> List[Dict[str, Any]]:
    rows = []

    for m_path, h_path, l_path in tqdm(files, desc=f"Fold {fold} inference"):
        sample_name = m_path.stem[2:]
        sample_id = f"M_{sample_name}"
        M = load_wav(m_path, sr=sr, seg_samples=seg_samples)
        H_ref = load_wav(h_path, sr=sr, seg_samples=seg_samples)
        L_ref = load_wav(l_path, sr=sr, seg_samples=seg_samples)

        M_t = torch.from_numpy(M).view(1, 1, -1).to(device)
        H_hat_t, L_hat_t = model(M_t)
        H_hat = H_hat_t.squeeze().detach().cpu().numpy().astype(np.float32)
        L_hat = L_hat_t.squeeze().detach().cpu().numpy().astype(np.float32)

        calib: Dict[str, float] = {}
        if apply_polarity_calibration:
            H_hat, L_hat, c = apply_global_polarity_calibration_np(M, H_hat, L_hat)
            calib.update(c)
        if apply_gain_calibration:
            H_hat, L_hat, c = apply_mixture_gain_calibration_np(M, H_hat, L_hat)
            calib.update(c)

        metrics = compute_metrics(M, H_ref, L_ref, H_hat, L_hat, sr=sr)
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "base_id": sample_name_to_base_id(sample_name),
            "fold": int(fold),
            "snr_label": get_snr_label(sample_id, snr_lookup),
            "m_path": str(m_path),
            "h_path": str(h_path),
            "l_path": str(l_path),
        }
        row.update(metrics)
        row.update(calib)
        row.update(spectral_features(M, sr, "M"))
        row.update(spectral_features(H_ref, sr, "H_ref"))
        row.update(spectral_features(L_ref, sr, "L_ref"))
        row.update(spectral_features(H_hat, sr, "H_hat"))
        row.update(spectral_features(L_hat, sr, "L_hat"))
        row["failure_tags"] = assign_failure_tags(row)
        rows.append(row)

    return rows


# -----------------------------------------------------------------------------
# Ranking, summaries, plots, audio
# -----------------------------------------------------------------------------


def mkdirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "metrics": out_dir / "metrics",
        "summaries": out_dir / "summaries",
        "rankings": out_dir / "rankings",
        "figures": out_dir / "figures",
        "audio": out_dir / "audio",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def rank_cases(df: pd.DataFrame, top_k: int) -> Dict[str, pd.DataFrame]:
    ranking_specs = {
        "worst_mean_si_sdr": ("mean_si_sdr", True),
        "worst_hs_si_sdr": ("si_sdr_h", True),
        "worst_ls_si_sdr": ("si_sdr_l", True),
        "worst_min_si_sdr": ("min_si_sdr", True),
        "worst_hs_sir": ("sir_h", True),
        "worst_ls_sir": ("sir_l", True),
        "worst_mix_corr": ("mix_corr", True),
        "worst_mix_nmse": ("mix_nmse_db", False),
        "worst_abs_gain_h": ("abs_gain_error_db_h", False),
        "worst_abs_gain_l": ("abs_gain_error_db_l", False),
        "possible_swaps": ("swap_gain_db", False),
        "hs_weakest_target": ("target_local_snr_db", True),
        "ls_weakest_target": ("target_local_snr_db", False),
        "highest_lung_zcr": ("L_ref_zcr", False),
        "highest_lung_centroid": ("L_ref_spectral_centroid", False),
    }

    work = df.copy()
    work["abs_gain_error_db_h"] = work["gain_error_db_h"].abs()
    work["abs_gain_error_db_l"] = work["gain_error_db_l"].abs()

    out = {}
    for name, (col, ascending) in ranking_specs.items():
        if col not in work.columns:
            continue
        sub = work.sort_values(col, ascending=ascending).head(top_k).copy()
        sub.insert(0, "rank_group", name)
        sub.insert(1, "rank", range(1, len(sub) + 1))
        out[name] = sub
    return out


def save_rankings(rankings: Dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    all_selected = []
    for name, rdf in rankings.items():
        path = out_dir / f"{name}.csv"
        rdf.to_csv(path, index=False)
        all_selected.append(rdf)
    selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    if not selected.empty:
        selected.to_csv(out_dir / "selected_cases_all_rankings_with_duplicates.csv", index=False)
        dedup = selected.drop_duplicates(subset=["sample_id"]).copy()
        dedup.to_csv(out_dir / "selected_cases_unique.csv", index=False)
        return dedup
    return selected


def numeric_summary(df: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    metrics = [
        "si_sdr_h", "si_sdr_l", "mean_si_sdr", "sir_h", "sir_l", "sar_h", "sar_l",
        "mix_corr", "mix_nmse_db", "mix_gain_error_db", "gain_error_db_h", "gain_error_db_l",
        "target_local_snr_db", "possible_swap",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if group_cols is None:
        rows = []
        for m in metrics:
            rows.append({
                "metric": m,
                "n": int(df[m].notna().sum()),
                "mean": float(df[m].mean()),
                "std": float(df[m].std()),
                "p05": float(df[m].quantile(0.05)),
                "p25": float(df[m].quantile(0.25)),
                "median": float(df[m].median()),
                "p75": float(df[m].quantile(0.75)),
                "p95": float(df[m].quantile(0.95)),
            })
        return pd.DataFrame(rows)
    return df.groupby(group_cols)[metrics].agg(["count", "mean", "std", "median"]).reset_index()


def save_failure_tag_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    for tags in df["failure_tags"].astype(str):
        for tag in tags.split(";"):
            counts[tag] = counts.get(tag, 0) + 1
    rows = [{"failure_tag": k, "count": v, "percentage": 100.0 * v / max(1, len(df))} for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "failure_tag_summary.csv", index=False)
    return out


def save_error_feature_correlations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    target_metrics = ["mean_si_sdr", "si_sdr_h", "si_sdr_l", "sir_h", "sir_l", "mix_corr", "mix_nmse_db"]
    feature_prefixes = ["M_", "H_ref_", "L_ref_", "H_hat_", "L_hat_"]
    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
    rows = []
    for metric in target_metrics:
        if metric not in df.columns:
            continue
        for col in feature_cols:
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df[metric], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 5 or x[valid].std() == 0 or y[valid].std() == 0:
                continue
            rows.append({
                "metric": metric,
                "feature": col,
                "pearson_corr": float(np.corrcoef(x[valid], y[valid])[0, 1]),
                "n": int(valid.sum()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_corr"] = out["pearson_corr"].abs()
        out = out.sort_values(["metric", "abs_corr"], ascending=[True, False])
    out.to_csv(out_dir / "error_feature_correlations.csv", index=False)
    return out


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s))[:180]


def peak_norm(x: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(x)) + 1e-8)
    return (x * (target_peak / peak)).astype(np.float32)


def plot_waveform_bundle(
    row: pd.Series,
    M: np.ndarray,
    H_ref: np.ndarray,
    L_ref: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    out_path: Path,
    sr: int,
) -> None:
    t = np.arange(len(M)) / sr
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(t, M, linewidth=0.8)
    axes[0].set_title("Mixture M")
    axes[1].plot(t, H_ref, linewidth=0.8, label="H_ref")
    axes[1].plot(t, H_hat, linewidth=0.8, alpha=0.8, label="H_hat")
    axes[1].legend(loc="upper right")
    axes[1].set_title(f"HS | SI-SDR={row['si_sdr_h']:+.2f} dB | SIR={row['sir_h']:+.2f} dB | gain={row['gain_error_db_h']:+.2f} dB")
    axes[2].plot(t, L_ref, linewidth=0.8, label="L_ref")
    axes[2].plot(t, L_hat, linewidth=0.8, alpha=0.8, label="L_hat")
    axes[2].legend(loc="upper right")
    axes[2].set_title(f"LS | SI-SDR={row['si_sdr_l']:+.2f} dB | SIR={row['sir_l']:+.2f} dB | gain={row['gain_error_db_l']:+.2f} dB")
    axes[3].plot(t, H_ref - H_hat, linewidth=0.7, label="H residual")
    axes[3].plot(t, L_ref - L_hat, linewidth=0.7, alpha=0.8, label="L residual")
    axes[3].legend(loc="upper right")
    axes[3].set_title(f"Residuals | mix_corr={row['mix_corr']:+.3f} | mix_NMSE={row['mix_nmse_db']:+.2f} dB | local SNR={row['target_local_snr_db']:+.2f} dB")
    axes[3].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Amp.")
    fig.suptitle(f"{row['rank_group']} #{int(row['rank'])} | {row['sample_id']} | tags={row['failure_tags']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram_bundle(
    row: pd.Series,
    signals: Dict[str, np.ndarray],
    out_path: Path,
    sr: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    names = ["M", "H_ref", "H_hat", "L_ref", "L_hat", "H+L residual"]
    data = [signals["M"], signals["H_ref"], signals["H_hat"], signals["L_ref"], signals["L_hat"], signals["M"] - (signals["H_hat"] + signals["L_hat"])]
    for ax, name, x in zip(axes.reshape(-1), names, data):
        ax.specgram(x, NFFT=256, Fs=sr, noverlap=192)
        ax.set_title(name)
        ax.set_ylim(0, 1800)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz")
    fig.suptitle(f"Spectrograms | {row['sample_id']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def export_selected_case_artifacts(
    selected: pd.DataFrame,
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
    dirs: Dict[str, Path],
) -> None:
    if selected.empty:
        return

    # Group by checkpoint/fold to avoid reloading model too many times.
    for fold, sub in selected.groupby("fold"):
        fold = int(fold)
        ckpt = Path(args.ckpt) if args.ckpt else resolve_fold_checkpoint(Path(args.ckpt_dir), fold)
        if ckpt is None or not ckpt.exists():
            print(f"[WARN] cannot export artifacts for fold={fold}: checkpoint not found")
            continue

        print(f"Export artifacts | fold={fold} | ckpt={ckpt}")
        model = load_checkpoint_model(ckpt, cfg, T=args.seg_samples, device=device)

        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"Export fold {fold}"):
            sample_name = str(row["sample_name"])
            M = load_wav(Path(row["m_path"]), args.sr, args.seg_samples)
            H_ref = load_wav(Path(row["h_path"]), args.sr, args.seg_samples)
            L_ref = load_wav(Path(row["l_path"]), args.sr, args.seg_samples)

            M_t = torch.from_numpy(M).view(1, 1, -1).to(device)
            H_hat_t, L_hat_t = model(M_t)
            H_hat = H_hat_t.squeeze().detach().cpu().numpy().astype(np.float32)
            L_hat = L_hat_t.squeeze().detach().cpu().numpy().astype(np.float32)

            if args.apply_polarity_calibration:
                H_hat, L_hat, _ = apply_global_polarity_calibration_np(M, H_hat, L_hat)
            if args.apply_gain_calibration:
                H_hat, L_hat, _ = apply_mixture_gain_calibration_np(M, H_hat, L_hat)

            group_dir = dirs["figures"] / str(row["rank_group"])
            group_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{int(row['rank']):02d}_{sanitize_filename(row['sample_id'])}"

            plot_waveform_bundle(row, M, H_ref, L_ref, H_hat, L_hat, group_dir / f"{stem}_waveforms.png", sr=args.sr)
            plot_spectrogram_bundle(row, {"M": M, "H_ref": H_ref, "L_ref": L_ref, "H_hat": H_hat, "L_hat": L_hat}, group_dir / f"{stem}_spectrograms.png", sr=args.sr)

            if args.save_audio:
                audio_dir = dirs["audio"] / str(row["rank_group"]) / stem
                audio_dir.mkdir(parents=True, exist_ok=True)
                sf.write(audio_dir / "M_mix.wav", peak_norm(M), args.sr)
                sf.write(audio_dir / "H_ref.wav", peak_norm(H_ref), args.sr)
                sf.write(audio_dir / "H_hat.wav", peak_norm(H_hat), args.sr)
                sf.write(audio_dir / "L_ref.wav", peak_norm(L_ref), args.sr)
                sf.write(audio_dir / "L_hat.wav", peak_norm(L_hat), args.sr)
                sf.write(audio_dir / "mix_reconstructed_Hhat_plus_Lhat.wav", peak_norm(H_hat + L_hat), args.sr)
                sf.write(audio_dir / "mix_residual_M_minus_Hhat_Lhat.wav", peak_norm(M - (H_hat + L_hat)), args.sr)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Mixed+SSL error analysis")
    parser.add_argument("--code-root", default="", help="Directory containing model_config.py, encoder.py, separator.py, decoder.py")
    parser.add_argument("--dataset-dir", default="", help="Directory containing M_*.wav, H_*.wav, L_*.wav")
    parser.add_argument("--split-csv", default="", help="Source-disjoint split CSV. Recommended.")
    parser.add_argument("--ckpt", default="", help="Single checkpoint path. Use with --fold.")
    parser.add_argument("--ckpt-dir", default="", help="Checkpoint directory for fold checkpoints.")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for single checkpoint evaluation. 0 means evaluate all data unless split omitted.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for --ckpt-dir mode")
    parser.add_argument("--out-dir", default="", help="Output directory for error analysis")
    parser.add_argument("--sr", type=int, default=4000)
    parser.add_argument("--seg-samples", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--only-orig", action="store_true", default=True, help="Evaluate only *_orig segments")
    parser.add_argument("--include-augmented", action="store_false", dest="only_orig", help="Evaluate all segments, not only *_orig")
    parser.add_argument("--apply-polarity-calibration", action="store_true", default=True)
    parser.add_argument("--no-polarity-calibration", action="store_false", dest="apply_polarity_calibration")
    parser.add_argument("--apply-gain-calibration", action="store_true", default=True)
    parser.add_argument("--no-gain-calibration", action="store_false", dest="apply_gain_calibration")
    parser.add_argument("--save-audio", action="store_true", help="Export WAV bundles for selected cases")
    parser.add_argument("--max-artifact-cases", type=int, default=120, help="Limit number of unique cases exported as plots/audio")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def write_run_config(args: argparse.Namespace, cfg: Any, out_dir: Path) -> None:
    cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper() and isinstance(getattr(cfg, k), (str, int, float, bool, tuple, list, type(None)))}
    payload = {
        "args": vars(args),
        "model_config_subset": cfg_dict,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    add_code_paths(args.code_root or None)
    cfg = load_model_config()

    dataset_dir = Path(args.dataset_dir or getattr(cfg, "SUPERVISED_DIR", ""))
    split_csv = Path(args.split_csv or getattr(cfg, "SOURCE_DISJOINT_SPLIT_CSV", "")) if (args.split_csv or getattr(cfg, "SOURCE_DISJOINT_SPLIT_CSV", "")) else None
    ckpt_dir = Path(args.ckpt_dir or getattr(cfg, "CKPT_DIR", "")) if (args.ckpt_dir or getattr(cfg, "CKPT_DIR", "")) else None
    out_dir = Path(args.out_dir or (Path(getattr(cfg, "PROJECT_ROOT", ".")) / "outputs" / "results" / "ERROR_ANALYSIS" / "final_mixed_ssl"))
    dirs = mkdirs(out_dir)
    write_run_config(args, cfg, out_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("=" * 100)
    print("FINAL MIXED+SSL ERROR ANALYSIS")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Split CSV: {split_csv}")
    print(f"Checkpoint: {args.ckpt or ''}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Output: {out_dir}")
    print(f"Calibration: polarity={args.apply_polarity_calibration}, gain={args.apply_gain_calibration}")
    print("=" * 100)

    all_rows: List[Dict[str, Any]] = []

    if args.ckpt:
        if args.fold <= 0 and split_csv is not None:
            raise ValueError("When using --ckpt with --split-csv, pass --fold <fold_number> so validation bases can be selected.")
        fold = int(args.fold) if args.fold > 0 else 1
        files, snr_lookup, val_bases = build_eval_file_list(dataset_dir, split_csv, fold if split_csv else None, only_orig=args.only_orig)
        print(f"Single checkpoint mode | fold={fold} | files={len(files)} | val_bases={len(val_bases) if val_bases else 'all'}")
        model = load_checkpoint_model(Path(args.ckpt), cfg, T=args.seg_samples, device=device)
        rows = evaluate_files(
            model=model,
            files=files,
            fold=fold,
            snr_lookup=snr_lookup,
            sr=args.sr,
            seg_samples=args.seg_samples,
            device=device,
            apply_polarity_calibration=args.apply_polarity_calibration,
            apply_gain_calibration=args.apply_gain_calibration,
        )
        all_rows.extend(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    else:
        if ckpt_dir is None or not ckpt_dir.exists():
            raise FileNotFoundError("Checkpoint directory not found. Provide --ckpt or --ckpt-dir.")
        for fold in range(1, args.n_folds + 1):
            ckpt = resolve_fold_checkpoint(ckpt_dir, fold)
            if ckpt is None:
                print(f"[WARN] missing checkpoint for fold {fold} in {ckpt_dir}; skipping")
                continue
            files, snr_lookup, val_bases = build_eval_file_list(dataset_dir, split_csv, fold if split_csv else None, only_orig=args.only_orig)
            print(f"Fold mode | fold={fold} | ckpt={ckpt} | files={len(files)} | val_bases={len(val_bases) if val_bases else 'all'}")
            model = load_checkpoint_model(ckpt, cfg, T=args.seg_samples, device=device)
            rows = evaluate_files(
                model=model,
                files=files,
                fold=fold,
                snr_lookup=snr_lookup,
                sr=args.sr,
                seg_samples=args.seg_samples,
                device=device,
                apply_polarity_calibration=args.apply_polarity_calibration,
                apply_gain_calibration=args.apply_gain_calibration,
            )
            all_rows.extend(rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not all_rows:
        raise RuntimeError("No rows evaluated. Check checkpoint paths and dataset split.")

    df = pd.DataFrame(all_rows)
    per_sample_path = dirs["metrics"] / "per_sample_error_metrics.csv"
    df.to_csv(per_sample_path, index=False)
    print(f"Per-sample metrics saved: {per_sample_path}")

    # Summaries
    numeric_summary(df).to_csv(dirs["summaries"] / "overall_metric_distribution.csv", index=False)
    numeric_summary(df, ["fold"]).to_csv(dirs["summaries"] / "summary_by_fold.csv", index=False)
    if "snr_label" in df.columns:
        numeric_summary(df, ["snr_label"]).to_csv(dirs["summaries"] / "summary_by_snr.csv", index=False)
    if "local_snr_bin" in df.columns:
        numeric_summary(df, ["local_snr_bin"]).to_csv(dirs["summaries"] / "summary_by_local_snr_bin.csv", index=False)
    save_failure_tag_summary(df, dirs["summaries"])
    save_error_feature_correlations(df, dirs["summaries"])

    # Rankings
    rankings = rank_cases(df, top_k=args.top_k)
    selected = save_rankings(rankings, dirs["rankings"])
    selected_for_artifacts = selected.head(args.max_artifact_cases).copy() if not selected.empty else selected
    selected_for_artifacts.to_csv(dirs["rankings"] / "selected_cases_for_artifact_export.csv", index=False)

    # Plots/audio
    export_selected_case_artifacts(selected_for_artifacts, cfg, args, device, dirs)

    # Compact text summary
    overall = numeric_summary(df)
    tag_summary = pd.read_csv(dirs["summaries"] / "failure_tag_summary.csv")
    with open(dirs["summaries"] / "error_analysis_readme_summary.txt", "w") as f:
        f.write("FINAL MIXED+SSL ERROR ANALYSIS\n")
        f.write("================================\n\n")
        f.write(f"Evaluated segments: {len(df)}\n")
        f.write(f"Output directory: {out_dir}\n\n")
        f.write("Main metrics distribution:\n")
        f.write(overall.to_string(index=False))
        f.write("\n\nFailure tag summary:\n")
        f.write(tag_summary.to_string(index=False))
        f.write("\n\nInterpretation guide:\n")
        f.write("- target_local_snr_db > +15: lung source locally weak / heart dominant.\n")
        f.write("- target_local_snr_db < -15: heart source locally weak / lung dominant.\n")
        f.write("- mix_corr < 0.95 or mix_nmse_db > -10: H_hat + L_hat does not reconstruct M well.\n")
        f.write("- possible_swap=1: swapped assignment gives >3 dB better average SI-SDR.\n")
        f.write("- high L_ref_zcr / L_ref_spectral_centroid can support a morphology/domain-gap explanation.\n")

    print("=" * 100)
    print("DONE")
    print(f"Per-sample CSV: {per_sample_path}")
    print(f"Rankings: {dirs['rankings']}")
    print(f"Summaries: {dirs['summaries']}")
    print(f"Figures: {dirs['figures']}")
    if args.save_audio:
        print(f"Audio: {dirs['audio']}")
    print("=" * 100)


if __name__ == "__main__":
    main()

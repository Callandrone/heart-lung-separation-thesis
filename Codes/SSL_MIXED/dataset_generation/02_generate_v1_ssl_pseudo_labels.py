#!/usr/bin/env python3
"""
02_generate_v1_ssl_pseudo_labels.py

STEP 2 SSL / V1 real mixtures, teacher pseudo-label generation.

Input:
    PROJECT_ROOT/dataset/processed/M1_segmented_only/
        M_*.wav
        manifest_m1_segmented_only.csv

Teacher:
    Stage-2 mixed-domain checkpoint (original experiment directory supported for backward compatibility).

Output:
    PROJECT_ROOT/dataset/processed/v1_real_ssl_pseudo_from_mixed_noaug/
        all/
            M_*.wav
            H_*.wav
            L_*.wav
        confident/
            M_*.wav
            H_*.wav
            L_*.wav
        manifest_pseudo_all.csv
        manifest_pseudo_confident.csv
        manifest_pseudo_rejected.csv
        summary.json
        summary.txt

Methodological choice:
- V1 real H/L files are NOT used.
- The teacher receives only M segments.
- H_pseudo and L_pseudo are teacher outputs, not ground truth.
- Mixture-informed polarity/gain calibration is input-only: it uses M and H_hat+L_hat,
  never true H/L references.
- A conservative input-only confidence filter is applied. The 'confident' folder is the
  one to use for the next SSL student training pilot.

Default run:
    python 02_generate_v1_ssl_pseudo_labels.py --overwrite

If the script cannot find encoder.py / separator.py / decoder.py automatically:
    python 02_generate_v1_ssl_pseudo_labels.py --model-code-dir /path/to/heart-lung-separation-thesis/Codes/MIXED_TUNING --overwrite

If the checkpoint has a different path:
    python 02_generate_v1_ssl_pseudo_labels.py --teacher-ckpt /path/to/finetune_fold1_best.pt --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("ESD_JASSNET_ROOT", str(REPO_ROOT))).expanduser().resolve()
DEFAULT_M1_DIR = PROJECT_ROOT / "dataset" / "processed" / "M1_segmented_only"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "v1_real_ssl_pseudo_from_mixed_noaug"
DEFAULT_TEACHER_EXPERIMENT = "EXP_H_FULL_TO_TORABI_FOLD2_MIXED_REPLAY005_NOAUG"
TARGET_SR = 4000
SEG_SAMPLES = 8000
WAV_SUBTYPE = "FLOAT"
EPS = 1e-8


@dataclass
class BatchItem:
    name: str
    base_id: str
    segment_index: int
    m_path: Path
    row: Dict[str, object]


# -----------------------------------------------------------------------------
# Path discovery
# -----------------------------------------------------------------------------

def has_model_files(path: Path) -> bool:
    return all((path / f).exists() for f in ("encoder.py", "separator.py", "decoder.py", "model_config.py"))


def resolve_model_code_dir(explicit: Optional[Path], project_root: Path) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir.parent,
            project_root / "Codes" / "SSL_MIXED",
            project_root / "Codes" / "MIXED_TUNING",
            project_root / "codes" / "SSL_MIXED",  # legacy local layout
            project_root / "codes" / "MIXED_TUNING",  # legacy local layout
        ]
    )

    for c in candidates:
        if c.exists() and c.is_dir() and has_model_files(c):
            return c

    raise RuntimeError(
        "Could not find model code files encoder.py, separator.py, decoder.py, model_config.py.\n"
        "Pass --model-code-dir explicitly. Checked:\n"
        + "\n".join(str(c) for c in candidates)
    )


def resolve_teacher_ckpt(explicit: Optional[Path], project_root: Path, experiment_name: str) -> Path:
    if explicit is not None:
        if explicit.exists() and explicit.is_file():
            return explicit
        raise RuntimeError(f"Teacher checkpoint not found: {explicit}")

    ckpt_dir = project_root / "outputs" / "checkpoints" / experiment_name
    candidates = [
        ckpt_dir / "finetune_fold1_best.pt",
        ckpt_dir / "scratch_fold1_best.pt",
        ckpt_dir / "fold1_best.pt",
        ckpt_dir / "best.pt",
        ckpt_dir / "model_best.pt",
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    if ckpt_dir.exists():
        discovered = sorted(
            list(ckpt_dir.glob("*fold1*best*.pt"))
            + list(ckpt_dir.glob("*best*.pt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if discovered:
            return discovered[0]

    raise RuntimeError(
        f"Could not auto-discover teacher checkpoint for experiment {experiment_name}.\n"
        f"Expected directory: {ckpt_dir}\n"
        "Pass --teacher-ckpt explicitly."
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def import_model_modules(model_code_dir: Path):
    sys.path.insert(0, str(model_code_dir))
    import model_config as cfg  # type: ignore
    from encoder import CardiopulmonaryEncoder  # type: ignore
    from separator import CardiopulmonarySeparator  # type: ignore
    from decoder import CardiopulmonaryDecoder  # type: ignore

    return cfg, CardiopulmonaryEncoder, CardiopulmonarySeparator, CardiopulmonaryDecoder


def build_model(model_code_dir: Path, device: torch.device) -> nn.Module:
    cfg, CardiopulmonaryEncoder, CardiopulmonarySeparator, CardiopulmonaryDecoder = import_model_modules(model_code_dir)

    class CardiopulmonaryNet(nn.Module):
        def __init__(self, T: int):
            super().__init__()
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
                n_stacks=cfg.SEP_N_STACKS,
                S=cfg.SEP_BLOCKS_PER_STACK,
                tcn_bottleneck=cfg.SEP_TCN_BOTTLENECK,
                attn_heads=cfg.ATTN_HEADS,
                attn_window=cfg.ATTN_WINDOW,
                n_global=cfg.N_GLOBAL_TOKENS,
                dropout_p=cfg.DROPOUT_P,
                mask_scale=cfg.MASK_SCALE,
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

    model = CardiopulmonaryNet(T=SEG_SAMPLES).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model code dir: {model_code_dir}")
    print(f"Total trainable parameters: {total:,}")
    return model


def extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            value = ckpt_obj.get(key)
            if isinstance(value, dict):
                return value
        # Some checkpoints are raw state_dicts.
        if ckpt_obj and all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj  # type: ignore[return-value]
    raise RuntimeError("Unsupported checkpoint format: no model_state/state_dict found.")


def clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "_orig_mod."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v
    return cleaned


def load_teacher_model(model: nn.Module, ckpt_path: Path, device: torch.device, allow_partial: bool = False) -> Dict[str, object]:
    print(f"Loading teacher checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = clean_state_dict_keys(extract_state_dict(ckpt))
    result = model.load_state_dict(state, strict=not allow_partial)
    model.eval()

    info: Dict[str, object] = {
        "teacher_ckpt": str(ckpt_path),
        "allow_partial_load": bool(allow_partial),
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
    }

    if isinstance(ckpt, dict):
        for k in ("epoch", "fold", "stage", "val_loss", "val_metrics", "config"):
            if k in ckpt:
                try:
                    json.dumps(ckpt[k], default=str)
                    info[f"checkpoint_{k}"] = ckpt[k]
                except Exception:
                    info[f"checkpoint_{k}"] = str(ckpt[k])

    return info


# -----------------------------------------------------------------------------
# Audio + metrics
# -----------------------------------------------------------------------------

def load_segment(path: Path, sr: int, seg_samples: int) -> np.ndarray:
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if int(file_sr) != int(sr):
        raise ValueError(f"Unexpected sample rate for {path}: expected {sr}, got {file_sr}")
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) > seg_samples:
        audio = audio[:seg_samples]
    elif len(audio) < seg_samples:
        audio = np.pad(audio, (0, seg_samples - len(audio))).astype(np.float32)
    return audio.astype(np.float32)


def rms_np(x: np.ndarray, eps: float = EPS) -> float:
    x0 = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    return float(np.sqrt(np.mean(x0 * x0) + eps))


def corr_np(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    a0 = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    b0 = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = float(np.sqrt(np.dot(a0, a0) * np.dot(b0, b0)) + eps)
    return float(np.dot(a0, b0) / denom)


def mixture_consistency_metrics_np(mixture: np.ndarray, H_hat: np.ndarray, L_hat: np.ndarray, eps: float = EPS) -> Dict[str, float]:
    mix = np.asarray(mixture, dtype=np.float64) - float(np.mean(mixture))
    reconstructed = np.asarray(H_hat + L_hat, dtype=np.float64)
    reconstructed = reconstructed - float(np.mean(reconstructed))

    residual = mix - reconstructed
    mix_energy = float(np.dot(mix, mix))
    reconstructed_energy = float(np.dot(reconstructed, reconstructed))

    if mix_energy < eps:
        return {
            "mix_nmse": float("nan"),
            "mix_nmse_db": float("nan"),
            "mix_gain_error_db": float("nan"),
            "mix_gamma": float("nan"),
            "negative_mix_gamma": float("nan"),
            "mix_corr": float("nan"),
        }

    nmse = float(np.dot(residual, residual) / (mix_energy + eps))
    nmse = max(nmse, eps)
    nmse_db = float(10.0 * np.log10(nmse))

    rms_mix = float(np.sqrt(np.mean(mix * mix) + eps))
    rms_rec = float(np.sqrt(np.mean(reconstructed * reconstructed) + eps))
    gain_error_db = float(20.0 * np.log10(rms_rec / rms_mix))

    gamma = float(np.dot(mix, reconstructed) / (reconstructed_energy + eps))
    corr = float(np.dot(mix, reconstructed) / (np.sqrt(mix_energy * reconstructed_energy) + eps))

    return {
        "mix_nmse": nmse,
        "mix_nmse_db": nmse_db,
        "mix_gain_error_db": gain_error_db,
        "mix_gamma": gamma,
        "negative_mix_gamma": float(gamma < 0.0),
        "mix_corr": corr,
    }


def apply_mixture_polarity_calibration_np(mixture: np.ndarray, H_hat: np.ndarray, L_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    before = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    should_flip = bool(before["mix_gamma"] < 0.0) if not math.isnan(before["mix_gamma"]) else False
    if should_flip:
        H_hat = -H_hat
        L_hat = -L_hat
    after = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    return H_hat, L_hat, {
        "polarity_flipped": float(should_flip),
        "mix_gamma_before_polarity": float(before["mix_gamma"]),
        "mix_corr_before_polarity": float(before["mix_corr"]),
        "mix_gamma_after_polarity": float(after["mix_gamma"]),
        "mix_corr_after_polarity": float(after["mix_corr"]),
    }


def apply_mixture_gain_calibration_np(mixture: np.ndarray, H_hat: np.ndarray, L_hat: np.ndarray, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    before = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    mix = np.asarray(mixture, dtype=np.float64) - float(np.mean(mixture))
    pred = np.asarray(H_hat + L_hat, dtype=np.float64)
    pred = pred - float(np.mean(pred))
    pred_energy = float(np.dot(pred, pred))
    gamma = 1.0 if pred_energy < eps else float(np.dot(mix, pred) / (pred_energy + eps))
    H_hat = (gamma * H_hat).astype(np.float32)
    L_hat = (gamma * L_hat).astype(np.float32)
    after = mixture_consistency_metrics_np(mixture, H_hat, L_hat)
    return H_hat, L_hat, {
        "gain_calibration_gamma": float(gamma),
        "mix_nmse_db_before_gain": float(before["mix_nmse_db"]),
        "mix_corr_before_gain": float(before["mix_corr"]),
        "mix_gain_error_db_before_gain": float(before["mix_gain_error_db"]),
        "mix_nmse_db_after_gain": float(after["mix_nmse_db"]),
        "mix_corr_after_gain": float(after["mix_corr"]),
        "mix_gain_error_db_after_gain": float(after["mix_gain_error_db"]),
    }


def compute_pseudo_audit(m: np.ndarray, h: np.ndarray, l: np.ndarray) -> Dict[str, float]:
    metrics = mixture_consistency_metrics_np(m, h, l)
    rms_m = rms_np(m)
    rms_h = rms_np(h)
    rms_l = rms_np(l)
    energy_h = rms_h * rms_h
    energy_l = rms_l * rms_l
    total_src_energy = energy_h + energy_l + EPS
    pseudo_snr_db = float(20.0 * np.log10(rms_h / max(rms_l, EPS)))
    hl_corr = corr_np(h, l)

    return {
        **{f"final_{k}": float(v) for k, v in metrics.items()},
        "rms_m": float(rms_m),
        "rms_h_pseudo": float(rms_h),
        "rms_l_pseudo": float(rms_l),
        "pseudo_h_over_l_snr_db": pseudo_snr_db,
        "abs_pseudo_h_over_l_snr_db": float(abs(pseudo_snr_db)),
        "pseudo_h_energy_fraction": float(energy_h / total_src_energy),
        "pseudo_l_energy_fraction": float(energy_l / total_src_energy),
        "pseudo_h_l_corr": float(hl_corr),
        "abs_pseudo_h_l_corr": float(abs(hl_corr)),
        "peak_m": float(np.max(np.abs(m))),
        "peak_h_pseudo": float(np.max(np.abs(h))),
        "peak_l_pseudo": float(np.max(np.abs(l))),
    }


def is_confident(row: Dict[str, object], args: argparse.Namespace) -> Tuple[bool, str]:
    reasons: List[str] = []

    def get_float(key: str) -> float:
        try:
            return float(row.get(key, float("nan")))
        except Exception:
            return float("nan")

    mix_corr = get_float("final_mix_corr")
    mix_nmse_db = get_float("final_mix_nmse_db")
    rms_h = get_float("rms_h_pseudo")
    rms_l = get_float("rms_l_pseudo")
    abs_snr = get_float("abs_pseudo_h_over_l_snr_db")
    abs_hl_corr = get_float("abs_pseudo_h_l_corr")
    gamma = get_float("gain_calibration_gamma")

    if not np.isfinite(mix_corr) or mix_corr < args.min_mix_corr:
        reasons.append(f"mix_corr<{args.min_mix_corr}")
    if not np.isfinite(mix_nmse_db) or mix_nmse_db > args.max_mix_nmse_db:
        reasons.append(f"mix_nmse_db>{args.max_mix_nmse_db}")
    if not np.isfinite(rms_h) or rms_h < args.min_source_rms:
        reasons.append(f"rms_h<{args.min_source_rms}")
    if not np.isfinite(rms_l) or rms_l < args.min_source_rms:
        reasons.append(f"rms_l<{args.min_source_rms}")
    if not np.isfinite(abs_snr) or abs_snr > args.max_abs_pseudo_snr_db:
        reasons.append(f"abs_pseudo_snr>{args.max_abs_pseudo_snr_db}")
    if np.isfinite(abs_hl_corr) and abs_hl_corr > args.max_abs_h_l_corr:
        reasons.append(f"abs_h_l_corr>{args.max_abs_h_l_corr}")
    if np.isfinite(gamma) and not (args.min_gain_gamma <= gamma <= args.max_gain_gamma):
        reasons.append(f"gain_gamma_not_in_[{args.min_gain_gamma},{args.max_gain_gamma}]")

    return len(reasons) == 0, ";".join(reasons) if reasons else "OK"


# -----------------------------------------------------------------------------
# Manifest handling
# -----------------------------------------------------------------------------

def load_m1_manifest(m1_dir: Path) -> List[BatchItem]:
    manifest_path = m1_dir / "manifest_m1_segmented_only.csv"
    rows: List[BatchItem] = []

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        if "m_path" not in df.columns:
            raise RuntimeError(f"Manifest exists but has no m_path column: {manifest_path}")
        for _, r in df.iterrows():
            m_path = Path(str(r["m_path"]))
            name = str(r.get("name", m_path.stem[2:] if m_path.stem.startswith("M_") else m_path.stem))
            base_id = str(r.get("base_id", name.split("_s")[0]))
            try:
                segment_index = int(r.get("segment_index", -1))
            except Exception:
                segment_index = -1
            rows.append(BatchItem(name=name, base_id=base_id, segment_index=segment_index, m_path=m_path, row=r.to_dict()))
    else:
        for p in sorted(m1_dir.glob("M_*.wav")):
            name = p.stem[2:]
            base_id = name.split("_s")[0]
            seg_idx = -1
            if "_s" in name:
                try:
                    seg_idx = int(name.split("_s", 1)[1].split("_", 1)[0])
                except Exception:
                    pass
            rows.append(BatchItem(name=name, base_id=base_id, segment_index=seg_idx, m_path=p, row={}))

    if not rows:
        raise RuntimeError(f"No M-only segments found in {m1_dir}")

    missing = [x.m_path for x in rows if not x.m_path.exists()]
    if missing:
        raise RuntimeError("Some M paths from manifest do not exist. First missing:\n" + "\n".join(str(p) for p in missing[:10]))

    rows = sorted(rows, key=lambda x: (x.base_id, x.segment_index, x.name))
    return rows


def write_triplet(folder: Path, name: str, m: np.ndarray, h: np.ndarray, l: np.ndarray, sr: int) -> Dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    m_out = folder / f"M_{name}.wav"
    h_out = folder / f"H_{name}.wav"
    l_out = folder / f"L_{name}.wav"
    sf.write(str(m_out), m.astype(np.float32), sr, subtype=WAV_SUBTYPE)
    sf.write(str(h_out), h.astype(np.float32), sr, subtype=WAV_SUBTYPE)
    sf.write(str(l_out), l.astype(np.float32), sr, subtype=WAV_SUBTYPE)
    return {"m_pseudo_path": str(m_out), "h_pseudo_path": str(h_out), "l_pseudo_path": str(l_out)}


def batched(items: List[BatchItem], batch_size: int) -> Iterable[List[BatchItem]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# -----------------------------------------------------------------------------
# Main generation
# -----------------------------------------------------------------------------

def generate_pseudo_labels(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_code_dir = resolve_model_code_dir(args.model_code_dir, args.project_root)
    teacher_ckpt = resolve_teacher_ckpt(args.teacher_ckpt, args.project_root, args.teacher_experiment)

    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory already exists and is not empty: {args.out_dir}. Use --overwrite.")

    all_dir = args.out_dir / "all"
    confident_dir = args.out_dir / "confident"
    rejected_dir = args.out_dir / "rejected"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_dir.mkdir(parents=True, exist_ok=True)
    confident_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rejected_triplets:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    items = load_m1_manifest(args.m1_dir)
    model = build_model(model_code_dir, device)
    ckpt_info = load_teacher_model(model, teacher_ckpt, device, allow_partial=args.allow_partial_load)

    rows_all: List[Dict[str, object]] = []
    use_amp = bool(args.amp and device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(list(batched(items, args.batch_size)), desc="Generating teacher pseudo-labels"):
            m_np_list = [load_segment(x.m_path, args.sr, SEG_SAMPLES) for x in batch]
            m_tensor = torch.from_numpy(np.stack(m_np_list, axis=0)).unsqueeze(1).float().to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    h_t, l_t = model(m_tensor)
            else:
                h_t, l_t = model(m_tensor)

            h_np_batch = h_t.detach().cpu().numpy()
            l_np_batch = l_t.detach().cpu().numpy()

            if h_np_batch.ndim == 3:
                h_np_batch = h_np_batch[:, 0, :]
            if l_np_batch.ndim == 3:
                l_np_batch = l_np_batch[:, 0, :]

            for idx, item in enumerate(batch):
                m = m_np_list[idx].astype(np.float32)
                h_raw = np.asarray(h_np_batch[idx], dtype=np.float32)[:SEG_SAMPLES]
                l_raw = np.asarray(l_np_batch[idx], dtype=np.float32)[:SEG_SAMPLES]

                if len(h_raw) < SEG_SAMPLES:
                    h_raw = np.pad(h_raw, (0, SEG_SAMPLES - len(h_raw))).astype(np.float32)
                if len(l_raw) < SEG_SAMPLES:
                    l_raw = np.pad(l_raw, (0, SEG_SAMPLES - len(l_raw))).astype(np.float32)

                pre_metrics = mixture_consistency_metrics_np(m, h_raw, l_raw)

                h_cal = h_raw.copy()
                l_cal = l_raw.copy()
                pol_info: Dict[str, float] = {
                    "polarity_flipped": 0.0,
                    "mix_gamma_before_polarity": float(pre_metrics["mix_gamma"]),
                    "mix_corr_before_polarity": float(pre_metrics["mix_corr"]),
                    "mix_gamma_after_polarity": float(pre_metrics["mix_gamma"]),
                    "mix_corr_after_polarity": float(pre_metrics["mix_corr"]),
                }
                gain_info: Dict[str, float] = {
                    "gain_calibration_gamma": 1.0,
                    "mix_nmse_db_before_gain": float(pre_metrics["mix_nmse_db"]),
                    "mix_corr_before_gain": float(pre_metrics["mix_corr"]),
                    "mix_gain_error_db_before_gain": float(pre_metrics["mix_gain_error_db"]),
                    "mix_nmse_db_after_gain": float(pre_metrics["mix_nmse_db"]),
                    "mix_corr_after_gain": float(pre_metrics["mix_corr"]),
                    "mix_gain_error_db_after_gain": float(pre_metrics["mix_gain_error_db"]),
                }

                if args.apply_mixture_polarity_calibration:
                    h_cal, l_cal, pol_info = apply_mixture_polarity_calibration_np(m, h_cal, l_cal)
                if args.apply_mixture_gain_calibration:
                    h_cal, l_cal, gain_info = apply_mixture_gain_calibration_np(m, h_cal, l_cal)

                audit = compute_pseudo_audit(m, h_cal, l_cal)

                row: Dict[str, object] = {
                    "name": item.name,
                    "base_id": item.base_id,
                    "segment_index": item.segment_index,
                    "source_m_path": str(item.m_path),
                    "teacher_ckpt": str(teacher_ckpt),
                    "teacher_experiment": args.teacher_experiment,
                    "sr": int(args.sr),
                    "segment_samples": int(SEG_SAMPLES),
                    "pseudo_label_type": "teacher_generated_not_ground_truth",
                    "uses_real_v1_h_l_targets": False,
                    "apply_mixture_polarity_calibration": bool(args.apply_mixture_polarity_calibration),
                    "apply_mixture_gain_calibration": bool(args.apply_mixture_gain_calibration),
                    **{f"pre_{k}": float(v) for k, v in pre_metrics.items()},
                    **pol_info,
                    **gain_info,
                    **audit,
                }

                confident, reason = is_confident(row, args)
                row["confidence_pass"] = bool(confident)
                row["confidence_reason"] = reason

                out_paths_all = write_triplet(all_dir, item.name, m, h_cal, l_cal, args.sr)
                row.update({f"all_{k}": v for k, v in out_paths_all.items()})

                if confident:
                    out_paths_conf = write_triplet(confident_dir, item.name, m, h_cal, l_cal, args.sr)
                    row.update({f"confident_{k}": v for k, v in out_paths_conf.items()})
                elif args.save_rejected_triplets:
                    out_paths_rej = write_triplet(rejected_dir, item.name, m, h_cal, l_cal, args.sr)
                    row.update({f"rejected_{k}": v for k, v in out_paths_rej.items()})

                rows_all.append(row)

    df = pd.DataFrame(rows_all)
    df_conf = df[df["confidence_pass"] == True].copy()
    df_rej = df[df["confidence_pass"] != True].copy()

    manifest_all = args.out_dir / "manifest_pseudo_all.csv"
    manifest_conf = args.out_dir / "manifest_pseudo_confident.csv"
    manifest_rej = args.out_dir / "manifest_pseudo_rejected.csv"
    df.to_csv(manifest_all, index=False)
    df_conf.to_csv(manifest_conf, index=False)
    df_rej.to_csv(manifest_rej, index=False)

    # Compatibility aliases for future scripts.
    df.to_csv(args.out_dir / "manifest_ssl_pseudo_all.csv", index=False)
    df_conf.to_csv(args.out_dir / "manifest_ssl_pseudo_confident.csv", index=False)

    by_base = df.groupby("base_id")["confidence_pass"].agg(["count", "sum", "mean"]).reset_index()
    by_base.to_csv(args.out_dir / "audit_confidence_by_mixture.csv", index=False)

    numeric_cols = [
        "pre_mix_corr",
        "pre_mix_nmse_db",
        "final_mix_corr",
        "final_mix_nmse_db",
        "rms_m",
        "rms_h_pseudo",
        "rms_l_pseudo",
        "pseudo_h_over_l_snr_db",
        "abs_pseudo_h_over_l_snr_db",
        "pseudo_h_l_corr",
        "gain_calibration_gamma",
    ]
    audit_stats = {}
    for col in numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            audit_stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p05": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }

    summary: Dict[str, object] = {
        "dataset_name": args.out_dir.name,
        "methodological_role": "SSL teacher pseudo-label dataset generated from V1 real mixture-only segments",
        "m1_dir": str(args.m1_dir),
        "out_dir": str(args.out_dir),
        "all_dir": str(all_dir),
        "confident_dir": str(confident_dir),
        "teacher_experiment": args.teacher_experiment,
        "teacher_ckpt": str(teacher_ckpt),
        "model_code_dir": str(model_code_dir),
        "device": str(device),
        "n_input_segments": int(len(df)),
        "n_confident_segments": int(len(df_conf)),
        "n_rejected_segments": int(len(df_rej)),
        "confidence_rate": float(len(df_conf) / max(len(df), 1)),
        "n_input_mixtures": int(df["base_id"].nunique()),
        "manifest_pseudo_all": str(manifest_all),
        "manifest_pseudo_confident": str(manifest_conf),
        "manifest_pseudo_rejected": str(manifest_rej),
        "thresholds": {
            "min_mix_corr": float(args.min_mix_corr),
            "max_mix_nmse_db": float(args.max_mix_nmse_db),
            "min_source_rms": float(args.min_source_rms),
            "max_abs_pseudo_snr_db": float(args.max_abs_pseudo_snr_db),
            "max_abs_h_l_corr": float(args.max_abs_h_l_corr),
            "min_gain_gamma": float(args.min_gain_gamma),
            "max_gain_gamma": float(args.max_gain_gamma),
        },
        "calibration": {
            "apply_mixture_polarity_calibration": bool(args.apply_mixture_polarity_calibration),
            "apply_mixture_gain_calibration": bool(args.apply_mixture_gain_calibration),
            "input_only": True,
            "uses_ground_truth_h_l": False,
        },
        "checkpoint_info": ckpt_info,
        "audit_stats": audit_stats,
        "next_step": "Inspect manifest_pseudo_confident.csv and run pseudo-label quality audit/listening before SSL student fine-tuning.",
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    lines = [
        "V1 real SSL pseudo-label generation",
        "=" * 88,
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        "",
        "Recommended dataset for the SSL student pilot:",
        str(confident_dir),
        "",
        "Important:",
        "- H/L pseudo-labels are teacher outputs, not ground truth.",
        "- V1 raw H/L files were not used.",
        "- The confident folder is filtered using only input/prediction consistency metrics.",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SSL pseudo-labels from M1_segmented_only using the Stage-2 teacher.")
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model-code-dir", type=Path, default=None)
    p.add_argument("--teacher-experiment", type=str, default=DEFAULT_TEACHER_EXPERIMENT)
    p.add_argument("--teacher-ckpt", type=Path, default=None)

    p.add_argument("--sr", type=int, default=TARGET_SR)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, cuda:0. Default: auto.")
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast during inference.")
    p.add_argument("--allow-partial-load", action="store_true", help="Load checkpoint with strict=False. Use only for debugging.")

    p.add_argument("--apply-mixture-polarity-calibration", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--apply-mixture-gain-calibration", action=argparse.BooleanOptionalAction, default=True)

    # Conservative input-only pseudo-label quality thresholds.
    p.add_argument("--min-mix-corr", type=float, default=0.95)
    p.add_argument("--max-mix-nmse-db", type=float, default=-10.0)
    p.add_argument("--min-source-rms", type=float, default=1e-4)
    p.add_argument("--max-abs-pseudo-snr-db", type=float, default=25.0)
    p.add_argument("--max-abs-h-l-corr", type=float, default=0.98)
    p.add_argument("--min-gain-gamma", type=float, default=0.10)
    p.add_argument("--max-gain-gamma", type=float, default=10.0)

    p.add_argument("--save-rejected-triplets", action="store_true", help="Also write rejected M/H/L triplets to out_dir/rejected.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_pseudo_labels(args)
    print("\n" + "=" * 88)
    print("V1 REAL SSL PSEUDO-LABEL DATASET CREATED")
    print("=" * 88)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print("\nUse this for the next audit/student step:")
    print(summary["confident_dir"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
03_white_noise_robustness_audit.py
=================================

Zero-shot white-noise robustness audit for the final Mixed NOAUG + SSL model.

Goal:
  Evaluate how the final separator behaves when controlled additive white noise
  is injected into the mixture input:

      M_noisy = H + L + N

  The clean references remain:

      H_ref = H
      L_ref = L

This audit does NOT retrain the model. It only evaluates the final checkpoint
under controlled noisy inputs.

Main questions:
  1. How much do HS / LS SI-SDR degrade as noise increases?
  2. Does the model absorb the noise into H_hat / L_hat?
  3. Or does the noise remain mainly in the residual:
         R = M_noisy - (H_hat + L_hat) ?

Important interpretation:
  The model has only two outputs, H_hat and L_hat. It has no explicit noise branch.
  Therefore it cannot explicitly output "noise". If it suppresses the noise, the
  residual with respect to M_noisy should become similar to the injected noise N.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_config import physical_fold, project_root, target_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = project_root()
PHYSICAL_FOLD = physical_fold()
ERROR_ANALYSIS_SCRIPT = Path(__file__).resolve().with_name("01_error_analysis_final_mixed_ssl.py")
DATASET_DIR = target_dir(PROJECT_ROOT, PHYSICAL_FOLD)
SPLIT_CSV = Path(os.environ.get("HLSCMDS_SPLIT_CSV", str(DATASET_DIR / "source_disjoint_split_smoke.csv")))
_ckpt_env = os.environ.get("ESD_JASSNET_SSL_CHECKPOINT") or os.environ.get("ESD_JASSNET_EVAL_CKPT")
CKPT: Optional[Path] = Path(_ckpt_env).expanduser() if _ckpt_env else None
OUT_DIR = Path(os.environ.get(
    "ERROR_ANALYSIS_OUT_DIR",
    str(PROJECT_ROOT / "outputs" / "results" / "ERROR_ANALYSIS" / f"white_noise_robustness_final_ssl_fold{PHYSICAL_FOLD}"),
))
# Each physical fold directory has one internal train/validation split.
FOLD = int(os.environ.get("ESD_JASSNET_SPLIT_FOLD", "1"))

SR = 4000
SEG_SAMPLES = 8000

# White-noise SNR relative to the clean-mixture RMS.
# None denotes the clean baseline.
NOISE_SNRS_DB: Sequence[Optional[float]] = [
    None,
    30.0,
    20.0,
    10.0,
    5.0,
    0.0,
]

# Optional cap for a reduced validation run.
# None evaluates the complete selected split.
MAX_FILES: Optional[int] = None

SEED = 12345

# Polarity calibration follows the final inference procedure.
# Gain calibration is disabled so outputs are not forced toward the noisy mixture.
APPLY_POLARITY_CALIBRATION = True
APPLY_GAIN_CALIBRATION = False

FORCE_CPU = False


# =============================================================================
# DYNAMIC IMPORT OF EXISTING ERROR ANALYSIS CODE
# =============================================================================

def import_error_analysis_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Error analysis script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("ea_final", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["ea_final"] = module
    spec.loader.exec_module(module)
    return module


ea = import_error_analysis_module(ERROR_ANALYSIS_SCRIPT)


# =============================================================================
# NOISE HELPERS
# =============================================================================

def deterministic_seed(sample_name: str, noise_snr_db: Optional[float], base_seed: int) -> int:
    key = f"{base_seed}|{sample_name}|{noise_snr_db}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16)


def add_white_noise_at_snr(
    clean_mixture: np.ndarray,
    noise_snr_db: float,
    sample_name: str,
    base_seed: int = SEED,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Adds white noise N so that:

        noise_snr_db = 20 * log10(RMS(M_clean) / RMS(N))

    Returns:
        M_noisy, N, measured_noise_snr_db
    """
    rng = np.random.default_rng(deterministic_seed(sample_name, noise_snr_db, base_seed))

    M = ea.dc(clean_mixture).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=M.shape).astype(np.float32)
    noise = ea.dc(noise).astype(np.float32)

    rms_m = ea.safe_rms(M, eps=eps)
    rms_n_target = rms_m / (10.0 ** (noise_snr_db / 20.0))

    noise = noise * (rms_n_target / (ea.safe_rms(noise, eps=eps) + eps))
    noise = noise.astype(np.float32)

    M_noisy = (clean_mixture + noise).astype(np.float32)

    measured = 20.0 * np.log10((ea.safe_rms(clean_mixture, eps=eps) + eps) / (ea.safe_rms(noise, eps=eps) + eps))
    return M_noisy, noise, float(measured)


def energy_db(x: np.ndarray, eps: float = 1e-8) -> float:
    x = ea.dc(x)
    return float(10.0 * np.log10(float(np.dot(x, x)) + eps))


def energy_ratio_db(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> float:
    num = ea.dc(num)
    den = ea.dc(den)
    return float(10.0 * np.log10((float(np.dot(num, num)) + eps) / (float(np.dot(den, den)) + eps)))


def projection_ratio_db(signal: np.ndarray, basis: np.ndarray, eps: float = 1e-8) -> float:
    """
    Energy fraction of 'signal' that is linearly explained by 'basis'.

    Example:
      projection_ratio_db(H_hat, N)

    More negative means less noise-like component in H_hat.
    """
    x = ea.dc(signal)
    b = ea.dc(basis)

    b_energy = float(np.dot(b, b))
    x_energy = float(np.dot(x, x))

    if b_energy < eps or x_energy < eps:
        return float("nan")

    alpha = float(np.dot(x, b) / (b_energy + eps))
    proj = alpha * b
    return float(10.0 * np.log10((float(np.dot(proj, proj)) + eps) / (x_energy + eps)))


def infer_one(
    model: torch.nn.Module,
    M_input: np.ndarray,
    mixture_for_calibration: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    M_t = torch.from_numpy(M_input.astype(np.float32)).view(1, 1, -1).to(device)

    with torch.no_grad():
        H_hat_t, L_hat_t = model(M_t)

    H_hat = H_hat_t.squeeze().detach().cpu().numpy().astype(np.float32)
    L_hat = L_hat_t.squeeze().detach().cpu().numpy().astype(np.float32)

    calib: Dict[str, float] = {}

    if APPLY_POLARITY_CALIBRATION:
        H_hat, L_hat, c = ea.apply_global_polarity_calibration_np(mixture_for_calibration, H_hat, L_hat)
        calib.update(c)

    if APPLY_GAIN_CALIBRATION:
        H_hat, L_hat, c = ea.apply_mixture_gain_calibration_np(mixture_for_calibration, H_hat, L_hat)
        calib.update(c)

    return H_hat, L_hat, calib


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def summarize(df: pd.DataFrame, group_cols: List[str], out_path: Path) -> pd.DataFrame:
    metrics = [
        "si_sdr_h",
        "si_sdr_l",
        "mean_si_sdr",
        "delta_si_sdr_h_vs_clean",
        "delta_si_sdr_l_vs_clean",
        "delta_mean_si_sdr_vs_clean",
        "noisy_mix_corr",
        "noisy_mix_nmse_db",
        "clean_mix_corr",
        "clean_mix_nmse_db",
        "residual_noise_corr",
        "residual_noise_nmse_db",
        "residual_to_noise_energy_db",
        "output_delta_to_noise_energy_db",
        "output_delta_noise_corr",
        "noise_projection_ratio_db_h",
        "noise_projection_ratio_db_l",
    ]

    metrics = [m for m in metrics if m in df.columns]

    out = (
        df.groupby(group_cols)[metrics]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    out.to_csv(out_path, index=False)
    return out


def save_line_plots(summary_by_noise_flat: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary_by_noise_flat.copy()
    plot_df = plot_df[plot_df["noise_snr_db"].notna()].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("noise_snr_db", ascending=False)

    # Mean SI-SDR vs noise SNR
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["noise_snr_db"], plot_df["si_sdr_h_mean"], marker="o", label="HS SI-SDR")
    plt.plot(plot_df["noise_snr_db"], plot_df["si_sdr_l_mean"], marker="o", label="LS SI-SDR")
    plt.plot(plot_df["noise_snr_db"], plot_df["mean_si_sdr_mean"], marker="o", label="Mean SI-SDR")
    plt.gca().invert_xaxis()
    plt.xlabel("Input noise SNR [dB]  (lower = more noise)")
    plt.ylabel("SI-SDR [dB]")
    plt.title("White-noise robustness: SI-SDR vs input noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_si_sdr_vs_noise_snr.png", dpi=180)
    plt.close()

    # Residual vs noise
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["noise_snr_db"], plot_df["residual_noise_corr_mean"], marker="o", label="corr(R, N)")
    plt.gca().invert_xaxis()
    plt.xlabel("Input noise SNR [dB]  (lower = more noise)")
    plt.ylabel("Correlation")
    plt.title("Does the injected noise remain in the residual?")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_residual_noise_corr_vs_noise_snr.png", dpi=180)
    plt.close()

    # Output delta energy vs noise energy
    plt.figure(figsize=(8, 5))
    plt.plot(
        plot_df["noise_snr_db"],
        plot_df["output_delta_to_noise_energy_db_mean"],
        marker="o",
        label="energy(Hhat/Lhat change) / energy(noise)",
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Input noise SNR [dB]  (lower = more noise)")
    plt.ylabel("Energy ratio [dB]")
    plt.title("How much of the injected noise affects the outputs?")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_output_delta_energy_vs_noise_snr.png", dpi=180)
    plt.close()


def flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens pandas MultiIndex columns after groupby agg.
    """
    if not isinstance(summary.columns, pd.MultiIndex):
        return summary

    flat_cols = []
    for col in summary.columns:
        if col[1] == "":
            flat_cols.append(col[0])
        else:
            flat_cols.append(f"{col[0]}_{col[1]}")
    summary.columns = flat_cols
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ea.add_code_paths(None)
    cfg = ea.load_model_config()

    device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")

    print("=" * 100)
    print("WHITE-NOISE ROBUSTNESS AUDIT")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Split CSV: {SPLIT_CSV}")
    print(f"Fold: {FOLD}")
    print(f"Checkpoint: {CKPT}")
    print(f"Output: {OUT_DIR}")
    print(f"Noise SNRs: {NOISE_SNRS_DB}")
    print(f"Polarity calibration: {APPLY_POLARITY_CALIBRATION}")
    print(f"Gain calibration: {APPLY_GAIN_CALIBRATION}")
    print("=" * 100)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")
    if CKPT is None:
        raise RuntimeError(
            "Set ESD_JASSNET_SSL_CHECKPOINT to the SSL checkpoint to evaluate."
        )
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    files, snr_lookup, val_bases = ea.build_eval_file_list(
        dataset_dir=DATASET_DIR,
        split_csv=SPLIT_CSV,
        fold=FOLD,
        only_orig=True,
    )

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Evaluation files: {len(files)}")
    print(f"Validation base IDs: {len(val_bases) if val_bases else 'N/A'}")

    model = ea.load_checkpoint_model(CKPT, cfg, T=SEG_SAMPLES, device=device)
    model.eval()

    rows: List[Dict[str, Any]] = []

    for m_path, h_path, l_path in tqdm(files, desc="White-noise audit"):
        sample_name = m_path.stem[2:]
        sample_id = f"M_{sample_name}"

        M_clean = ea.load_wav(m_path, SR, SEG_SAMPLES)
        H_ref = ea.load_wav(h_path, SR, SEG_SAMPLES)
        L_ref = ea.load_wav(l_path, SR, SEG_SAMPLES)

        # Clean baseline prediction for this exact sample.
        H_hat_clean, L_hat_clean, calib_clean = infer_one(
            model=model,
            M_input=M_clean,
            mixture_for_calibration=M_clean,
            device=device,
        )

        clean_metrics = ea.compute_metrics(
            M_clean,
            H_ref,
            L_ref,
            H_hat_clean,
            L_hat_clean,
            sr=SR,
        )

        clean_pred_mix = ea.dc(H_hat_clean + L_hat_clean)

        for noise_snr_db in NOISE_SNRS_DB:
            if noise_snr_db is None:
                M_in = M_clean.copy()
                N = np.zeros_like(M_clean, dtype=np.float32)
                measured_noise_snr_db = float("inf")
                noise_condition = "clean"
            else:
                M_in, N, measured_noise_snr_db = add_white_noise_at_snr(
                    clean_mixture=M_clean,
                    noise_snr_db=float(noise_snr_db),
                    sample_name=sample_name,
                    base_seed=SEED,
                )
                noise_condition = f"{noise_snr_db:.0f}dB"

            if noise_snr_db is None:
                H_hat = H_hat_clean.copy()
                L_hat = L_hat_clean.copy()
                calib = dict(calib_clean)
            else:
                H_hat, L_hat, calib = infer_one(
                    model=model,
                    M_input=M_in,
                    mixture_for_calibration=M_in,
                    device=device,
                )

            metrics = ea.compute_metrics(
                M_in,
                H_ref,
                L_ref,
                H_hat,
                L_hat,
                sr=SR,
            )

            # Mixture consistency against noisy input and against clean M.
            noisy_mix_metrics = ea.mixture_consistency_metrics_np(M_in, H_hat, L_hat)
            clean_mix_metrics = ea.mixture_consistency_metrics_np(M_clean, H_hat, L_hat)

            residual = ea.dc(M_in - (H_hat + L_hat))
            pred_mix = ea.dc(H_hat + L_hat)
            output_delta = ea.dc(pred_mix - clean_pred_mix)

            if noise_snr_db is None:
                residual_noise_corr = float("nan")
                residual_noise_nmse_db = float("nan")
                residual_to_noise_energy_db = float("nan")
                output_delta_noise_corr = float("nan")
                output_delta_to_noise_energy_db = float("nan")
                noise_projection_h = float("nan")
                noise_projection_l = float("nan")
            else:
                residual_noise_corr = ea.safe_corr(residual, N)
                residual_noise_nmse_db = ea.nmse_raw_db_np(residual, N)
                residual_to_noise_energy_db = energy_ratio_db(residual, N)
                output_delta_noise_corr = ea.safe_corr(output_delta, N)
                output_delta_to_noise_energy_db = energy_ratio_db(output_delta, N)
                noise_projection_h = projection_ratio_db(H_hat, N)
                noise_projection_l = projection_ratio_db(L_hat, N)

            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "sample_name": sample_name,
                "base_id": ea.sample_name_to_base_id(sample_name),
                "fold": FOLD,
                "snr_label": ea.get_snr_label(sample_id, snr_lookup),
                "noise_condition": noise_condition,
                "noise_snr_db": noise_snr_db if noise_snr_db is not None else np.nan,
                "measured_noise_snr_db": measured_noise_snr_db,
                "m_path": str(m_path),
                "h_path": str(h_path),
                "l_path": str(l_path),

                # Standard metrics under noisy input.
                "si_sdr_h": metrics["si_sdr_h"],
                "si_sdr_l": metrics["si_sdr_l"],
                "mean_si_sdr": metrics["mean_si_sdr"],
                "sir_h": metrics["sir_h"],
                "sir_l": metrics["sir_l"],
                "sar_h": metrics["sar_h"],
                "sar_l": metrics["sar_l"],
                "target_local_snr_db": metrics["target_local_snr_db"],
                "local_snr_bin": metrics["local_snr_bin"],

                # Degradation vs clean input prediction for same sample.
                "clean_si_sdr_h": clean_metrics["si_sdr_h"],
                "clean_si_sdr_l": clean_metrics["si_sdr_l"],
                "clean_mean_si_sdr": clean_metrics["mean_si_sdr"],
                "delta_si_sdr_h_vs_clean": metrics["si_sdr_h"] - clean_metrics["si_sdr_h"],
                "delta_si_sdr_l_vs_clean": metrics["si_sdr_l"] - clean_metrics["si_sdr_l"],
                "delta_mean_si_sdr_vs_clean": metrics["mean_si_sdr"] - clean_metrics["mean_si_sdr"],

                # Mixture consistency against noisy M.
                "noisy_mix_corr": noisy_mix_metrics["mix_corr"],
                "noisy_mix_nmse_db": noisy_mix_metrics["mix_nmse_db"],
                "noisy_mix_gain_error_db": noisy_mix_metrics["mix_gain_error_db"],

                # Mixture consistency against clean M = H + L.
                "clean_mix_corr": clean_mix_metrics["mix_corr"],
                "clean_mix_nmse_db": clean_mix_metrics["mix_nmse_db"],
                "clean_mix_gain_error_db": clean_mix_metrics["mix_gain_error_db"],

                # Noise-specific diagnostics.
                "residual_noise_corr": residual_noise_corr,
                "residual_noise_nmse_db": residual_noise_nmse_db,
                "residual_to_noise_energy_db": residual_to_noise_energy_db,
                "output_delta_noise_corr": output_delta_noise_corr,
                "output_delta_to_noise_energy_db": output_delta_to_noise_energy_db,
                "noise_projection_ratio_db_h": noise_projection_h,
                "noise_projection_ratio_db_l": noise_projection_l,
            }

            row.update({f"calib_{k}": v for k, v in calib.items()})
            rows.append(row)

    df = pd.DataFrame(rows)

    per_sample_path = OUT_DIR / "per_sample_white_noise_metrics.csv"
    df.to_csv(per_sample_path, index=False)

    # Summaries.
    summary_noise = summarize(df, ["noise_condition", "noise_snr_db"], OUT_DIR / "summary_by_noise_snr_raw.csv")
    summary_noise_flat = flatten_summary(summary_noise.copy())
    summary_noise_flat.to_csv(OUT_DIR / "summary_by_noise_snr.csv", index=False)

    summary_noise_snr = summarize(df, ["noise_condition", "noise_snr_db", "snr_label"], OUT_DIR / "summary_by_noise_snr_and_global_snr_raw.csv")
    flatten_summary(summary_noise_snr.copy()).to_csv(OUT_DIR / "summary_by_noise_snr_and_global_snr.csv", index=False)

    summary_noise_local = summarize(df, ["noise_condition", "noise_snr_db", "local_snr_bin"], OUT_DIR / "summary_by_noise_snr_and_local_snr_bin_raw.csv")
    flatten_summary(summary_noise_local.copy()).to_csv(OUT_DIR / "summary_by_noise_snr_and_local_snr_bin.csv", index=False)

    save_line_plots(summary_noise_flat, OUT_DIR)

    # Compact README.
    with open(OUT_DIR / "README_white_noise_audit.txt", "w") as f:
        f.write("WHITE-NOISE ROBUSTNESS AUDIT\n")
        f.write("============================\n\n")
        f.write(f"Checkpoint: {CKPT}\n")
        f.write(f"Dataset: {DATASET_DIR}\n")
        f.write(f"Split CSV: {SPLIT_CSV}\n")
        f.write(f"Fold: {FOLD}\n")
        f.write(f"Evaluated rows: {len(df)}\n")
        f.write(f"Unique clean samples: {df['sample_id'].nunique()}\n")
        f.write(f"Noise SNRs: {NOISE_SNRS_DB}\n")
        f.write(f"Polarity calibration: {APPLY_POLARITY_CALIBRATION}\n")
        f.write(f"Gain calibration: {APPLY_GAIN_CALIBRATION}\n\n")

        f.write("Interpretation guide:\n")
        f.write("- delta_mean_si_sdr_vs_clean < 0 means the noisy input degraded separation.\n")
        f.write("- residual_noise_corr close to +1 means the unexplained residual resembles the injected noise.\n")
        f.write("- residual_noise_nmse_db close to very negative means residual approximates injected noise well.\n")
        f.write("- output_delta_to_noise_energy_db very negative means the outputs changed much less than the injected noise energy.\n")
        f.write("- noise_projection_ratio_db_h/l very negative means little white-noise-like component is present in each branch.\n")
        f.write("- noisy_mix_corr may decrease if the model suppresses noise instead of reconstructing it.\n")
        f.write("- clean_mix_corr staying high means H_hat + L_hat remains close to the clean H + L mixture.\n\n")

        f.write("Summary by noise SNR:\n")
        f.write(summary_noise_flat.to_string(index=False))

    print("=" * 100)
    print("DONE")
    print(f"Per-sample metrics: {per_sample_path}")
    print(f"Summary: {OUT_DIR / 'summary_by_noise_snr.csv'}")
    print(f"Plots: {OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()

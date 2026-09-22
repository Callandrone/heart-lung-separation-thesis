import os
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import soundfile as sf
from scipy.signal import stft as scipy_stft
from tqdm import tqdm
from sklearn.model_selection import KFold

import model_config as cfg

from snr_filter import (
    is_excluded_window,
    load_excluded_window_keys,
)



BASELINES = { }


####################
#Metrics
####################

def si_sdr_np(
    estimate: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    est = estimate - estimate.mean()
    tgt = target - target.mean()

    alpha = np.dot(est, tgt) / (np.dot(tgt, tgt) + eps)
    proj = alpha * tgt
    noise = est - proj

    return float(
        10 * np.log10(
            (np.dot(proj, proj) + eps)
            / (np.dot(noise, noise) + eps)
        )
    )

def nmse_aligned_np(
    estimate: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, float]:

    est = estimate - estimate.mean()
    tgt = target - target.mean()

    est_energy = np.dot(est, est)
    tgt_energy = np.dot(tgt, tgt)

    if tgt_energy < eps:
        return float("nan"), float("nan")

    if est_energy < eps:
        return 1.0, 0.0

    alpha = np.dot(est, tgt) / (est_energy + eps)
    est_aligned = alpha * est

    error = tgt - est_aligned

    nmse = np.dot(error, error) / (tgt_energy + eps)
    nmse = max(float(nmse), eps)

    nmse_db = 10.0 * np.log10(nmse)

    return float(nmse), float(nmse_db)

def nmse_raw_np(
    estimate: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, float]:

    est = estimate - estimate.mean()
    tgt = target - target.mean()

    target_energy = np.dot(tgt, tgt)

    if target_energy < eps:
        return float("nan"), float("nan")

    error = tgt - est
    nmse = np.dot(error, error) / (target_energy + eps)
    nmse = max(float(nmse), eps)

    nmse_db = 10.0 * np.log10(nmse)

    return float(nmse), float(nmse_db)


def rms_gain_error_db_np(
    estimate: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:

    est = estimate - estimate.mean()
    tgt = target - target.mean()

    rms_est = np.sqrt(np.mean(est ** 2) + eps)
    rms_tgt = np.sqrt(np.mean(tgt ** 2) + eps)

    return float(20.0 * np.log10(rms_est / rms_tgt))

def signed_projection_alpha_np(
    estimate: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Signed scale of the target component in the estimate.
    Correct scale and polarity correspond to alpha ~= +1.
    Negative alpha indicates a polarity-inverted target component.
    """
    est = estimate - estimate.mean()
    tgt = target - target.mean()

    return float(
        np.dot(est, tgt) / (np.dot(tgt, tgt) + eps)
    )


def rms_rescale_to_reference(
    estimate: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Rescale estimate so its RMS matches the reference RMS.
    Used only for visualization — this uses the ground truth,
    so it is NOT applicable in deployment. Only call when
    GAIN_RESCALE_MODE = 'gt'.
    """
    est = estimate - estimate.mean()
    ref = reference - reference.mean()

    rms_est = np.sqrt(np.mean(est ** 2) + eps)
    rms_ref = np.sqrt(np.mean(ref ** 2) + eps)

    return est * (rms_ref / rms_est)

def local_target_snr_db_np(
    H_ref: np.ndarray,
    L_ref: np.ndarray,
    eps: float = 1e-8,
) -> float:

    h = H_ref - H_ref.mean()
    l = L_ref - L_ref.mean()

    rms_h = np.sqrt(np.mean(h ** 2) + eps)
    rms_l = np.sqrt(np.mean(l ** 2) + eps)

    return float(20.0 * np.log10(rms_h / rms_l))
def mixture_consistency_metrics_np(
    mixture: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    eps: float = 1e-8,
) -> dict:
    """
    Evaluate whether the two predicted sources reconstruct the input mixture.

    Metrics are computed after DC removal, consistently with the source-level
    waveform metrics already reported by this evaluation script.
    """
    mix = mixture - mixture.mean()
    reconstructed_mix = H_hat + L_hat
    reconstructed_mix = reconstructed_mix - reconstructed_mix.mean()

    residual = mix - reconstructed_mix
    mix_energy = np.dot(mix, mix)
    reconstructed_energy = np.dot(reconstructed_mix, reconstructed_mix)

    if mix_energy < eps:
        return {
            "mix_nmse": float("nan"),
            "mix_nmse_db": float("nan"),
            "mix_gain_error_db": float("nan"),
            "mix_gamma": float("nan"),
            "negative_mix_gamma": float("nan"),
            "mix_corr": float("nan"),
        }

    nmse = np.dot(residual, residual) / (mix_energy + eps)
    nmse = max(float(nmse), eps)
    nmse_db = 10.0 * np.log10(nmse)

    rms_mix = np.sqrt(np.mean(mix ** 2) + eps)
    rms_reconstructed = np.sqrt(np.mean(reconstructed_mix ** 2) + eps)
    gain_error_db = 20.0 * np.log10(rms_reconstructed / rms_mix)

    gamma = float(
        np.dot(mix, reconstructed_mix)
        / (reconstructed_energy + eps)
    )

    denominator = np.sqrt(mix_energy * reconstructed_energy) + eps
    corr = float(np.dot(mix, reconstructed_mix) / denominator)

    return {
        "mix_nmse": float(nmse),
        "mix_nmse_db": float(nmse_db),
        "mix_gain_error_db": float(gain_error_db),
        "mix_gamma": gamma,
        "negative_mix_gamma": float(gamma < 0.0),
        "mix_corr": corr,
    }

def apply_global_polarity_calibration_np(
    mixture: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Correct a global polarity inversion using only the input mixture.

    The correction is valid when the expected forward relation has positive
    polarity, e.g. synthetic additive mixtures M = H + L.

    No ground-truth source is used.
    """
    pre_metrics = mixture_consistency_metrics_np(
        mixture=mixture,
        H_hat=H_hat,
        L_hat=L_hat,
        eps=eps,
    )

    gamma_before = float(pre_metrics["mix_gamma"])
    corr_before = float(pre_metrics["mix_corr"])

    should_flip = gamma_before < 0.0

    if should_flip:
        H_hat = -H_hat
        L_hat = -L_hat

    post_metrics = mixture_consistency_metrics_np(
        mixture=mixture,
        H_hat=H_hat,
        L_hat=L_hat,
        eps=eps,
    )

    calibration_info = {
        "polarity_calibration_applied": 1.0,
        "polarity_flipped": float(should_flip),
        "mix_gamma_before_calibration": gamma_before,
        "mix_corr_before_calibration": corr_before,
        "mix_gamma_after_calibration": float(post_metrics["mix_gamma"]),
        "mix_corr_after_calibration": float(post_metrics["mix_corr"]),
    }

    return H_hat, L_hat, calibration_info
def apply_mixture_gain_calibration_np(
    mixture: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Deterministic mixture-informed gain calibration.

    It uses only:
    - input mixture M
    - estimated H_hat
    - estimated L_hat

    It does NOT use H_ref or L_ref, so it is deployment-safe.

    gamma = <M, H_hat + L_hat> / ||H_hat + L_hat||²
    """
    pre_metrics = mixture_consistency_metrics_np(
        mixture=mixture,
        H_hat=H_hat,
        L_hat=L_hat,
        eps=eps,
    )

    mix = mixture - mixture.mean()
    pred_mix = H_hat + L_hat
    pred_mix = pred_mix - pred_mix.mean()

    pred_energy = np.dot(pred_mix, pred_mix)

    if pred_energy < eps:
        gamma = 1.0
    else:
        gamma = float(np.dot(mix, pred_mix) / (pred_energy + eps))

    H_hat = gamma * H_hat
    L_hat = gamma * L_hat

    post_metrics = mixture_consistency_metrics_np(
        mixture=mixture,
        H_hat=H_hat,
        L_hat=L_hat,
        eps=eps,
    )

    calibration_info = {
        "gain_calibration_applied": 1.0,
        "gain_calibration_gamma": gamma,
        "mix_nmse_db_before_gain_calibration": float(pre_metrics["mix_nmse_db"]),
        "mix_corr_before_gain_calibration": float(pre_metrics["mix_corr"]),
        "mix_gain_error_db_before_gain_calibration": float(pre_metrics["mix_gain_error_db"]),
        "mix_nmse_db_after_gain_calibration": float(post_metrics["mix_nmse_db"]),
        "mix_corr_after_gain_calibration": float(post_metrics["mix_corr"]),
        "mix_gain_error_db_after_gain_calibration": float(post_metrics["mix_gain_error_db"]),
    }

    return H_hat, L_hat, calibration_info


def bss_eval_sources(
    estimate: np.ndarray,
    target: np.ndarray,
    interference: np.ndarray,
    eps: float = 1e-8,
) -> dict:
    def proj(v, onto):
        return onto * (np.dot(v, onto) / (np.dot(onto, onto) + eps))

    e = estimate - estimate.mean()
    t = target - target.mean()
    i = interference - interference.mean()

    s_tgt = proj(e, t)
    e_interf = proj(e - s_tgt, i)
    e_artif = e - s_tgt - e_interf

    def safe_db(num, den):
        n = np.dot(num, num)
        d = np.dot(den, den) + eps
        return float(10 * np.log10((n + eps) / d))

    return {
        "SDR": safe_db(s_tgt, e_interf + e_artif),
        "SIR": safe_db(s_tgt, e_interf),
        "SAR": safe_db(s_tgt + e_interf, e_artif),
    }


def log_spectral_distance(
    estimate: np.ndarray,
    target: np.ndarray,
    sr: int = cfg.SR,
    n_fft: int = 256,
    hop: int = 64,
) -> float:
    def log_mag(x):
        _, _, S = scipy_stft(
            x,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop,
        )
        return np.log10(np.abs(S) + 1e-8)

    lm_est = log_mag(estimate)
    lm_ref = log_mag(target)

    diff = lm_est - lm_ref

    return float(
        np.mean(
            np.sqrt(
                np.mean(diff ** 2, axis=0)
            )
        )
    )


def log_spectrogram_correlation(
    estimate: np.ndarray,
    target: np.ndarray,
    sr: int = cfg.SR,
    n_fft: int = 256,
    hop: int = 64,
) -> float:
    def log_mag_flat(x):
        _, _, S = scipy_stft(
            x,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop,
        )
        return np.log10(np.abs(S) + 1e-8).flatten()

    lm_est = log_mag_flat(estimate)
    lm_ref = log_mag_flat(target)

    if lm_est.std() < 1e-8 or lm_ref.std() < 1e-8:
        return float("nan")

    return float(np.corrcoef(lm_est, lm_ref)[0, 1])


def stoi_proxy(
    estimate: np.ndarray,
    target: np.ndarray,
    sr: int = cfg.SR,
    n_fft: int = 256,
    hop: int = 64,
    n_bands: int = 15,
) -> float:
    def envelope(x):
        _, _, S = scipy_stft(
            x,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop,
        )
        return np.abs(S)

    env_est = envelope(estimate)
    env_ref = envelope(target)

    F = env_est.shape[0]
    band_size = max(1, F // n_bands)
    correlations = []

    for b in range(n_bands):
        lo = b * band_size
        hi = min(F, (b + 1) * band_size)

        e_band = env_est[lo:hi].mean(axis=0)
        r_band = env_ref[lo:hi].mean(axis=0)

        if e_band.std() < 1e-8 or r_band.std() < 1e-8:
            continue

        c = np.corrcoef(e_band, r_band)[0, 1]
        correlations.append(np.clip(c, -1, 1))

    return float(np.mean(correlations)) if correlations else 0.0


def evaluate_sample(
    M: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    H_ref: np.ndarray,
    L_ref: np.ndarray,
    sr: int = cfg.SR,
) -> dict:
    M = M - M.mean()
    H_hat = H_hat - H_hat.mean()
    L_hat = L_hat - L_hat.mean()
    H_ref = H_ref - H_ref.mean()
    L_ref = L_ref - L_ref.mean()

    si_h = si_sdr_np(H_hat, H_ref)
    si_l = si_sdr_np(L_hat, L_ref)

    nmse_aligned_h, nmse_aligned_db_h = nmse_aligned_np(
        H_hat,
        H_ref,
    )

    nmse_aligned_l, nmse_aligned_db_l = nmse_aligned_np(
        L_hat,
        L_ref,
    )
    nmse_raw_h, nmse_raw_db_h = nmse_raw_np(
        H_hat,
        H_ref,
    )

    nmse_raw_l, nmse_raw_db_l = nmse_raw_np(
        L_hat,
        L_ref,
    )

    gain_error_db_h = rms_gain_error_db_np(
        H_hat,
        H_ref,
    )

    gain_error_db_l = rms_gain_error_db_np(
        L_hat,
        L_ref,
    )

    signed_alpha_h = signed_projection_alpha_np(
        H_hat,
        H_ref,
    )

    signed_alpha_l = signed_projection_alpha_np(
        L_hat,
        L_ref,
    )

    target_local_snr_db = local_target_snr_db_np(
        H_ref,
        L_ref,
    )

    mix_metrics = mixture_consistency_metrics_np(
        mixture=M,
        H_hat=H_hat,
        L_hat=L_hat,
    )

    bss_h = bss_eval_sources(H_hat, H_ref, L_ref)
    bss_l = bss_eval_sources(L_hat, L_ref, H_ref)

    lsd_h = log_spectral_distance(H_hat, H_ref, sr=sr)
    lsd_l = log_spectral_distance(L_hat, L_ref, sr=sr)

    rho_h = log_spectrogram_correlation(H_hat, H_ref, sr=sr)
    rho_l = log_spectrogram_correlation(L_hat, L_ref, sr=sr)

    stoi_h = stoi_proxy(H_hat, H_ref, sr=sr)
    stoi_l = stoi_proxy(L_hat, L_ref, sr=sr)

    return {
        "si_sdr_h": si_h,
        "si_sdr_l": si_l,

        "nmse_aligned_h": nmse_aligned_h,
        "nmse_aligned_l": nmse_aligned_l,
        "nmse_aligned_db_h": nmse_aligned_db_h,
        "nmse_aligned_db_l": nmse_aligned_db_l,

        "nmse_raw_h": nmse_raw_h,
        "nmse_raw_l": nmse_raw_l,
        "nmse_raw_db_h": nmse_raw_db_h,
        "nmse_raw_db_l": nmse_raw_db_l,

        "gain_error_db_h": gain_error_db_h,
        "gain_error_db_l": gain_error_db_l,
        "signed_alpha_h": signed_alpha_h,
        "signed_alpha_l": signed_alpha_l,
        "negative_alpha_h": float(signed_alpha_h < 0.0),
        "negative_alpha_l": float(signed_alpha_l < 0.0),
        "target_local_snr_db": target_local_snr_db,

        "mix_nmse": mix_metrics["mix_nmse"],
        "mix_nmse_db": mix_metrics["mix_nmse_db"],
        "mix_gain_error_db": mix_metrics["mix_gain_error_db"],
        "mix_gamma": mix_metrics["mix_gamma"],
        "negative_mix_gamma": mix_metrics["negative_mix_gamma"],
        "mix_corr": mix_metrics["mix_corr"],

        "sdr_h": bss_h["SDR"],
        "sdr_l": bss_l["SDR"],
        "sir_h": bss_h["SIR"],
        "sir_l": bss_l["SIR"],
        "sar_h": bss_h["SAR"],
        "sar_l": bss_l["SAR"],
        "lsd_h": lsd_h,
        "lsd_l": lsd_l,
        "rho_h": rho_h,
        "rho_l": rho_l,
        "stoi_h": stoi_h,
        "stoi_l": stoi_l,
    }


####################
#Model loading
####################

def load_model(
    ckpt_path: str,
    T: int,
    device: torch.device,
):
    from encoder import CardiopulmonaryEncoder
    from separator import CardiopulmonarySeparator
    from decoder import CardiopulmonaryDecoder
    import torch.nn as nn

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

        def forward(
            self,
            M: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            Z = self.encoder(M)
            Z_hs, Z_ls, _ = self.separator(Z)

            H_hat = self.decoder(Z_hs, normalise=False)
            L_hat = self.decoder(Z_ls, normalise=False)

            return H_hat, L_hat

    model = CardiopulmonaryNet(T=T).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])


    model.eval()

    return model


def read_target_gain_info_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> dict:
    """
    Read target scaling saved by train.py.

    If the checkpoint was not produced by the V2 fine-tuning pipeline, the
    evaluation falls back to unscaled references.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_saved = ckpt.get("config", {})
    info = cfg_saved.get("TARGET_GAIN_INFO", None)

    if not isinstance(info, dict):
        return {
            "target_gain_mode": "none",
            "alpha_h": 1.0,
            "alpha_l": 1.0,
            "fit_nmse_db": float("nan"),
            "fit_corr": float("nan"),
        }

    out = {
        "target_gain_mode": info.get("target_gain_mode", "none"),
        "alpha_h": float(info.get("alpha_h", 1.0)),
        "alpha_l": float(info.get("alpha_l", 1.0)),
        "fit_nmse_db": float(info.get("fit_nmse_db", float("nan"))),
        "fit_corr": float(info.get("fit_corr", float("nan"))),
    }

    if isinstance(info.get("per_base_gains", None), dict):
        out["per_base_gains"] = info["per_base_gains"]
        out["n_gain_bases"] = int(info.get("n_gain_bases", len(info["per_base_gains"])))
        out["alpha_h_min"] = float(info.get("alpha_h_min", float("nan")))
        out["alpha_h_max"] = float(info.get("alpha_h_max", float("nan")))
        out["alpha_l_min"] = float(info.get("alpha_l_min", float("nan")))
        out["alpha_l_max"] = float(info.get("alpha_l_max", float("nan")))
        out["fit_corr_min"] = float(info.get("fit_corr_min", float("nan")))

    return out


def _base_id_from_sample_id(sample_id: str) -> str:
    sid = Path(str(sample_id)).stem
    if sid.startswith("M_"):
        sid = sid[2:]
    elif sid.startswith("H_") or sid.startswith("L_"):
        sid = sid[2:]
    return sid.split("_s")[0]


def apply_target_gain_to_references(
    H_ref: np.ndarray,
    L_ref: np.ndarray,
    target_gain_info: dict | None,
    sample_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if target_gain_info is None:
        target_gain_info = {
            "target_gain_mode": "none",
            "alpha_h": 1.0,
            "alpha_l": 1.0,
            "fit_nmse_db": float("nan"),
            "fit_corr": float("nan"),
        }

    gain_info = dict(target_gain_info)
    mode = str(gain_info.get("target_gain_mode", "none")).lower().strip()

    if mode == "per_triplet" and sample_id is not None:
        base = _base_id_from_sample_id(sample_id)
        per_base = gain_info.get("per_base_gains", {})
        base_gains = per_base.get(base, None)

        if isinstance(base_gains, dict):
            gain_info["alpha_h"] = float(base_gains.get("alpha_h", 1.0))
            gain_info["alpha_l"] = float(base_gains.get("alpha_l", 1.0))
            gain_info["fit_nmse_db"] = float(base_gains.get("fit_nmse_db", float("nan")))
            gain_info["fit_corr"] = float(base_gains.get("fit_corr", float("nan")))
            gain_info["target_gain_base_id"] = base

    alpha_h = float(gain_info.get("alpha_h", 1.0))
    alpha_l = float(gain_info.get("alpha_l", 1.0))

    return H_ref * alpha_h, L_ref * alpha_l, gain_info


####################
#Audio saving helpers
####################

def peak_norm_np(
    x: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-8
    return x * (target_peak / peak)


def normalise_sample_id(value: str) -> str:
    sid = Path(str(value)).stem.strip()

    if sid == "":
        return ""

    if not sid.startswith("M_"):
        sid = "M_" + sid

    return sid


def load_selected_sample_ids(csv_path: str | None) -> set[str] | None:
    if csv_path is None or str(csv_path).strip() == "":
        return None

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Selected-cases CSV not found: {path}"
        )

    selected = set()

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if "sample_id" in row and row["sample_id"]:
                sid = normalise_sample_id(row["sample_id"])
                if sid:
                    selected.add(sid)

            for col in ["m_path", "M_path", "mixture_path", "path"]:
                if col in row and row[col]:
                    sid = normalise_sample_id(row[col])
                    if sid:
                        selected.add(sid)

    print(
        f"Selected audio cases loaded: {len(selected)} "
        f"from {path}"
    )

    return selected


def prepare_audio_for_saving(
    x: np.ndarray,
    peak_normalise: bool = True,
) -> np.ndarray:
    if peak_normalise:
        return peak_norm_np(x)

    return x.astype(np.float32)


def save_audio_bundle(
    audio_out_dir: str,
    sample_id: str,
    M_np: np.ndarray,
    H_np: np.ndarray,
    L_np: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    sr: int,
    save_references: bool = True,
    peak_normalise: bool = True,
) -> None:
    os.makedirs(audio_out_dir, exist_ok=True)

    base = sample_id.replace("M_", "", 1)

    sf.write(
        os.path.join(audio_out_dir, f"H_hat_{base}.wav"),
        prepare_audio_for_saving(H_hat, peak_normalise),
        sr,
    )

    sf.write(
        os.path.join(audio_out_dir, f"L_hat_{base}.wav"),
        prepare_audio_for_saving(L_hat, peak_normalise),
        sr,
    )

    if save_references:
        sf.write(
            os.path.join(audio_out_dir, f"M_mix_{base}.wav"),
            prepare_audio_for_saving(M_np, peak_normalise),
            sr,
        )

        sf.write(
            os.path.join(audio_out_dir, f"H_ref_{base}.wav"),
            prepare_audio_for_saving(H_np, peak_normalise),
            sr,
        )

        sf.write(
            os.path.join(audio_out_dir, f"L_ref_{base}.wav"),
            prepare_audio_for_saving(L_np, peak_normalise),
            sr,
        )

def load_wav_for_plot(
    path: Path,
    sr: int,
    seg_samples: int,
) -> np.ndarray:
    audio, file_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    if file_sr != sr:
        raise ValueError(
            f"Unexpected sample rate for {path}: "
            f"expected {sr}, got {file_sr}"
        )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) > seg_samples:
        audio = audio[:seg_samples]
    elif len(audio) < seg_samples:
        audio = np.pad(audio, (0, seg_samples - len(audio)))

    return audio.astype(np.float32)

def plot_prediction_case(
    sample_id: str,
    case_group: str,
    rank: int,
    M_np: np.ndarray,
    H_ref: np.ndarray,
    L_ref: np.ndarray,
    H_hat: np.ndarray,
    L_hat: np.ndarray,
    metrics: dict,
    sr: int,
    out_dir: Path,
    gain_rescale_mode: str = 'none',
) -> None:
    time = np.arange(len(M_np)) / sr

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    axes[0].plot(time, M_np, linewidth=0.8)
    axes[0].set_title("Input mixture")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        time,
        H_ref,
        linewidth=0.9,
        label="Ground truth",
    )
    axes[1].plot(
        time,
        H_hat,
        linewidth=0.8,
        alpha=0.80,
        label="Prediction",
    )
    axes[1].set_title(
        "Heart sound | "
        f"SI-SDR = {metrics['si_sdr_h']:+.2f} dB | "
        f"NMSE aligned = {metrics['nmse_aligned_db_h']:+.2f} dB"
    )
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(
        time,
        L_ref,
        linewidth=0.9,
        label="Ground truth",
    )
    axes[2].plot(
        time,
        L_hat,
        linewidth=0.8,
        alpha=0.80,
        label="Prediction",
    )
    axes[2].set_title(
        "Lung sound | "
        f"SI-SDR = {metrics['si_sdr_l']:+.2f} dB | "
        f"NMSE aligned = {metrics['nmse_aligned_db_l']:+.2f} dB"
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(
        f"{case_group.capitalize()} case #{rank} — {sample_id}",
        fontsize=14,
    )

    fig.tight_layout()

    target_dir = out_dir / "prediction_plots" / gain_rescale_mode / case_group
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{rank:02d}_{sample_id}.png"

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Plot saved: {output_path}")

def select_best_worst_cases(
            results_csv: Path,
            top_k: int = 5,
    ) -> pd.DataFrame:
        if not results_csv.exists():
            raise FileNotFoundError(
                f"Results CSV not found: {results_csv}"
            )

        df = pd.read_csv(results_csv)

        required_columns = {
            "sample_id",
            "si_sdr_h",
            "si_sdr_l",
            "nmse_aligned_db_h",
            "nmse_aligned_db_l",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                "Missing columns in results CSV: "
                f"{sorted(missing_columns)}"
            )

        df["mean_si_sdr"] = df[["si_sdr_h", "si_sdr_l"]].mean(axis=1)
        df["min_si_sdr"] = df[["si_sdr_h", "si_sdr_l"]].min(axis=1)

        best = (
            df.sort_values("mean_si_sdr", ascending=False)
            .head(top_k)
            .copy()
        )
        best["case_group"] = "best"
        best["rank"] = range(1, len(best) + 1)
        best["selection_score"] = best["mean_si_sdr"]

        worst = (
            df.sort_values("min_si_sdr", ascending=True)
            .head(top_k)
            .copy()
        )
        worst["case_group"] = "worst"
        worst["rank"] = range(1, len(worst) + 1)
        worst["selection_score"] = worst["min_si_sdr"]

        selected = pd.concat(
            [best, worst],
            ignore_index=True,
        )

        return selected

def select_best_worst_cases(
    results_csv: Path,
    top_k: int = 5,
) -> pd.DataFrame:
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {results_csv}"
        )

    df = pd.read_csv(results_csv)

    required_columns = {
        "sample_id",
        "si_sdr_h",
        "si_sdr_l",
        "nmse_aligned_db_h",
        "nmse_aligned_db_l",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns in results CSV: "
            f"{sorted(missing_columns)}"
        )

    df["mean_si_sdr"] = df[["si_sdr_h", "si_sdr_l"]].mean(axis=1)
    df["min_si_sdr"] = df[["si_sdr_h", "si_sdr_l"]].min(axis=1)

    best = (
        df.sort_values("mean_si_sdr", ascending=False)
          .head(top_k)
          .copy()
    )
    best["case_group"] = "best"
    best["rank"] = range(1, len(best) + 1)
    best["selection_score"] = best["mean_si_sdr"]

    worst = (
        df.sort_values("min_si_sdr", ascending=True)
          .head(top_k)
          .copy()
    )
    worst["case_group"] = "worst"
    worst["rank"] = range(1, len(worst) + 1)
    worst["selection_score"] = worst["min_si_sdr"]

    selected = pd.concat(
        [best, worst],
        ignore_index=True,
    )

    return selected


def load_sample_fold_mapping(
    out_dir: Path,
    n_folds: int,
) -> dict[str, int]:
    mapping = {}

    for fold_no in range(1, n_folds + 1):
        fold_csv = out_dir / f"results_fold{fold_no}.csv"

        if not fold_csv.exists():
            raise FileNotFoundError(
                f"Fold results file not found: {fold_csv}"
            )

        fold_df = pd.read_csv(
            fold_csv,
            usecols=["sample_id"],
        )

        for sample_id in fold_df["sample_id"].astype(str):
            mapping[sample_id] = fold_no

    return mapping


@torch.no_grad()
def generate_best_worst_prediction_plots(
    args,
    device: torch.device,
) -> None:
    results_dir = Path(args.out_dir)
    results_csv = results_dir / "results_all.csv"
    saved_plot_audio_count = 0

    selected = select_best_worst_cases(
        results_csv=results_csv,
        top_k=args.plot_top_k,
    )

    sample_fold_mapping = load_sample_fold_mapping(
        out_dir=results_dir,
        n_folds=args.n_folds,
    )

    selected["fold"] = selected["sample_id"].map(sample_fold_mapping)

    if selected["fold"].isna().any():
        missing = selected.loc[
            selected["fold"].isna(),
            "sample_id",
        ].tolist()

        raise RuntimeError(
            "Could not determine fold for samples: "
            f"{missing}"
        )

    selected_cases_path = (
        results_dir
        / "prediction_plots"
        / "selected_best_worst_cases.csv"
    )

    selected_cases_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        selected_cases_path,
        index=False,
    )

    print("=" * 80)
    print("PREDICTION PLOTS — SELECTED CASES")
    print("=" * 80)
    print(
        selected[
            [
                "case_group",
                "rank",
                "sample_id",
                "fold",
                "si_sdr_h",
                "si_sdr_l",
                "nmse_aligned_db_h",
                "nmse_aligned_db_l",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Selected cases saved: {selected_cases_path}")

    dataset_root = Path(args.test_dir)

    for fold_no, fold_cases in selected.groupby("fold"):
        fold_no = int(fold_no)

        ckpt_path = resolve_fold_checkpoint(
            args.ckpt_dir,
            fold_no - 1,
        )

        if ckpt_path is None:
            raise FileNotFoundError(
                f"Checkpoint not found for fold {fold_no}"
            )

        print()
        print(f"Loading checkpoint for fold {fold_no}: {ckpt_path}")
        target_gain_info = read_target_gain_info_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
        )
        print(
            f"Plot target gain: mode={target_gain_info.get('target_gain_mode')} | "
            f"alpha_h={target_gain_info.get('alpha_h', 1.0):+.6f} | "
            f"alpha_l={target_gain_info.get('alpha_l', 1.0):+.6f}"
        )

        model = load_model(
            ckpt_path=ckpt_path,
            T=args.seg_samples,
            device=device,
        )

        for _, row in fold_cases.iterrows():
            sample_id = str(row["sample_id"])
            sample_name = sample_id.replace("M_", "", 1)

            m_path = dataset_root / f"M_{sample_name}.wav"
            h_path = dataset_root / f"H_{sample_name}.wav"
            l_path = dataset_root / f"L_{sample_name}.wav"

            M_np = load_wav_for_plot(
                m_path,
                sr=args.sr,
                seg_samples=args.seg_samples,
            )
            H_ref = load_wav_for_plot(
                h_path,
                sr=args.sr,
                seg_samples=args.seg_samples,
            )
            L_ref = load_wav_for_plot(
                l_path,
                sr=args.sr,
                seg_samples=args.seg_samples,
            )
            H_ref, L_ref, _ = apply_target_gain_to_references(
                H_ref=H_ref,
                L_ref=L_ref,
                target_gain_info=target_gain_info,
                sample_id=sample_id,
            )

            M_t = (
                torch.from_numpy(M_np)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            H_hat_t, L_hat_t = model(M_t)

            H_hat = H_hat_t.squeeze().detach().cpu().numpy()
            L_hat = L_hat_t.squeeze().detach().cpu().numpy()

            if args.apply_mixture_polarity_calibration:
                H_hat, L_hat, _ = apply_global_polarity_calibration_np(
                    mixture=M_np,
                    H_hat=H_hat,
                    L_hat=L_hat,
                )

            if getattr(args, "apply_mixture_gain_calibration", False):
                H_hat, L_hat, _ = apply_mixture_gain_calibration_np(
                    mixture=M_np,
                    H_hat=H_hat,
                    L_hat=L_hat,
                )

            if getattr(args, 'GAIN_RESCALE_MODE', 'none') == 'gt':
                H_hat = rms_rescale_to_reference(H_hat, H_ref)
                L_hat = rms_rescale_to_reference(L_hat, L_ref)

            if sample_id == "M_001528_s001_orig":
                def stats(name: str, x: np.ndarray) -> None:
                    print(
                        f"{name:<8} | "
                        f"min={x.min():+.6f} | "
                        f"max={x.max():+.6f} | "
                        f"peak={np.max(np.abs(x)):.6f} | "
                        f"rms={np.sqrt(np.mean(x ** 2)):.6f}"
                    )

                print("\n" + "=" * 80)
                print(f"RAW AMPLITUDE CHECK — {sample_id}")
                print("=" * 80)
                stats("M", M_np)
                stats("H_ref", H_ref)
                stats("L_ref", L_ref)
                stats("H_hat", H_hat)
                stats("L_hat", L_hat)
                print("=" * 80 + "\n")
            metrics = evaluate_sample(
                M=M_np,
                H_hat=H_hat,
                L_hat=L_hat,
                H_ref=H_ref,
                L_ref=L_ref,
                sr=args.sr,
            )

            plot_prediction_case(
                sample_id=sample_id,
                case_group=str(row["case_group"]),
                rank=int(row["rank"]),
                M_np=M_np,
                H_ref=H_ref,
                L_ref=L_ref,
                H_hat=H_hat,
                L_hat=L_hat,
                metrics=metrics,
                sr=args.sr,
                out_dir=results_dir,
                gain_rescale_mode=getattr(args, 'GAIN_RESCALE_MODE', 'none'),
            )
            if getattr(args, "export_best_worst_audio", False):
                plot_audio_dir = (
                        results_dir
                        / "prediction_plot_audio"
                        / getattr(args, "GAIN_RESCALE_MODE", "none")
                        / str(row["case_group"])
                        / f"{int(row['rank']):02d}_{sample_id}"
                )

                save_audio_bundle(
                    audio_out_dir=str(plot_audio_dir),
                    sample_id=sample_id,
                    M_np=M_np,
                    H_np=H_ref,
                    L_np=L_ref,
                    H_hat=H_hat,
                    L_hat=L_hat,
                    sr=args.sr,
                    save_references=True,
                    peak_normalise=getattr(args, "peak_normalise_saved_audio", True),
                )

                saved_plot_audio_count += 1

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("Prediction plots completed.")
    print(
        "Plots output directory: "
        f"{results_dir / 'prediction_plots'}"
    )
    print(
        "Plot audio output directory: "
        f"{results_dir / 'prediction_plot_audio'}"
    )
    print(f"Saved plot audio bundles: {saved_plot_audio_count}")
    print("=" * 80)

####################
#Inference
####################

@torch.no_grad()
def run_inference(
    model,
    test_files: list[tuple],
    sr: int,
    seg_samples: int,
    device: torch.device,
    save_audio: bool = False,
    audio_out_dir: str | None = None,
    selected_sample_ids: set[str] | None = None,
    save_audio_references: bool = True,
    peak_normalise_saved_audio: bool = True,
    apply_mixture_polarity_calibration: bool = False,
    apply_mixture_gain_calibration: bool = False,
    target_gain_info: dict | None = None,
    snr_lookup: dict[str, str] | None = None,
) -> list[dict]:

    results = []
    saved_audio_count = 0

    def load_wav(path: Path) -> np.ndarray:
        audio, file_sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )

        if file_sr != sr:
            raise ValueError(
                f"Unexpected sample rate for {path}: "
                f"expected {sr}, got {file_sr}"
            )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if len(audio) > seg_samples:
            audio = audio[:seg_samples]
        elif len(audio) < seg_samples:
            audio = np.pad(audio, (0, seg_samples - len(audio)))

        return audio.astype(np.float32)

    for m_path, h_path, l_path in tqdm(
        test_files,
        desc="Evaluating",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        sample_id = Path(m_path).stem

        M_np = load_wav(m_path)
        H_np_raw = load_wav(h_path)
        L_np_raw = load_wav(l_path)

        H_np, L_np, gain_info = apply_target_gain_to_references(
            H_ref=H_np_raw,
            L_ref=L_np_raw,
            target_gain_info=target_gain_info,
            sample_id=sample_id,
        )

        M_t = torch.from_numpy(M_np).unsqueeze(0).unsqueeze(0).to(device)

        H_hat_t, L_hat_t = model(M_t)

        H_hat = H_hat_t.squeeze().detach().cpu().numpy()
        L_hat = L_hat_t.squeeze().detach().cpu().numpy()

        calibration_info = {
            "polarity_calibration_applied": 0.0,
            "polarity_flipped": 0.0,
            "mix_gamma_before_calibration": float("nan"),
            "mix_corr_before_calibration": float("nan"),
            "mix_gamma_after_calibration": float("nan"),
            "mix_corr_after_calibration": float("nan"),

            "gain_calibration_applied": 0.0,
            "gain_calibration_gamma": float("nan"),
            "mix_nmse_db_before_gain_calibration": float("nan"),
            "mix_corr_before_gain_calibration": float("nan"),
            "mix_gain_error_db_before_gain_calibration": float("nan"),
            "mix_nmse_db_after_gain_calibration": float("nan"),
            "mix_corr_after_gain_calibration": float("nan"),
            "mix_gain_error_db_after_gain_calibration": float("nan"),
        }

        if apply_mixture_polarity_calibration:
            H_hat, L_hat, polarity_info = apply_global_polarity_calibration_np(
                mixture=M_np,
                H_hat=H_hat,
                L_hat=L_hat,
            )
            calibration_info.update(polarity_info)

        if apply_mixture_gain_calibration:
            H_hat, L_hat, gain_calib_info = apply_mixture_gain_calibration_np(
                mixture=M_np,
                H_hat=H_hat,
                L_hat=L_hat,
            )
            calibration_info.update(gain_calib_info)

        metrics = evaluate_sample(
            M=M_np,
            H_hat=H_hat,
            L_hat=L_hat,
            H_ref=H_np,
            L_ref=L_np,
            sr=sr,
        )

        metrics.update(calibration_info)

        metrics["target_gain_mode"] = gain_info.get("target_gain_mode", "none")
        metrics["target_gain_alpha_h"] = gain_info.get("alpha_h", 1.0)
        metrics["target_gain_alpha_l"] = gain_info.get("alpha_l", 1.0)
        metrics["target_gain_fit_nmse_db"] = gain_info.get("fit_nmse_db", float("nan"))
        metrics["target_gain_fit_corr"] = gain_info.get("fit_corr", float("nan"))

        metrics["sample_id"] = sample_id
        metrics["base_id"] = sample_name_to_base_id(sample_id)
        metrics["snr_label"] = get_snr_label_from_name_or_split(
            sample_id=sample_id,
            split_lookup=snr_lookup,
        )
        metrics["target_gain_base_id"] = gain_info.get("target_gain_base_id", "")

        results.append(metrics)

        should_save_audio = (
            save_audio
            and audio_out_dir is not None
            and (
                selected_sample_ids is None
                or sample_id in selected_sample_ids
            )
        )

        if should_save_audio:
            save_audio_bundle(
                audio_out_dir=audio_out_dir,
                sample_id=sample_id,
                M_np=M_np,
                H_np=H_np,
                L_np=L_np,
                H_hat=H_hat,
                L_hat=L_hat,
                sr=sr,
                save_references=save_audio_references,
                peak_normalise=peak_normalise_saved_audio,
            )

            saved_audio_count += 1

    if save_audio:
        print(
            f"Saved audio bundles: {saved_audio_count} "
            f"to {audio_out_dir}"
        )

    return results



####################
#Aggregation and reporting
####################

METRIC_KEYS = [
    "si_sdr_h",
    "si_sdr_l",

    "nmse_aligned_h",
    "nmse_aligned_l",
    "nmse_aligned_db_h",
    "nmse_aligned_db_l",

    "nmse_raw_h",
    "nmse_raw_l",
    "nmse_raw_db_h",
    "nmse_raw_db_l",

    "gain_error_db_h",
    "gain_error_db_l",
    "signed_alpha_h",
    "signed_alpha_l",
    "negative_alpha_h",
    "negative_alpha_l",
    "target_local_snr_db",

    "mix_nmse",
    "mix_nmse_db",
    "mix_gain_error_db",
    "mix_gamma",
    "negative_mix_gamma",
    "mix_corr",
    "polarity_calibration_applied",
    "polarity_flipped",
    "mix_gamma_before_calibration",
    "mix_corr_before_calibration",
    "mix_gamma_after_calibration",
    "mix_corr_after_calibration",
    "sdr_h",
    "sdr_l",
    "sir_h",
    "sir_l",
    "sar_h",
    "sar_l",
    "lsd_h",
    "lsd_l",
    "rho_h",
    "rho_l",
    "stoi_h",
    "stoi_l",
]


def aggregate(
    results: list[dict],
) -> dict:
    agg = {}

    for key in METRIC_KEYS:
        vals = [
            r[key]
            for r in results
            if key in r and np.isfinite(r[key])
        ]

        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")

    return agg


def format_metric(
    value,
    decimals: int = 2,
    signed: bool = True,
) -> str:
    if value is None:
        return "—"

    if isinstance(value, float) and np.isnan(value):
        return "—"

    sign = "+" if signed else ""

    return f"{value:{sign}.{decimals}f}"


def print_comparison_table(
    agg: dict,
    model_name: str = "Proposed model",
):
    methods = list(BASELINES.keys()) + [model_name]
    col_w = 30

    print()
    print("=" * 90)
    print("SEPARATION RESULTS — COMPARISON WITH UPDATED SYNTHETIC BASELINES")
    print("=" * 90)
    print()

    print("Heart sounds (HS)")
    print(
        f"{'Method':<{col_w}} "
        f"{'SI-SDR':>14} "
        f"{'SDR':>14} "
        f"{'SIR':>10} "
        f"{'SAR':>10} "
        f"{'rho':>10}"
    )
    print("─" * 90)

    for method in methods:
        if method == model_name:
            si = agg.get("si_sdr_h_mean", float("nan"))
            si_std = agg.get("si_sdr_h_std", float("nan"))
            sdr = agg.get("sdr_h_mean", float("nan"))
            sdr_std = agg.get("sdr_h_std", float("nan"))
            sir = agg.get("sir_h_mean", float("nan"))
            sar = agg.get("sar_h_mean", float("nan"))
            rho = agg.get("rho_h_mean", float("nan"))

            print(
                f"{'>> ' + method:<{col_w}} "
                f"{format_metric(si)}±{format_metric(si_std, signed=False):<7} "
                f"{format_metric(sdr)}±{format_metric(sdr_std, signed=False):<7} "
                f"{format_metric(sir):>10} "
                f"{format_metric(sar):>10} "
                f"{format_metric(rho, decimals=3):>10}"
            )

        else:
            b = BASELINES[method]["hs"]

            print(
                f"{method:<{col_w}} "
                f"{format_metric(b.get('SI-SDR')):>14} "
                f"{format_metric(b.get('SDR')):>14} "
                f"{format_metric(b.get('SIR')):>10} "
                f"{format_metric(b.get('SAR')):>10} "
                f"{format_metric(b.get('rho'), decimals=3):>10}"
            )

    print()
    print("Lung sounds (LS)")
    print(
        f"{'Method':<{col_w}} "
        f"{'SI-SDR':>14} "
        f"{'SDR':>14} "
        f"{'SIR':>10} "
        f"{'SAR':>10} "
        f"{'rho':>10}"
    )
    print("─" * 90)

    for method in methods:
        if method == model_name:
            si = agg.get("si_sdr_l_mean", float("nan"))
            si_std = agg.get("si_sdr_l_std", float("nan"))
            sdr = agg.get("sdr_l_mean", float("nan"))
            sdr_std = agg.get("sdr_l_std", float("nan"))
            sir = agg.get("sir_l_mean", float("nan"))
            sar = agg.get("sar_l_mean", float("nan"))
            rho = agg.get("rho_l_mean", float("nan"))

            print(
                f"{'>> ' + method:<{col_w}} "
                f"{format_metric(si)}±{format_metric(si_std, signed=False):<7} "
                f"{format_metric(sdr)}±{format_metric(sdr_std, signed=False):<7} "
                f"{format_metric(sir):>10} "
                f"{format_metric(sar):>10} "
                f"{format_metric(rho, decimals=3):>10}"
            )

        else:
            b = BASELINES[method]["ls"]

            print(
                f"{method:<{col_w}} "
                f"{format_metric(b.get('SI-SDR')):>14} "
                f"{format_metric(b.get('SDR')):>14} "
                f"{format_metric(b.get('SIR')):>10} "
                f"{format_metric(b.get('SAR')):>10} "
                f"{format_metric(b.get('rho'), decimals=3):>10}"
            )

    print()
    print("Additional metrics — proposed model only")
    print(f"{'Metric':<18} {'HS':>14} {'LS':>14}")
    print("─" * 50)

    add_pairs = [
        ("NMSE aligned", "nmse_aligned_h_mean", "nmse_aligned_l_mean"),
        ("NMSE align dB", "nmse_aligned_db_h_mean", "nmse_aligned_db_l_mean"),
        #("NMSE raw", "nmse_raw_h_mean", "nmse_raw_l_mean"),
        #("NMSE raw dB", "nmse_raw_db_h_mean", "nmse_raw_db_l_mean"),
        ("Gain error dB", "gain_error_db_h_mean", "gain_error_db_l_mean"),
        ("Signed alpha", "signed_alpha_h_mean", "signed_alpha_l_mean"),
        ("SIR (dB)", "sir_h_mean", "sir_l_mean"),
        ("SAR (dB)", "sar_h_mean", "sar_l_mean"),
        ("LSD", "lsd_h_mean", "lsd_l_mean"),
        ("STOI proxy", "stoi_h_mean", "stoi_l_mean"),
        ("rho", "rho_h_mean", "rho_l_mean"),
    ]

    for label, key_h, key_l in add_pairs:
        vh = agg.get(key_h, float("nan"))
        vl = agg.get(key_l, float("nan"))

        print(
            f"{label:<18} "
            f"{format_metric(vh, decimals=3):>14} "
            f"{format_metric(vl, decimals=3):>14}"
        )

    neg_h = 100.0 * agg.get("negative_alpha_h_mean", float("nan"))
    neg_l = 100.0 * agg.get("negative_alpha_l_mean", float("nan"))
    print(
        f"{'Negative alpha %':<18} "
        f"{neg_h:+14.3f} "
        f"{neg_l:+14.3f}"
    )

    print()
    print("Mixture reconstruction diagnostics — proposed model only")
    print("─" * 62)
    print(f"{'Mix NMSE':<24} {format_metric(agg.get('mix_nmse_mean', float('nan')), decimals=4):>14}")
    print(f"{'Mix NMSE dB':<24} {format_metric(agg.get('mix_nmse_db_mean', float('nan')), decimals=3):>14}")
    print(f"{'Mix gain error dB':<24} {format_metric(agg.get('mix_gain_error_db_mean', float('nan')), decimals=3):>14}")
    print(f"{'Mix gamma':<24} {format_metric(agg.get('mix_gamma_mean', float('nan')), decimals=3):>14}")
    print(f"{'Negative mix gamma %':<24} {100.0 * agg.get('negative_mix_gamma_mean', float('nan')):+14.3f}")
    print(f"{'Mix correlation':<24} {format_metric(agg.get('mix_corr_mean', float('nan')), decimals=3):>14}")
    if "polarity_flipped_mean" in agg:
        print()
        print("Mixture-informed global polarity calibration")
        print("─" * 62)
        print(
            f"{'Calibration enabled':<24} "
            f"{100.0 * agg.get('polarity_calibration_applied_mean', 0.0):+14.3f}%"
        )
        print(
            f"{'Output pairs flipped':<24} "
            f"{100.0 * agg.get('polarity_flipped_mean', 0.0):+14.3f}%"
        )
        print(
            f"{'Pre-calibration mix corr':<24} "
            f"{format_metric(agg.get('mix_corr_before_calibration_mean', float('nan')), decimals=3):>14}"
        )
    print()


def save_csv(
    results: list[dict],
    path: str,
):
    if not results:
        return

    fieldnames = [
        "sample_id",
        "base_id",
        "fold",
        "snr_label",
        "target_gain_mode",
        "target_gain_base_id",
        "target_gain_alpha_h",
        "target_gain_alpha_l",
        "target_gain_fit_nmse_db",
        "target_gain_fit_corr",
    ] + METRIC_KEYS

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for result in results:
            row = {}

            for key in fieldnames:
                value = result.get(key, "")

                if value is None:
                    row[key] = ""
                elif isinstance(value, str):
                    row[key] = value
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    value = float(value)
                    row[key] = f"{value:.4f}" if np.isfinite(value) else ""
                else:
                    row[key] = str(value)

            writer.writerow(row)

    print(f"Per-sample results saved: {path}")


def save_summary(
    agg: dict,
    path: str,
    model_name: str = "Proposed model",
):
    import io
    import contextlib

    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        print_comparison_table(agg, model_name=model_name)

    with open(path, "w") as f:
        f.write(buf.getvalue())
        f.write("\nFull aggregated metrics\n")
        f.write("─" * 40 + "\n")

        for key in METRIC_KEYS:
            mean = agg.get(f"{key}_mean", float("nan"))
            std = agg.get(f"{key}_std", float("nan"))

            f.write(f"{key:<16}: {mean:+.4f} ± {std:.4f}\n")

    print(f"Summary saved: {path}")


####################
#Fold evaluation
####################

def resolve_fold_checkpoint(
    ckpt_dir: str,
    fold_idx: int,
) -> str | None:
    fold_num = fold_idx + 1

    candidates = [
        os.path.join(ckpt_dir, f"scratch_fold{fold_num}_best.pt"),
        os.path.join(ckpt_dir, f"finetune_fold{fold_num}_best.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None

def read_best_checkpoint_info(
    ckpt_path: str,
    fold_no: int,
    device: torch.device,
) -> dict:
    """
    Read metadata from the selected best checkpoint.

    Note:
    - checkpoint['epoch'] is saved as zero-based index during training;
    - best_epoch is therefore epoch + 1, consistent with printed training logs.
    """
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    epoch_zero_based = ckpt.get("epoch", None)

    if epoch_zero_based is None:
        best_epoch = None
    else:
        best_epoch = int(epoch_zero_based) + 1

    return {
        "fold": fold_no,
        "checkpoint": os.path.basename(ckpt_path),
        "best_epoch": best_epoch,
        "epoch_zero_based": epoch_zero_based,
        "val_loss": ckpt.get("val_loss", float("nan")),
        "stage": ckpt.get("stage", ""),
        "ckpt_path": ckpt_path,
    }


def print_and_save_best_checkpoint_summary(
    rows: list[dict],
    out_dir: str,
) -> None:
    if not rows:
        return

    print()
    print("=" * 80)
    print("BEST CHECKPOINT EPOCH SUMMARY")
    print("=" * 80)

    for row in rows:
        best_epoch = row["best_epoch"]

        if best_epoch is None:
            best_epoch_text = "unknown"
        else:
            best_epoch_text = str(best_epoch)

        val_loss = row["val_loss"]

        if isinstance(val_loss, float) and np.isfinite(val_loss):
            val_loss_text = f"{val_loss:.6f}"
        else:
            val_loss_text = "unknown"

        print(
            f"Fold {row['fold']}: "
            f"best checkpoint confirmed at epoch {best_epoch_text} | "
            f"val_loss={val_loss_text} | "
            f"stage={row['stage']} | "
            f"{row['checkpoint']}"
        )

    print("=" * 80)

    out_path = os.path.join(
        out_dir,
        "best_checkpoint_epochs.csv",
    )

    pd.DataFrame(rows).to_csv(
        out_path,
        index=False,
    )

    print(f"Best checkpoint epoch summary saved: {out_path}")

def get_eval_base_ids(test_dir: str, seed: int) -> list[str]:
    get_base_triplet_ids = get_base_triplet_ids_func()
    base_ids_all = get_base_triplet_ids(test_dir)
    base_ids_all = [normalise_base_id_for_split(x) for x in base_ids_all]

    max_base_triplets = getattr(cfg, "SMOKE_MAX_BASE_TRIPLETS", None)

    if max_base_triplets is not None and max_base_triplets < len(base_ids_all):
        rng = np.random.default_rng(seed)

        selected_idx = rng.choice(
            len(base_ids_all),
            size=max_base_triplets,
            replace=False,
        )

        base_ids = [
            base_ids_all[i]
            for i in sorted(selected_idx)
        ]

        print(
            f"[SMOKE EVAL MODE] Using {len(base_ids)} base triplets "
            f"out of {len(base_ids_all)}"
        )

        return base_ids

    return base_ids_all


def normalise_base_id_for_split(value) -> str:
    """
    Normalize base/triplet identifiers so that split CSV ids and dataset ids match.

    Handles examples such as:
    - 000001
    - 1
    - 1.0
    - M_000001
    - H_000001
    - L_000001
    - 000001_s003_orig
    - M_000001_s003_orig
    """
    if value is None:
        return ""

    s = str(value).strip()

    if s == "" or s.lower() == "nan":
        return ""

    s = Path(s).stem

    if s.startswith("M_") or s.startswith("H_") or s.startswith("L_"):
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
    """
    Convert dataset segment name to base id.

    Example:
    000001_s000_orig -> 000001
    M_000001_s000_orig -> 000001
    """
    return normalise_base_id_for_split(name)


def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_source_disjoint_val_bases(
    split_csv: str,
    fold_no: int,
) -> set[str]:
    """
    Load validation base ids for a given source-disjoint fold.

    This replaces the old KFold-based evaluation when
    USE_SOURCE_DISJOINT_SPLIT=True.
    """
    path = Path(split_csv)

    if not path.exists():
        raise FileNotFoundError(
            f"Source-disjoint split CSV not found: {path}"
        )

    df = pd.read_csv(path)

    fold_col = pick_existing_column(
        df,
        ["fold_no", "fold", "fold_idx"],
    )

    split_col = pick_existing_column(
        df,
        ["split", "set", "subset"],
    )

    base_col = pick_existing_column(
        df,
        ["base_id", "case_id", "triplet_id", "id"],
    )

    missing = []

    if fold_col is None:
        missing.append("fold_no/fold/fold_idx")

    if split_col is None:
        missing.append("split/set/subset")

    if base_col is None:
        missing.append("base_id/case_id/triplet_id/id")

    if missing:
        raise RuntimeError(
            "Missing required columns in source-disjoint split CSV: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    fold_values = pd.to_numeric(df[fold_col], errors="coerce")
    split_values = df[split_col].astype(str).str.lower().str.strip()

    val_df = df[
        (fold_values == int(fold_no))
        & (split_values == "val")
    ].copy()

    val_bases = {
        normalise_base_id_for_split(x)
        for x in val_df[base_col].tolist()
    }

    val_bases = {x for x in val_bases if x}

    if not val_bases:
        raise RuntimeError(
            f"No validation base ids found for fold {fold_no} "
            f"in source-disjoint split CSV: {path}"
        )

    print(
        f"SOURCE-DISJOINT EVALUATION ENABLED | "
        f"fold={fold_no} | val bases={len(val_bases)} | "
        f"split_csv={path}"
    )

    return val_bases


def get_snr_label_from_name_or_split(
    sample_id: str,
    split_lookup: dict[str, str] | None = None,
) -> str:
    """
    Best-effort SNR label extraction for optional per-SNR analysis.

    Priority:
    1. split_lookup by base id, if available;
    2. sample name tokens, if the SNR label is embedded in the filename;
    3. unknown.
    """
    base_id = normalise_base_id_for_split(sample_id)

    if split_lookup is not None and base_id in split_lookup:
        return str(split_lookup[base_id])

    s = str(sample_id)
    lower = s.lower()

    if "snr" in lower:
        tokens = s.replace("-", "_minus").replace("+", "_plus").split("_")
        for i, tok in enumerate(tokens):
            if tok.lower() == "snr" and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                nxt = nxt.replace("minus", "-").replace("plus", "+")
                return nxt

    return "unknown"


def load_source_disjoint_snr_lookup(
    split_csv: str,
) -> dict[str, str]:
    """
    Optional helper for per-SNR reporting.
    Returns base_id -> snr_label when the split CSV contains snr_label.
    """
    path = Path(split_csv)

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    base_col = pick_existing_column(
        df,
        ["base_id", "case_id", "triplet_id", "id"],
    )

    snr_col = pick_existing_column(
        df,
        ["snr_label", "snr", "snr_db", "snr_condition"],
    )

    if base_col is None or snr_col is None:
        return {}

    lookup = {}

    for _, row in df[[base_col, snr_col]].drop_duplicates().iterrows():
        base = normalise_base_id_for_split(row[base_col])
        if base:
            lookup[base] = str(row[snr_col])

    return lookup


def print_per_snr_summary(results: list[dict]) -> pd.DataFrame | None:
    """
    Print a compact per-SNR diagnostic table if snr_label exists in results.
    Returns the table as a DataFrame so it can also be saved.
    """
    if not results:
        return None

    df = pd.DataFrame(results)

    if "snr_label" not in df.columns:
        return None

    if df["snr_label"].isna().all():
        return None

    print()
    print("=" * 90)
    print("PER-SNR DIAGNOSTIC SUMMARY — PROPOSED MODEL ONLY")
    print("=" * 90)

    rows = []

    preferred_order = ["-6", "-3", "0", "3", "+3", "6", "+6", "No", "no", "unknown"]

    labels = list(df["snr_label"].astype(str).unique())

    def order_key(x):
        if x in preferred_order:
            return preferred_order.index(x)
        try:
            return 100 + float(x.replace("+", ""))
        except Exception:
            return 999

    for snr in sorted(labels, key=order_key):
        sub = df[df["snr_label"].astype(str) == str(snr)]

        rows.append(
            {
                "SNR": snr,
                "N": len(sub),
                "HS SI-SDR": sub["si_sdr_h"].mean(),
                "LS SI-SDR": sub["si_sdr_l"].mean(),
                "HS SIR": sub["sir_h"].mean(),
                "LS SIR": sub["sir_l"].mean(),
                "HS SAR": sub["sar_h"].mean(),
                "LS SAR": sub["sar_l"].mean(),
                "HS rho": sub["rho_h"].mean(),
                "LS rho": sub["rho_l"].mean(),
                "Mix corr": sub["mix_corr"].mean(),
                "Mix NMSE dB": sub["mix_nmse_db"].mean(),
            }
        )

    out = pd.DataFrame(rows)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        160,
    ):
        print(out.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("=" * 90)
    print()

    return out


def save_per_snr_summary(
    results: list[dict],
    out_dir: str,
) -> None:
    per_snr = print_per_snr_summary(results)

    if per_snr is None:
        return

    out_path = Path(out_dir) / "per_snr_summary.csv"
    per_snr.to_csv(out_path, index=False)
    print(f"Per-SNR summary saved: {out_path}")


def get_triplet_dataset_class():
    """
    Import TripletDataset from train_disjoint when available.
    Falls back to train.py for old experiments.
    """
    try:
        from train_disjoint import TripletDataset
        return TripletDataset
    except Exception:
        from train import TripletDataset
        return TripletDataset


def get_base_triplet_ids_func():
    """
    Import get_base_triplet_ids from train_disjoint when available.
    Falls back to train.py for old experiments.
    """
    try:
        from train_disjoint import get_base_triplet_ids
        return get_base_triplet_ids
    except Exception:
        from train import get_base_triplet_ids
        return get_base_triplet_ids


def evaluate_all_folds(
    args,
    device: torch.device,
):
    TripletDataset = get_triplet_dataset_class()

    dataset = TripletDataset(
        args.test_dir,
        sr=args.sr,
        segment_samples=args.seg_samples,
    )

    use_source_disjoint = bool(
        getattr(args, "use_source_disjoint_split", False)
    )

    split_csv = getattr(args, "source_disjoint_split_csv", "")

    if use_source_disjoint:
        if split_csv is None or str(split_csv).strip() == "":
            raise RuntimeError(
                "use_source_disjoint_split=True but source_disjoint_split_csv is empty."
            )

        print()
        print("=" * 80)
        print("SOURCE-DISJOINT EVALUATION ENABLED")
        print("=" * 80)
        print(f"Split CSV: {split_csv}")
        print(
            "Expected all-SNR source-disjoint fold size: "
            "600 validation base triplets -> 16200  validation segments"
        )
        print("=" * 80)
        print()

        snr_lookup = load_source_disjoint_snr_lookup(split_csv)

    else:
        print()
        print("=" * 80)
        print("STANDARD KFOLD EVALUATION ENABLED")
        print("=" * 80)
        print(
            "WARNING: this is the old KFold evaluation. "
            "For strict all-SNR source-disjoint results, "
            "set USE_SOURCE_DISJOINT_SPLIT=True in model_config.py."
        )
        print("=" * 80)
        print()

        base_ids = get_eval_base_ids(args.test_dir, args.seed)
        n_base = len(base_ids)

        kf = KFold(
            n_splits=args.n_folds,
            shuffle=True,
            random_state=args.seed,
        )

        standard_kfold_val_bases = []

        for _, val_base_idx in kf.split(range(n_base)):
            standard_kfold_val_bases.append(
                {
                    normalise_base_id_for_split(base_ids[i])
                    for i in val_base_idx
                }
            )

        snr_lookup = {}

    selected_sample_ids = None

    if args.save_audio and args.save_selected_audio_only:
        selected_sample_ids = load_selected_sample_ids(
            args.selected_cases_csv
        )

    excluded_window_keys = set()

    if getattr(cfg, "FILTER_LOCAL_SNR", False):
        excluded_window_keys = load_excluded_window_keys(
            analysis_csv=cfg.LOCAL_SNR_ANALYSIS_CSV,
            threshold_db=cfg.LOCAL_SNR_THRESHOLD_DB,
        )

    all_results = []
    best_checkpoint_rows = []

    for fold_idx in range(args.n_folds):
        fold_no = fold_idx + 1
        only_fold = getattr(cfg, "ONLY_FOLD", None)

        if only_fold is not None and fold_no != only_fold:
            print(f"Skipping Fold {fold_no}/{args.n_folds}")
            continue

        ckpt_path = resolve_fold_checkpoint(
            args.ckpt_dir,
            fold_idx,
        )

        if ckpt_path is None:
            print(
                f"[WARN] Missing checkpoint for fold {fold_no} "
                f"in {args.ckpt_dir}"
            )
            continue

        print(f"\n── Fold {fold_no}/{args.n_folds} ──")
        print(f"Checkpoint: {ckpt_path}")

        target_gain_info = read_target_gain_info_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
        )

        print(
            f"Evaluation target gain: mode={target_gain_info.get('target_gain_mode')} | "
            f"alpha_h={target_gain_info.get('alpha_h', 1.0):+.6f} | "
            f"alpha_l={target_gain_info.get('alpha_l', 1.0):+.6f}"
        )

        best_checkpoint_rows.append(
            read_best_checkpoint_info(
                ckpt_path=ckpt_path,
                fold_no=fold_no,
                device=device,
            )
        )

        if use_source_disjoint:
            val_bases = load_source_disjoint_val_bases(
                split_csv=split_csv,
                fold_no=fold_no,
            )
        else:
            val_bases = standard_kfold_val_bases[fold_idx]

        test_names_before_filter = [
            name
            for name in dataset.names
            if sample_name_to_base_id(name) in val_bases
            and name.endswith("_orig")
        ]

        test_names = [
            name
            for name in test_names_before_filter
            if not is_excluded_window(name, excluded_window_keys)
        ]

        removed_test_names = [
            name
            for name in test_names_before_filter
            if is_excluded_window(name, excluded_window_keys)
        ]

        test_files = [
            (
                dataset.root / f"M_{name}.wav",
                dataset.root / f"H_{name}.wav",
                dataset.root / f"L_{name}.wav",
            )
            for name in test_names
        ]

        print(f"Validation base triplets: {len(val_bases)}")
        print(
            f"Test original segments: {len(test_names_before_filter)} -> "
            f"{len(test_files)} | "
            f"removed extreme windows={len(removed_test_names)}"
        )

        if use_source_disjoint and len(test_names_before_filter) != 16200 :
            print(f"look")

        model = load_model(
            ckpt_path,
            T=args.seg_samples,
            device=device,
        )

        audio_dir = (
            os.path.join(args.out_dir, f"audio_fold{fold_no}")
            if args.save_audio
            else None
        )

        fold_results = run_inference(
            model=model,
            test_files=test_files,
            sr=args.sr,
            seg_samples=args.seg_samples,
            device=device,
            save_audio=args.save_audio,
            audio_out_dir=audio_dir,
            selected_sample_ids=selected_sample_ids,
            save_audio_references=args.save_audio_references,
            peak_normalise_saved_audio=args.peak_normalise_saved_audio,
            apply_mixture_polarity_calibration=args.apply_mixture_polarity_calibration,
            apply_mixture_gain_calibration=args.apply_mixture_gain_calibration,
            target_gain_info=target_gain_info,
            snr_lookup=snr_lookup,
        )

        for r in fold_results:
            r["fold"] = fold_no

        csv_path = os.path.join(
            args.out_dir,
            f"results_fold{fold_no}.csv",
        )

        save_csv(fold_results, csv_path)

        all_results.extend(fold_results)

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_and_save_best_checkpoint_summary(
        rows=best_checkpoint_rows,
        out_dir=args.out_dir,
    )

    return all_results



def evaluate_single_ckpt(
    args,
    device: torch.device,
):
    TripletDataset = get_triplet_dataset_class()

    dataset = TripletDataset(
        args.test_dir,
        sr=args.sr,
        segment_samples=args.seg_samples,
    )

    excluded_window_keys = set()

    if getattr(cfg, "FILTER_LOCAL_SNR", False):
        excluded_window_keys = load_excluded_window_keys(
            analysis_csv=cfg.LOCAL_SNR_ANALYSIS_CSV,
            threshold_db=cfg.LOCAL_SNR_THRESHOLD_DB,
        )

    test_names_before_filter = [
        name
        for name in dataset.names
        if name.endswith("_orig")
    ]

    test_names = [
        name
        for name in test_names_before_filter
        if not is_excluded_window(name, excluded_window_keys)
    ]

    test_files = [
        (
            dataset.root / f"M_{name}.wav",
            dataset.root / f"H_{name}.wav",
            dataset.root / f"L_{name}.wav",
        )
        for name in test_names
    ]

    print(
        f"Evaluating original segments: {len(test_names_before_filter)} -> "
        f"{len(test_files)} | "
        f"removed extreme windows="
        f"{len(test_names_before_filter) - len(test_names)}"
    )
    print(f"Checkpoint: {args.ckpt}")

    target_gain_info = read_target_gain_info_from_checkpoint(
        ckpt_path=args.ckpt,
        device=device,
    )

    print(
        f"Evaluation target gain: mode={target_gain_info.get('target_gain_mode')} | "
        f"alpha_h={target_gain_info.get('alpha_h', 1.0):+.6f} | "
        f"alpha_l={target_gain_info.get('alpha_l', 1.0):+.6f}"
    )

    model = load_model(
        args.ckpt,
        T=args.seg_samples,
        device=device,
    )

    audio_dir = (
        os.path.join(args.out_dir, "audio")
        if args.save_audio
        else None
    )

    selected_sample_ids = None

    if args.save_audio and args.save_selected_audio_only:
        selected_sample_ids = load_selected_sample_ids(
            args.selected_cases_csv
        )

    snr_lookup = {}

    if getattr(args, "use_source_disjoint_split", False):
        split_csv = getattr(args, "source_disjoint_split_csv", "")
        if split_csv:
            snr_lookup = load_source_disjoint_snr_lookup(split_csv)

    return run_inference(
        model=model,
        test_files=test_files,
        sr=args.sr,
        seg_samples=args.seg_samples,
        device=device,
        save_audio=args.save_audio,
        audio_out_dir=audio_dir,
        selected_sample_ids=selected_sample_ids,
        save_audio_references=args.save_audio_references,
        peak_normalise_saved_audio=args.peak_normalise_saved_audio,
        apply_mixture_polarity_calibration=args.apply_mixture_polarity_calibration,
        apply_mixture_gain_calibration=args.apply_mixture_gain_calibration,
        target_gain_info=target_gain_info,
        snr_lookup=snr_lookup,
    )



#Config

class Config:
    ckpt = None
    ckpt_dir = cfg.CKPT_DIR
    test_dir = cfg.SUPERVISED_DIR

    out_dir = cfg.RESULTS_DIR
    apply_mixture_polarity_calibration = bool(
        getattr(cfg, "APPLY_MIXTURE_POLARITY_CALIBRATION", False)
    )

    apply_mixture_gain_calibration = bool(
        getattr(cfg, "APPLY_MIXTURE_GAIN_CALIBRATION", False)
    )

    use_source_disjoint_split = bool(
        getattr(cfg, "USE_SOURCE_DISJOINT_SPLIT", False)
    )

    source_disjoint_split_csv = getattr(
        cfg,
        "SOURCE_DISJOINT_SPLIT_CSV",
        "",
    )
    model_name = (
        "Evaluation of the run "
    )
    n_folds = cfg.N_FOLDS
    seed = cfg.SEED
    sr = cfg.SR
    seg_samples = cfg.SEG_SAMPLES
    save_audio = False
    save_selected_audio_only = False

    selected_cases_csv = os.environ.get(
        "ESD_JASSNET_SELECTED_CASES_CSV",
        str(
            Path(getattr(cfg, "PROJECT_ROOT", Path(__file__).resolve().parents[2]))
            / "outputs"
            / "results"
            / "eval_synth_1000_medium"
            / "failure_analysis"
            / "cases_to_listen_top30.csv"
        ),
    )

    save_audio_references = False #if true it saves also mixture and target reference
    peak_normalise_saved_audio = False #true-> normalizes audio to listen them more clearly
    generate_prediction_plots_only =False
    export_best_worst_audio = False
    plot_top_k = 5 #number of best and worst cases to plot
    GAIN_RESCALE_MODE = 'none' #gt/none


####################
#MAIN
####################

def main():
    args = Config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(
        "Evaluation config: "
        f"N_FILTERS={cfg.N_FILTERS}, "
        f"N_LATENT={cfg.N_LATENT}, "
        f"ATTN_WINDOW={cfg.ATTN_WINDOW}, "
        f"MASK_SCALE={cfg.MASK_SCALE}, "
        f"LAMBDA_MIX={getattr(cfg, 'LAMBDA_MIX', 0.0)}, "
        f"CKPT_DIR={args.ckpt_dir}, "
        f"OUT_DIR={args.out_dir}, "
        f"TEST_DIR={args.test_dir}, "
        f"SAVE_AUDIO={args.save_audio}, "
        f"POLARITY_CALIBRATION={args.apply_mixture_polarity_calibration}, "
        f"GAIN_CALIBRATION={args.apply_mixture_gain_calibration}, "
        f"USE_SOURCE_DISJOINT_SPLIT={args.use_source_disjoint_split}, "
        f"SOURCE_DISJOINT_SPLIT_CSV={args.source_disjoint_split_csv}, "
        f"SELECTED_ONLY={args.save_selected_audio_only}"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    if args.generate_prediction_plots_only:
        generate_best_worst_prediction_plots(
            args=args,
            device=device,
        )
        return

    if args.ckpt is not None:
        results = evaluate_single_ckpt(args, device)
    else:
        results = evaluate_all_folds(args, device)

    if not results:
        print("No results collected. Check checkpoints and test_dir.")
        return

    agg = aggregate(results)

    print_comparison_table(
        agg,
        model_name=args.model_name,
    )

    save_per_snr_summary(
        results,
        args.out_dir,
    )

    save_csv(
        results,
        os.path.join(args.out_dir, "results_all.csv"),
    )

    save_summary(
        agg,
        os.path.join(args.out_dir, "summary.txt"),
        model_name=args.model_name,
    )
    if getattr(args, "generate_prediction_plots_after_eval", False):
        generate_best_worst_prediction_plots(
            args=args,
            device=device,
        )
if __name__ == "__main__":
    main()

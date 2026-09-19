import os
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import model_config as cfg
from snr_filter import (
    count_unique_windows,
    filter_dataset_indices,
    load_excluded_window_keys,
)

#################
# Loss functions
#################

def si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio.

    estimate, target: [B, 1, T]
    Returns one SI-SDR value per sample: [B]
    """
    est = estimate.squeeze(1)
    tgt = target.squeeze(1)

    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    dot = (est * tgt).sum(dim=-1, keepdim=True)
    norm = (tgt * tgt).sum(dim=-1, keepdim=True) + eps
    alpha = dot / norm

    projected_target = alpha * tgt
    residual_noise = est - projected_target

    si_sdr_value = 10.0 * torch.log10(
        (projected_target * projected_target).sum(dim=-1)
        /
        ((residual_noise * residual_noise).sum(dim=-1) + eps)
    )

    return si_sdr_value


def si_sdr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Minimising this loss maximises SI-SDR.
    """
    return -si_sdr(estimate, target).mean()

def rms_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Unsigned amplitude loss.

    Penalises differences between the RMS amplitude of the prediction and
    the RMS amplitude of the target, after DC removal.

    This constrains output loudness, but it does not detect polarity
    inversions because RMS(x) == RMS(-x).
    """
    est = estimate.squeeze(1)
    tgt = target.squeeze(1)

    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    rms_est = torch.sqrt((est ** 2).mean(dim=-1) + eps)
    rms_tgt = torch.sqrt((tgt ** 2).mean(dim=-1) + eps)

    return ((rms_est - rms_tgt) ** 2).mean()

def polarity_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    One-sided polarity penalty.

    Penalises only negative correlation between prediction and target.
    If polarity is already correct, this term becomes zero and does not
    directly push the model toward a different positive-scale solution.
    """
    est = estimate.squeeze(1)
    tgt = target.squeeze(1)

    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    corr = F.cosine_similarity(
        est,
        tgt,
        dim=-1,
        eps=eps,
    )

    return F.relu(-corr).mean()

def rms_loss_per_sample(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return one RMS loss value per sample: [B].
    """
    est = estimate.squeeze(1)
    tgt = target.squeeze(1)

    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    rms_est = torch.sqrt((est ** 2).mean(dim=-1) + eps)
    rms_tgt = torch.sqrt((tgt ** 2).mean(dim=-1) + eps)

    return (rms_est - rms_tgt) ** 2


def polarity_loss_per_sample(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return one one-sided polarity penalty value per sample: [B].
    """
    est = estimate.squeeze(1)
    tgt = target.squeeze(1)

    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    corr = F.cosine_similarity(
        est,
        tgt,
        dim=-1,
        eps=eps,
    )

    return F.relu(-corr)


def source_loss_weights(
    H_ref: torch.Tensor,
    L_ref: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample HS/LS weights from the reference local SNR.

    If HS is locally weaker, HS loss receives a larger weight.
    If LS is locally weaker, LS loss receives a larger weight.
    Otherwise both weights remain 1.

    Returns:
        w_h, w_l: tensors with shape [B]
    """
    use_dynamic = bool(getattr(cfg, "USE_DYNAMIC_SOURCE_WEIGHTS", False))

    if not use_dynamic:
        w_h = H_ref.new_full(
            (H_ref.shape[0],),
            float(getattr(cfg, "LOSS_WEIGHT_H", 1.0)),
        )
        w_l = L_ref.new_full(
            (L_ref.shape[0],),
            float(getattr(cfg, "LOSS_WEIGHT_L", 1.0)),
        )
        return w_h, w_l

    h = H_ref.squeeze(1)
    l = L_ref.squeeze(1)

    h = h - h.mean(dim=-1, keepdim=True)
    l = l - l.mean(dim=-1, keepdim=True)

    rms_h = torch.sqrt((h ** 2).mean(dim=-1) + eps)
    rms_l = torch.sqrt((l ** 2).mean(dim=-1) + eps)

    snr_db = 20.0 * torch.log10((rms_h + eps) / (rms_l + eps))

    threshold = float(getattr(cfg, "DYNAMIC_SOURCE_WEIGHT_THRESHOLD_DB", 3.0))
    weak_weight = float(getattr(cfg, "DYNAMIC_SOURCE_WEIGHT_WEAK", 2.0))
    strong_weight = float(getattr(cfg, "DYNAMIC_SOURCE_WEIGHT_STRONG", 1.0))

    w_h = torch.ones_like(snr_db)
    w_l = torch.ones_like(snr_db)

    # snr_db < 0 means HS is weaker than LS.
    w_h = torch.where(snr_db <= -threshold, torch.full_like(w_h, weak_weight), w_h)
    w_l = torch.where(snr_db <= -threshold, torch.full_like(w_l, strong_weight), w_l)

    # snr_db > 0 means LS is weaker than HS.
    w_h = torch.where(snr_db >= threshold, torch.full_like(w_h, strong_weight), w_h)
    w_l = torch.where(snr_db >= threshold, torch.full_like(w_l, weak_weight), w_l)

    return w_h, w_l

def snr_sample_weights(
    H_ref: torch.Tensor,
    L_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute one sample-level weight per item in the batch.

    This does not change the relative HS/LS loss balance.
    It only makes locally imbalanced samples count more.
    """
    use_snr_weights = bool(getattr(cfg, "USE_SNR_SAMPLE_WEIGHTS", False))

    if not use_snr_weights:
        return H_ref.new_ones((H_ref.shape[0],))

    h = H_ref.squeeze(1)
    l = L_ref.squeeze(1)

    h = h - h.mean(dim=-1, keepdim=True)
    l = l - l.mean(dim=-1, keepdim=True)

    rms_h = torch.sqrt((h ** 2).mean(dim=-1) + eps)
    rms_l = torch.sqrt((l ** 2).mean(dim=-1) + eps)

    local_snr_db = 20.0 * torch.log10((rms_h + eps) / (rms_l + eps))
    abs_snr_db = torch.abs(local_snr_db)

    mid_db = float(getattr(cfg, "SNR_SAMPLE_WEIGHT_MID_DB", 3.0))
    extreme_db = float(getattr(cfg, "SNR_SAMPLE_WEIGHT_EXTREME_DB", 6.0))

    mid_weight = float(getattr(cfg, "SNR_SAMPLE_WEIGHT_MID", 1.25))
    extreme_weight = float(getattr(cfg, "SNR_SAMPLE_WEIGHT_EXTREME", 1.50))

    w = torch.ones_like(abs_snr_db)

    w = torch.where(
        abs_snr_db >= mid_db,
        torch.full_like(w, mid_weight),
        w,
    )

    w = torch.where(
        abs_snr_db >= extreme_db,
        torch.full_like(w, extreme_weight),
        w,
    )

    return w

def weighted_pair_mean(
    loss_h: torch.Tensor,
    loss_l: torch.Tensor,
    w_h: torch.Tensor,
    w_l: torch.Tensor,
    sample_w: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Average two per-sample source losses using source weights and optional
    sample-level SNR weights.
    """
    per_sample = (w_h * loss_h + w_l * loss_l) / (w_h + w_l + eps)

    if sample_w is None:
        return per_sample.mean()

    return (sample_w * per_sample).sum() / (sample_w.sum() + eps)

def mixture_consistency_loss(
    mixture: torch.Tensor,
    H_hat: torch.Tensor,
    L_hat: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalise inconsistency between the observed mixture and the sum of the two
    predicted sources.

    This term is useful during V2 adaptation because the source-level SI-SDR is
    scale-invariant. It encourages H_hat + L_hat to keep the same physical scale
    as the input mixture.
    """
    mix = mixture.squeeze(1)
    pred_mix = (H_hat + L_hat).squeeze(1)

    mix = mix - mix.mean(dim=-1, keepdim=True)
    pred_mix = pred_mix - pred_mix.mean(dim=-1, keepdim=True)

    error = mix - pred_mix
    nmse = (error ** 2).sum(dim=-1) / ((mix ** 2).sum(dim=-1) + eps)

    return nmse.mean()


def mixture_polarity_loss(
    mixture: torch.Tensor,
    H_hat: torch.Tensor,
    L_hat: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Input-only polarity penalty for the reconstructed mixture.

    This discourages the global inverted basin H_hat ~= -H and L_hat ~= -L
    without using the target sources.
    """
    mix = mixture.squeeze(1)
    pred_mix = (H_hat + L_hat).squeeze(1)

    mix = mix - mix.mean(dim=-1, keepdim=True)
    pred_mix = pred_mix - pred_mix.mean(dim=-1, keepdim=True)

    corr = F.cosine_similarity(
        pred_mix,
        mix,
        dim=-1,
        eps=eps,
    )

    return F.relu(-corr).mean()


def combined_loss(
    H_hat: torch.Tensor,
    H_ref: torch.Tensor,
    L_hat: torch.Tensor,
    L_ref: torch.Tensor,
    M: torch.Tensor | None = None,
    lambda_l1: float = cfg.LAMBDA_L1,
    lambda_rms: float | None = None,
    lambda_polarity: float | None = None,
    lambda_mix: float | None = None,
    lambda_mix_polarity: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Composite objective:

        L_total =
            L_SI-SDR
            + lambda_l1 * L_SmoothL1
            + lambda_rms * L_RMS
            + lambda_polarity * L_Polarity
            + lambda_mix * L_MixtureConsistency
            + lambda_mix_polarity * L_MixturePolarity

    SI-SDR optimises scale-invariant separation quality.
    SmoothL1 preserves waveform shape after peak normalisation.
    RMS loss constrains absolute output amplitude.
    Polarity loss penalises only negative correlation with the target.
    Mixture losses use only the observed mixture and the model outputs.
    """
    if lambda_rms is None:
        lambda_rms = float(getattr(cfg, "LAMBDA_RMS", 0.0))

    if lambda_polarity is None:
        lambda_polarity = float(getattr(cfg, "LAMBDA_POLARITY", 0.0))

    if lambda_mix is None:
        lambda_mix = float(getattr(cfg, "LAMBDA_MIX", 0.0))

    if lambda_mix_polarity is None:
        lambda_mix_polarity = float(getattr(cfg, "LAMBDA_MIX_POLARITY", 0.0))

    w_h, w_l = source_loss_weights(H_ref, L_ref)
    sample_w = snr_sample_weights(H_ref, L_ref)

    # --------------------------------------------------------------------------
    # 1. SI-SDR separation term
    # --------------------------------------------------------------------------
    loss_sisdr_h_vec = -si_sdr(H_hat, H_ref)
    loss_sisdr_l_vec = -si_sdr(L_hat, L_ref)

    loss_sisdr_h = loss_sisdr_h_vec.mean()
    loss_sisdr_l = loss_sisdr_l_vec.mean()

    loss_sisdr = weighted_pair_mean(
        loss_sisdr_h_vec,
        loss_sisdr_l_vec,
        w_h,
        w_l,
        sample_w,
    )
    # --------------------------------------------------------------------------
    # 2. Smooth L1 waveform-shape term on peak-normalised signals
    # --------------------------------------------------------------------------
    eps = 1e-8

    H_hat_n = H_hat / (H_hat.abs().amax(dim=-1, keepdim=True) + eps)
    H_ref_n = H_ref / (H_ref.abs().amax(dim=-1, keepdim=True) + eps)

    L_hat_n = L_hat / (L_hat.abs().amax(dim=-1, keepdim=True) + eps)
    L_ref_n = L_ref / (L_ref.abs().amax(dim=-1, keepdim=True) + eps)

    loss_l1_h_vec = F.smooth_l1_loss(H_hat_n, H_ref_n, reduction="none").mean(dim=(1, 2))
    loss_l1_l_vec = F.smooth_l1_loss(L_hat_n, L_ref_n, reduction="none").mean(dim=(1, 2))

    loss_l1_h = loss_l1_h_vec.mean()
    loss_l1_l = loss_l1_l_vec.mean()

    loss_l1 = weighted_pair_mean(
        loss_l1_h_vec,
        loss_l1_l_vec,
        w_h,
        w_l,
        sample_w,
    )

    weighted_loss_l1 = lambda_l1 * loss_l1

    # --------------------------------------------------------------------------
    # 3. RMS amplitude term on raw reconstructed waveforms
    # --------------------------------------------------------------------------
    loss_rms_h_vec = rms_loss_per_sample(H_hat, H_ref)
    loss_rms_l_vec = rms_loss_per_sample(L_hat, L_ref)

    loss_rms_h = loss_rms_h_vec.mean()
    loss_rms_l = loss_rms_l_vec.mean()

    loss_rms = weighted_pair_mean(
        loss_rms_h_vec,
        loss_rms_l_vec,
        w_h,
        w_l,
        sample_w,
    )

    weighted_loss_rms = lambda_rms * loss_rms

    # --------------------------------------------------------------------------
    # 4. One-sided polarity penalty
    # --------------------------------------------------------------------------
    loss_polarity_h_vec = polarity_loss_per_sample(H_hat, H_ref)
    loss_polarity_l_vec = polarity_loss_per_sample(L_hat, L_ref)

    loss_polarity_h = loss_polarity_h_vec.mean()
    loss_polarity_l = loss_polarity_l_vec.mean()

    loss_polarity = weighted_pair_mean(
        loss_polarity_h_vec,
        loss_polarity_l_vec,
        w_h,
        w_l,
        sample_w,
    )

    weighted_loss_polarity = lambda_polarity * loss_polarity

    # --------------------------------------------------------------------------
    # 5. Mixture-level physical consistency terms
    # --------------------------------------------------------------------------
    if M is not None and (lambda_mix > 0.0 or lambda_mix_polarity > 0.0):
        loss_mix = mixture_consistency_loss(
            mixture=M,
            H_hat=H_hat,
            L_hat=L_hat,
        )
        loss_mix_polarity = mixture_polarity_loss(
            mixture=M,
            H_hat=H_hat,
            L_hat=L_hat,
        )
    else:
        loss_mix = H_hat.new_tensor(0.0)
        loss_mix_polarity = H_hat.new_tensor(0.0)

    weighted_loss_mix = lambda_mix * loss_mix
    weighted_loss_mix_polarity = lambda_mix_polarity * loss_mix_polarity

    # --------------------------------------------------------------------------
    # Total objective
    # --------------------------------------------------------------------------
    total = (
        loss_sisdr
        + weighted_loss_l1
        + weighted_loss_rms
        + weighted_loss_polarity
        + weighted_loss_mix
        + weighted_loss_mix_polarity
    )

    metrics = {
        "loss_total": float(total.item()),
        "snr_sample_weights": bool(getattr(cfg, "USE_SNR_SAMPLE_WEIGHTS", False)),
        "snr_sample_weight_mean": float(sample_w.mean().item()),
        "loss_sisdr": float(loss_sisdr.item()),
        "loss_sisdr_h": float(loss_sisdr_h.item()),
        "loss_sisdr_l": float(loss_sisdr_l.item()),

        "loss_l1": float(loss_l1.item()),
        "loss_l1_h": float(loss_l1_h.item()),
        "loss_l1_l": float(loss_l1_l.item()),
        "weighted_loss_l1": float(weighted_loss_l1.item()),

        "loss_rms": float(loss_rms.item()),
        "loss_rms_h": float(loss_rms_h.item()),
        "loss_rms_l": float(loss_rms_l.item()),
        "weighted_loss_rms": float(weighted_loss_rms.item()),

        "loss_polarity": float(loss_polarity.item()),
        "loss_polarity_h": float(loss_polarity_h.item()),
        "loss_polarity_l": float(loss_polarity_l.item()),
        "weighted_loss_polarity": float(weighted_loss_polarity.item()),

        "loss_mix": float(loss_mix.item()),
        "weighted_loss_mix": float(weighted_loss_mix.item()),
        "loss_mix_polarity": float(loss_mix_polarity.item()),
        "weighted_loss_mix_polarity": float(weighted_loss_mix_polarity.item()),

        "lambda_l1": float(lambda_l1),
        "lambda_rms": float(lambda_rms),
        "lambda_polarity": float(lambda_polarity),
        "lambda_mix": float(lambda_mix),
        "lambda_mix_polarity": float(lambda_mix_polarity),

        "loss_weight_h": float(w_h.mean().item()),
        "loss_weight_l": float(w_l.mean().item()),
        "dynamic_source_weights": bool(getattr(cfg, "USE_DYNAMIC_SOURCE_WEIGHTS", False)),
    }

    return total, metrics

###########
#Dataset
##########

class TripletDataset(Dataset):


    def __init__(
        self,
        root_dir: str,
        sr: int = cfg.SR,
        segment_samples: int = cfg.SEG_SAMPLES,
    ):
        self.root = Path(root_dir)
        self.sr = sr
        self.seg = segment_samples

        m_files = sorted(self.root.glob("M_*.wav"))
        self.names = [f.stem[2:] for f in m_files]

        if len(self.names) == 0:
            raise RuntimeError(f"No M_*.wav files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.names)

    #load wav file, verify sample rate and trim if needed
    def _load(self, path: Path) -> torch.Tensor:
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)

        if file_sr != self.sr:
            raise ValueError(
                f"Unexpected sample rate for {path}: "
                f"expected {self.sr}, got {file_sr}"
            )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if len(audio) > self.seg:
            audio = audio[:self.seg]
        elif len(audio) < self.seg:
            audio = np.pad(audio, (0, self.seg - len(audio)))

        return torch.from_numpy(audio).unsqueeze(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.names[idx]

        M = self._load(self.root / f"M_{name}.wav")
        H = self._load(self.root / f"H_{name}.wav")
        L = self._load(self.root / f"L_{name}.wav")

        return M, H, L

class SourceGainJitterDataset(Dataset):
    """
    Training-only source gain jitter.

    It loads the original clean triplet M, H, L from the base dataset,
    applies independent gain jitter to H and L, then rebuilds:

        M_aug = H_aug + L_aug

    This preserves exact mixture consistency while creating continuous
    SNR variations around the original fixed SNR levels.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        gain_jitter_db: float = 3.0,
        prob: float = 1.0,
        preserve_peak: bool = True,
        peak_value: float = 0.95,
        eps: float = 1e-8,
    ):
        self.base_dataset = base_dataset
        self.gain_jitter_db = float(gain_jitter_db)
        self.prob = float(prob)
        self.preserve_peak = bool(preserve_peak)
        self.peak_value = float(peak_value)
        self.eps = float(eps)

        if hasattr(base_dataset, "names"):
            self.names = base_dataset.names

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _sample_gain(self) -> float:
        gain_db = (2.0 * torch.rand(1).item() - 1.0) * self.gain_jitter_db
        return 10.0 ** (gain_db / 20.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, H, L = self.base_dataset[idx]

        if self.gain_jitter_db <= 0.0:
            return M, H, L

        if torch.rand(1).item() > self.prob:
            return M, H, L

        gain_h = self._sample_gain()
        gain_l = self._sample_gain()

        H_aug = H * gain_h
        L_aug = L * gain_l
        M_aug = H_aug + L_aug

        if self.preserve_peak:
            peak = torch.max(
                torch.stack([
                    M_aug.abs().max(),
                    H_aug.abs().max(),
                    L_aug.abs().max(),
                ])
            )

            if peak > self.peak_value:
                scale = self.peak_value / (peak + self.eps)
                M_aug = M_aug * scale
                H_aug = H_aug * scale
                L_aug = L_aug * scale

        return M_aug, H_aug, L_aug

class TargetGainDataset(Dataset):
    """
    Lightweight wrapper used during V2 fine-tuning.

    The input mixture remains unchanged, while the references are optionally
    scaled so that the supervised target becomes coherent with the observed
    V2 relation M ~= alpha * H + beta * L.
    """

    def __init__(
        self,
        base_dataset: TripletDataset,
        alpha_h: float = 1.0,
        alpha_l: float = 1.0,
    ):
        self.base_dataset = base_dataset
        self.alpha_h = float(alpha_h)
        self.alpha_l = float(alpha_l)
        self.names = base_dataset.names

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, H, L = self.base_dataset[idx]
        return M, H * self.alpha_h, L * self.alpha_l


def _base_id_from_name(name: str) -> str:
    return str(name).split("_s")[0]


def normalise_base_triplet_id(value: str) -> str:
    """
    Normalise a V2 base triplet identifier.

    Accepted inputs include examples such as:
    - "0112"
    - "112"
    - "M_0112_s000_orig"
    - "H_0112_s000_orig.wav"
    """
    s = Path(str(value)).stem.strip()

    if s.startswith(("M_", "H_", "L_")):
        s = s[2:]

    s = s.split("_s")[0]

    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return digits.zfill(4)

    return s


def _normalise_triplet_list(values) -> set[str]:
    if values is None:
        return set()

    if isinstance(values, str):
        values = [v.strip() for v in values.split(",") if v.strip()]

    return {
        normalise_base_triplet_id(v)
        for v in values
        if str(v).strip()
    }


def apply_base_triplet_filter(
    base_ids: list[str],
    context: str = "dataset",
) -> list[str]:
    """
    Apply optional V2 base-triplet allow/deny lists from model_config.py.

    This is intentionally applied at the base-triplet level before KFold, so no
    segment from a removed triplet can enter either training or validation.
    Evaluation imports get_base_triplet_ids() from this file, therefore it uses
    the same filtered base list automatically.
    """
    filtered = [normalise_base_triplet_id(b) for b in base_ids]
    original = list(filtered)

    use_allowlist = bool(getattr(cfg, "USE_BASE_TRIPLET_ALLOWLIST", False))
    allowlist = _normalise_triplet_list(
        getattr(cfg, "BASE_TRIPLET_ALLOWLIST", ())
    )
    denylist = _normalise_triplet_list(
        getattr(cfg, "BASE_TRIPLET_DENYLIST", ())
    )

    if use_allowlist:
        if not allowlist:
            raise ValueError(
                "USE_BASE_TRIPLET_ALLOWLIST=True but BASE_TRIPLET_ALLOWLIST is empty."
            )

        filtered = [b for b in filtered if b in allowlist]
        missing = sorted(allowlist - set(original))

        print(
            f"[BASE TRIPLET FILTER] {context}: allowlist enabled -> "
            f"{len(filtered)}/{len(original)} base triplets kept"
        )
        print(
            "[BASE TRIPLET FILTER] kept IDs: "
            + ", ".join(filtered)
        )

        if missing:
            print(
                "[BASE TRIPLET FILTER][WARNING] allowlisted IDs not found: "
                + ", ".join(missing)
            )

    if denylist:
        before = len(filtered)
        filtered = [b for b in filtered if b not in denylist]

        print(
            f"[BASE TRIPLET FILTER] {context}: denylist removed "
            f"{before - len(filtered)} base triplets"
        )

    if len(filtered) == 0:
        raise RuntimeError(
            "No base triplets left after applying BASE_TRIPLET_ALLOWLIST / "
            "BASE_TRIPLET_DENYLIST."
        )

    return filtered


class PerTripletTargetGainDataset(Dataset):
    """
    Apply a different alpha/beta pair to each original V2 triplet.

    This is useful for V2 because the scaled-linear relation is strong inside
    many individual recordings, but the alpha/beta values can vary widely from
    one triplet to another. A single fold-level gain can therefore be a poor
    target definition.
    """

    def __init__(
        self,
        base_dataset: TripletDataset,
        per_base_gains: dict[str, dict],
    ):
        self.base_dataset = base_dataset
        self.per_base_gains = per_base_gains
        self.names = base_dataset.names

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, H, L = self.base_dataset[idx]
        base = _base_id_from_name(self.names[idx])
        gains = self.per_base_gains.get(base, None)

        if gains is None:
            alpha_h = 1.0
            alpha_l = 1.0
        else:
            alpha_h = float(gains.get("alpha_h", 1.0))
            alpha_l = float(gains.get("alpha_l", 1.0))

        return M, H * alpha_h, L * alpha_l


def _dc_remove_np(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def estimate_v2_train_fold_gains(
    dataset: TripletDataset,
    indices: list[int],
    remove_dc: bool = True,
    eps: float = 1e-12,
) -> dict:
    """
    Estimate alpha and beta from the training split only:

        M ~= alpha * H + beta * L

    This avoids leakage from validation/test triplets into target preparation.
    The implementation accumulates the 2x2 normal equations without loading the
    whole dataset into memory.
    """
    if len(indices) == 0:
        raise RuntimeError("Cannot estimate V2 gains from an empty train split.")

    hh = 0.0
    ll = 0.0
    hl = 0.0
    hm = 0.0
    lm = 0.0
    n_samples = 0

    for idx in indices:
        M, H, L = dataset[idx]
        m = M.squeeze(0).numpy().astype(np.float64)
        h = H.squeeze(0).numpy().astype(np.float64)
        l = L.squeeze(0).numpy().astype(np.float64)

        if remove_dc:
            m = _dc_remove_np(m)
            h = _dc_remove_np(h)
            l = _dc_remove_np(l)

        hh += float(np.dot(h, h))
        ll += float(np.dot(l, l))
        hl += float(np.dot(h, l))
        hm += float(np.dot(h, m))
        lm += float(np.dot(l, m))
        n_samples += len(m)

    A = np.array([[hh, hl], [hl, ll]], dtype=np.float64)
    b = np.array([hm, lm], dtype=np.float64)

    if abs(np.linalg.det(A)) < eps:
        gains = np.linalg.pinv(A) @ b
    else:
        gains = np.linalg.solve(A, b)

    alpha_h = float(gains[0])
    alpha_l = float(gains[1])

    # Diagnostics for the scaled additive relation after applying the gains.
    residual_energy = 0.0
    mixture_energy = 0.0
    corr_num = 0.0
    corr_den_m = 0.0
    corr_den_p = 0.0

    for idx in indices:
        M, H, L = dataset[idx]
        m = M.squeeze(0).numpy().astype(np.float64)
        h = H.squeeze(0).numpy().astype(np.float64)
        l = L.squeeze(0).numpy().astype(np.float64)

        if remove_dc:
            m = _dc_remove_np(m)
            h = _dc_remove_np(h)
            l = _dc_remove_np(l)

        pred = alpha_h * h + alpha_l * l
        residual = m - pred

        residual_energy += float(np.dot(residual, residual))
        mixture_energy += float(np.dot(m, m))
        corr_num += float(np.dot(m, pred))
        corr_den_m += float(np.dot(m, m))
        corr_den_p += float(np.dot(pred, pred))

    nmse = residual_energy / (mixture_energy + eps)
    nmse_db = float(10.0 * np.log10(max(nmse, eps)))
    corr = float(corr_num / (np.sqrt(corr_den_m * corr_den_p) + eps))

    return {
        "alpha_h": alpha_h,
        "alpha_l": alpha_l,
        "fit_nmse_db": nmse_db,
        "fit_corr": corr,
        "fit_segments": len(indices),
        "fit_samples": n_samples,
        "remove_dc": bool(remove_dc),
    }


def estimate_v2_per_triplet_gains(
    dataset: TripletDataset,
    indices: list[int] | None = None,
    remove_dc: bool = True,
) -> dict:
    """
    Estimate one alpha/beta pair for each base triplet.

    The model is not given these gains as input. They only define the supervised
    waveform targets: H_target = alpha_base * H and L_target = beta_base * L.
    """
    if indices is None:
        indices = list(range(len(dataset)))

    by_base: dict[str, list[int]] = {}
    for idx in indices:
        base = _base_id_from_name(dataset.names[idx])
        by_base.setdefault(base, []).append(idx)

    per_base_gains = {}
    for base, base_indices in sorted(by_base.items(), key=lambda kv: kv[0]):
        g = estimate_v2_train_fold_gains(
            dataset=dataset,
            indices=base_indices,
            remove_dc=remove_dc,
        )
        per_base_gains[base] = {
            "alpha_h": float(g["alpha_h"]),
            "alpha_l": float(g["alpha_l"]),
            "fit_nmse_db": float(g["fit_nmse_db"]),
            "fit_corr": float(g["fit_corr"]),
            "fit_segments": int(g.get("fit_segments", len(base_indices))),
        }

    alpha_h_values = np.array(
        [v["alpha_h"] for v in per_base_gains.values()],
        dtype=np.float64,
    )
    alpha_l_values = np.array(
        [v["alpha_l"] for v in per_base_gains.values()],
        dtype=np.float64,
    )
    corr_values = np.array(
        [v["fit_corr"] for v in per_base_gains.values()],
        dtype=np.float64,
    )
    nmse_values = np.array(
        [v["fit_nmse_db"] for v in per_base_gains.values()],
        dtype=np.float64,
    )

    return {
        "target_gain_mode": "per_triplet",
        "per_base_gains": per_base_gains,
        "n_gain_bases": int(len(per_base_gains)),
        "alpha_h": float(np.mean(alpha_h_values)) if len(alpha_h_values) else 1.0,
        "alpha_l": float(np.mean(alpha_l_values)) if len(alpha_l_values) else 1.0,
        "alpha_h_min": float(np.min(alpha_h_values)) if len(alpha_h_values) else 1.0,
        "alpha_h_max": float(np.max(alpha_h_values)) if len(alpha_h_values) else 1.0,
        "alpha_l_min": float(np.min(alpha_l_values)) if len(alpha_l_values) else 1.0,
        "alpha_l_max": float(np.max(alpha_l_values)) if len(alpha_l_values) else 1.0,
        "fit_nmse_db": float(np.mean(nmse_values)) if len(nmse_values) else float("nan"),
        "fit_corr": float(np.mean(corr_values)) if len(corr_values) else float("nan"),
        "fit_corr_min": float(np.min(corr_values)) if len(corr_values) else float("nan"),
        "remove_dc": bool(remove_dc),
    }


def make_target_dataset_for_fold(
    dataset: TripletDataset,
    train_indices: list[int],
) -> tuple[Dataset, dict]:
    """
    Apply the target-gain policy configured in model_config.py.

    Modes:
    - none:        no scaling, targets are the dataset H/L files as-is.
    - global:      use cfg.TARGET_GAIN_H and cfg.TARGET_GAIN_L.
    - train_fold:  estimate one alpha/beta pair on the current train split.
    - per_triplet: estimate one alpha/beta pair for each original V2 triplet.
    """
    mode = str(getattr(cfg, "TARGET_GAIN_MODE", "none")).lower().strip()

    if mode in {"none", "off", "false"}:
        info = {
            "target_gain_mode": "none",
            "alpha_h": 1.0,
            "alpha_l": 1.0,
            "fit_nmse_db": float("nan"),
            "fit_corr": float("nan"),
        }
        return dataset, info

    if mode == "global":
        alpha_h = float(getattr(cfg, "TARGET_GAIN_H", 1.0))
        alpha_l = float(getattr(cfg, "TARGET_GAIN_L", 1.0))
        info = {
            "target_gain_mode": "global",
            "alpha_h": alpha_h,
            "alpha_l": alpha_l,
            "fit_nmse_db": float("nan"),
            "fit_corr": float("nan"),
        }
        return TargetGainDataset(dataset, alpha_h, alpha_l), info

    if mode == "train_fold":
        info = estimate_v2_train_fold_gains(
            dataset=dataset,
            indices=train_indices,
            remove_dc=bool(getattr(cfg, "TARGET_GAIN_REMOVE_DC", True)),
        )
        info["target_gain_mode"] = "train_fold"
        return TargetGainDataset(dataset, info["alpha_h"], info["alpha_l"]), info

    if mode in {"per_triplet", "per_base", "per_recording"}:
        info = estimate_v2_per_triplet_gains(
            dataset=dataset,
            indices=None,
            remove_dc=bool(getattr(cfg, "TARGET_GAIN_REMOVE_DC", True)),
        )
        return PerTripletTargetGainDataset(dataset, info["per_base_gains"]), info

    raise ValueError(
        "Unknown TARGET_GAIN_MODE. Use 'none', 'global', 'train_fold', "
        "or 'per_triplet'. "
        f"Got: {mode}"
    )


def print_target_gain_info(fold_no: int, info: dict) -> None:
    if info.get("target_gain_mode") == "per_triplet":
        print(
            f"Fold {fold_no} target gain mode=per_triplet | "
            f"n_bases={info.get('n_gain_bases', 0)} | "
            f"alpha_h_mean={info.get('alpha_h', float('nan')):+.6f} "
            f"[{info.get('alpha_h_min', float('nan')):+.3f}, {info.get('alpha_h_max', float('nan')):+.3f}] | "
            f"alpha_l_mean={info.get('alpha_l', float('nan')):+.6f} "
            f"[{info.get('alpha_l_min', float('nan')):+.3f}, {info.get('alpha_l_max', float('nan')):+.3f}] | "
            f"mean_fit_nmse={info.get('fit_nmse_db', float('nan')):+.2f} dB | "
            f"mean_fit_corr={info.get('fit_corr', float('nan')):+.6f} | "
            f"min_fit_corr={info.get('fit_corr_min', float('nan')):+.6f}"
        )
        return

    print(
        f"Fold {fold_no} target gain mode={info.get('target_gain_mode')} | "
        f"alpha_h={info.get('alpha_h', float('nan')):+.6f} | "
        f"alpha_l={info.get('alpha_l', float('nan')):+.6f} | "
        f"fit_nmse={info.get('fit_nmse_db', float('nan')):+.2f} dB | "
        f"fit_corr={info.get('fit_corr', float('nan')):+.6f}"
    )

#Return base triplet IDs used for leakage-safe cross-validation
def get_base_triplet_ids(supervised_dir: str) -> list[str]:

    root = Path(supervised_dir)
    orig_files = sorted(root.glob("M_*_orig.wav"))

    base_ids = []
    seen = set()

    for f in orig_files:
        name = f.stem[2:]
        base = normalise_base_triplet_id(name.split("_s")[0])

        if base not in seen:
            seen.add(base)
            base_ids.append(base)

    return apply_base_triplet_filter(
        sorted(base_ids),
        context=str(root),
    )


#################################
#Training and validation steps
#################################

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_l1: float = cfg.LAMBDA_L1,
    grad_clip: float = cfg.GRAD_CLIP,
) -> dict:
    model.train()

    accum = {}
    n_batches = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for M, H_ref, L_ref in progress:
        M = M.to(device, non_blocking=True)
        H_ref = H_ref.to(device, non_blocking=True)
        L_ref = L_ref.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        H_hat, L_hat = model(M)

        loss, metrics = combined_loss(
            H_hat=H_hat,
            H_ref=H_ref,
            L_hat=L_hat,
            L_ref=L_ref,
            M=M,
            lambda_l1=lambda_l1,
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        optimizer.step()

        for key, value in metrics.items():
            accum[key] = accum.get(key, 0.0) + value

        n_batches += 1

        progress.set_postfix({
            "loss": f"{metrics['loss_total']:.4f}",
            "sisdr": f"{metrics['loss_sisdr']:.4f}",
            "l1x": f"{metrics['weighted_loss_l1']:.5f}",
            "rmsx": f"{metrics['weighted_loss_rms']:.6f}",
            "polx": f"{metrics['weighted_loss_polarity']:.5f}",
            "mixx": f"{metrics['weighted_loss_mix']:.5f}",
            "mixpol": f"{metrics['weighted_loss_mix_polarity']:.5f}",
            "H": f"{-metrics['loss_sisdr_h']:.2f}dB",
            "L": f"{-metrics['loss_sisdr_l']:.2f}dB",
        })

    if n_batches == 0:
        raise RuntimeError("Training loader is empty.")

    return {
        key: value / n_batches
        for key, value in accum.items()
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_l1: float = cfg.LAMBDA_L1,
) -> dict:
    model.eval()

    accum = {}
    sisdr_h_list = []
    sisdr_l_list = []
    n_batches = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for M, H_ref, L_ref in progress:
        M = M.to(device, non_blocking=True)
        H_ref = H_ref.to(device, non_blocking=True)
        L_ref = L_ref.to(device, non_blocking=True)

        H_hat, L_hat = model(M)

        _, metrics = combined_loss(
            H_hat=H_hat,
            H_ref=H_ref,
            L_hat=L_hat,
            L_ref=L_ref,
            M=M,
            lambda_l1=lambda_l1,
        )

        for key, value in metrics.items():
            accum[key] = accum.get(key, 0.0) + value

        sisdr_h = -metrics["loss_sisdr_h"]
        sisdr_l = -metrics["loss_sisdr_l"]

        sisdr_h_list.append(sisdr_h)
        sisdr_l_list.append(sisdr_l)

        n_batches += 1

        progress.set_postfix({
            "loss": f"{metrics['loss_total']:.4f}",
            "rmsx": f"{metrics['weighted_loss_rms']:.6f}",
            "polx": f"{metrics['weighted_loss_polarity']:.5f}",
            "mixx": f"{metrics['weighted_loss_mix']:.5f}",
            "mixpol": f"{metrics['weighted_loss_mix_polarity']:.5f}",
            "H": f"{sisdr_h:.2f}dB",
            "L": f"{sisdr_l:.2f}dB",
        })

    if n_batches == 0:
        raise RuntimeError("Validation loader is empty.")

    result = {
        key: value / n_batches
        for key, value in accum.items()
    }

    result["si_sdr_h_dB"] = float(np.mean(sisdr_h_list))
    result["si_sdr_l_dB"] = float(np.mean(sisdr_l_list))

    return result


############################################
# Mixed synthetic replay + clean V2 training
############################################

def _next_cycled_batch(loader: DataLoader, iterator):
    """Return the next batch, restarting the iterator when the loader ends."""
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)

    return batch, iterator


def get_mixed_v2_probability(epoch_idx: int) -> float:
    """
    Decide how often an optimisation step should use a clean-V2 batch.

    If MIXED_USE_CURRICULUM=True, the probability is read from
    MIXED_CURRICULUM, e.g. ((0, 0.25), (3, 0.50), (8, 0.75)).
    Epoch indices are zero-based: epoch 0 is the first epoch.
    """
    if not bool(getattr(cfg, "MIXED_USE_CURRICULUM", True)):
        return float(getattr(cfg, "MIXED_V2_PROB", 0.5))

    schedule = getattr(
        cfg,
        "MIXED_CURRICULUM",
        ((0, 0.25), (3, 0.50), (8, 0.75)),
    )

    p_v2 = float(schedule[0][1])

    for start_epoch, value in schedule:
        if epoch_idx >= int(start_epoch):
            p_v2 = float(value)

    return min(max(p_v2, 0.0), 1.0)


def train_mixed_one_epoch(
    model: nn.Module,
    v2_loader: DataLoader,
    synth_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    lambda_l1: float = cfg.LAMBDA_L1,
    grad_clip: float = cfg.GRAD_CLIP,
) -> dict:
    """
    One mixed-domain epoch.

    Each optimisation step uses either:
    - one clean V2 batch, with per-triplet target gain calibration already
      applied by the dataset wrapper, or
    - one synthetic replay batch, using the original synthetic targets.

    The domain is sampled according to a curriculum probability. Synthetic
    batches are kept with a smaller gradient weight so they stabilise the model
    without dominating the real-domain adaptation.
    """
    model.train()

    p_v2 = get_mixed_v2_probability(epoch_idx)
    w_v2 = float(getattr(cfg, "MIXED_WEIGHT_V2", 1.0))
    w_synth = float(getattr(cfg, "MIXED_WEIGHT_SYNTH", 0.3))

    configured_steps = getattr(cfg, "MIXED_STEPS_PER_EPOCH", None)
    if configured_steps is None:
        n_steps = max(1, len(v2_loader))
    else:
        n_steps = int(configured_steps)

    v2_iter = iter(v2_loader)
    synth_iter = iter(synth_loader)

    accum = {}
    domain_counts = {"v2": 0, "synthetic": 0}

    progress = tqdm(
        range(n_steps),
        desc="Mixed training",
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for _ in progress:
        use_v2 = random.random() < p_v2

        if use_v2:
            batch, v2_iter = _next_cycled_batch(v2_loader, v2_iter)
            domain = "v2"
            domain_weight = w_v2
        else:
            batch, synth_iter = _next_cycled_batch(synth_loader, synth_iter)
            domain = "synthetic"
            domain_weight = w_synth

        M, H_ref, L_ref = batch
        M = M.to(device, non_blocking=True)
        H_ref = H_ref.to(device, non_blocking=True)
        L_ref = L_ref.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        H_hat, L_hat = model(M)

        loss_unweighted, metrics = combined_loss(
            H_hat=H_hat,
            H_ref=H_ref,
            L_hat=L_hat,
            L_ref=L_ref,
            M=M,
            lambda_l1=lambda_l1,
        )

        loss = domain_weight * loss_unweighted
        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        optimizer.step()

        domain_counts[domain] += 1

        # Global weighted metrics: these correspond to what actually drives
        # the optimiser.
        for key, value in metrics.items():
            accum[key] = accum.get(key, 0.0) + value

        accum["loss_total_weighted_by_domain"] = (
            accum.get("loss_total_weighted_by_domain", 0.0)
            + float(loss.item())
        )
        accum[f"{domain}_loss_total_unweighted"] = (
            accum.get(f"{domain}_loss_total_unweighted", 0.0)
            + float(loss_unweighted.item())
        )

        progress.set_postfix({
            "domain": domain,
            "pV2": f"{p_v2:.2f}",
            "w": f"{domain_weight:.2f}",
            "loss": f"{loss.item():.4f}",
            "H": f"{-metrics['loss_sisdr_h']:.2f}dB",
            "L": f"{-metrics['loss_sisdr_l']:.2f}dB",
            "mixx": f"{metrics['weighted_loss_mix']:.5f}",
            "mixpol": f"{metrics['weighted_loss_mix_polarity']:.5f}",
        })

    result = {
        key: value / n_steps
        for key, value in accum.items()
    }

    result["mixed_p_v2"] = p_v2
    result["mixed_weight_v2"] = w_v2
    result["mixed_weight_synth"] = w_synth
    result["mixed_v2_batches"] = domain_counts["v2"]
    result["mixed_synth_batches"] = domain_counts["synthetic"]
    result["mixed_steps_per_epoch"] = n_steps

    for domain, count in domain_counts.items():
        key = f"{domain}_loss_total_unweighted"
        if count > 0 and key in accum:
            result[key] = accum[key] / count

    return result


###################
#Model architecture
###################

#Build encoder, separator and decoder architecture
def build_model(args, device: torch.device) -> nn.Module:

    from encoder import CardiopulmonaryEncoder
    from separator import CardiopulmonarySeparator
    from decoder import CardiopulmonaryDecoder

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

            Z = self.encoder(M) #get Z from the mixture M
            Z_hs, Z_ls, _ = self.separator(Z) #obtaine the 2 masks
            H_hat = self.decoder(Z_hs, normalise=False) #compute heart
            L_hat = self.decoder(Z_ls, normalise=False) #compute lung
            return H_hat, L_hat

    model = CardiopulmonaryNet(T=args.seg_samples).to(device)

    total = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"Total trainable parameters: {total:,}")
    return model


#########################
#Stage 1 — Pre-training
#########################

def run_pretrain(
    args,
    model: nn.Module,
    device: torch.device,
) -> str:

    print("\n" + "=" * 60)
    print("STAGE 1 — Pre-training on pseudo-mixtures")
    print("=" * 60)

    dataset = TripletDataset(
        args.pseudo_dir,
        sr=args.sr,
        segment_samples=args.seg_samples,
    )

    print(f"Pseudo-mix dataset: {len(dataset)} segments")

    n_val = max(1, int(0.10 * len(dataset)))
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.pretrain_lr,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt = os.path.join(args.ckpt_dir, "pretrain_best.pt")
    patience_count = 0

    for epoch in range(args.pretrain_epochs):
        train_m = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_l1=args.lambda_l1,
        )

        val_m = validate(
            model,
            val_loader,
            device,
            lambda_l1=args.lambda_l1,
        )

        scheduler.step(val_m["loss_total"])

        print(
            f"[Pre-train] Epoch {epoch + 1:3d}/{args.pretrain_epochs} | "
            f"train={train_m['loss_total']:.4f} | "
            f"val={val_m['loss_total']:.4f} | "
            f"SI-SDR H={val_m['si_sdr_h_dB']:+.2f} dB  "
            f"L={val_m['si_sdr_l_dB']:+.2f} dB | "
            f"envL={val_m.get('loss_env_l', 0.0):.4f} | "
            f"envCorrL={val_m.get('env_corr_l', 0.0):+.3f} | "
            f"envLeakLH={val_m.get('env_leak_lh', 0.0):+.3f} | "
            f"mixLoss={val_m.get('loss_mix', 0.0):.4f} "
            f"(x{val_m.get('lambda_mix', 0.0):.3f}) | "
            f"mixNMSE={val_m.get('mix_nmse_db', float('nan')):+.2f}dB | "
            f"gammaNeg={100.0 * val_m.get('mix_gamma_negative_rate', 0.0):.1f}% | "
            f"aH={val_m.get('alpha_h_mean', float('nan')):+.2f} "
            f"aL={val_m.get('alpha_l_mean', float('nan')):+.2f} | "
            f"negH={100.0 * val_m.get('alpha_h_negative_rate', 0.0):.1f}% "
            f"negL={100.0 * val_m.get('alpha_l_negative_rate', 0.0):.1f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_m["loss_total"] < best_val_loss:
            best_val_loss = val_m["loss_total"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "stage": "pretrain",
                    "config": {
                        "N_FILTERS": cfg.N_FILTERS,
                        "N_LATENT": cfg.N_LATENT,
                        "STRIDE": cfg.STRIDE,
                        "KERNEL_SIZE": cfg.KERNEL_SIZE,
                    },
                },
                best_ckpt,
            )

            patience_count = 0

        else:
            patience_count += 1

            if patience_count >= args.pretrain_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {args.pretrain_patience} epochs)"
                )
                break

    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved: {best_ckpt}")

    return best_ckpt




def set_trainable_scope(model: nn.Module, scope: str) -> None:
    """
    Configure which part of the model is updated during fine-tuning.

    Supported scopes:
    - full:           encoder + separator + decoder
    - separator_only: separator only, encoder/decoder frozen
    """
    scope = scope.lower().strip()

    if scope == "full":
        for p in model.parameters():
            p.requires_grad = True

    elif scope == "separator_only":
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = False
        for p in model.separator.parameters():
            p.requires_grad = True

    else:
        raise ValueError(
            "Unknown fine-tuning scope. Use 'full' or 'separator_only'. "
            f"Got: {scope}"
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_trainable_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    total_epochs: int,
    warmup_epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters selected for fine-tuning.")

    optimizer = torch.optim.Adam(
        params,
        lr=lr,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=1e-6,
    )

    return optimizer, scheduler


def load_pretrained_weights(model: nn.Module, ckpt_path: str, device: torch.device) -> dict:
    """
    Load model weights only. The optimizer state is intentionally ignored.
    This keeps fine-tuning independent from the synthetic pre-training optimizer.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded pretrained model weights from: {ckpt_path}")
    print("Optimizer state from the checkpoint was NOT loaded.")
    return ckpt

########################################################
#Stage 2 — Fine-tuning from pretrained checkpoint
########################################################

def run_finetune(
    args,
    model: nn.Module,
    device: torch.device,
    pretrain_ckpt: str,
) -> dict:

    print("\n" + "=" * 60)
    print("STAGE 2 — Fine-tuning on real triplets (k-fold CV)")
    print("=" * 60)

    dataset = TripletDataset(
        args.supervised_dir,
        sr=args.sr,
        segment_samples=args.seg_samples,
    )

    mixed_training = bool(getattr(cfg, "MIXED_TRAINING", False))
    synth_dataset = None

    if mixed_training:
        synth_dir = getattr(cfg, "SYNTH_SUPERVISED_DIR", None)

        if synth_dir is None or str(synth_dir).strip() == "":
            raise ValueError(
                "MIXED_TRAINING=True requires SYNTH_SUPERVISED_DIR in model_config.py"
            )

        synth_dataset = TripletDataset(
            synth_dir,
            sr=args.sr,
            segment_samples=args.seg_samples,
        )

        print()
        print("=" * 76)
        print("MIXED TRAINING ENABLED: clean V2 adaptation + synthetic replay")
        print("=" * 76)
        print(f"V2 supervised dir     : {args.supervised_dir}")
        print(f"Synthetic replay dir  : {synth_dir}")
        print(f"Synthetic replay items: {len(synth_dataset)}")
        print(
            "Curriculum            : "
            f"{getattr(cfg, 'MIXED_CURRICULUM', None)} | "
            f"use_curriculum={getattr(cfg, 'MIXED_USE_CURRICULUM', True)}"
        )
        print(
            "Domain weights        : "
            f"V2={getattr(cfg, 'MIXED_WEIGHT_V2', 1.0)} | "
            f"synthetic={getattr(cfg, 'MIXED_WEIGHT_SYNTH', 0.3)}"
        )
        print(
            "Mixture losses        : "
            f"lambda_mix={getattr(cfg, 'LAMBDA_MIX', 0.0)} | "
            f"lambda_mix_polarity={getattr(cfg, 'LAMBDA_MIX_POLARITY', 0.0)}"
        )

    base_ids_all = get_base_triplet_ids(args.supervised_dir)
    max_base_triplets = getattr(cfg, "SMOKE_MAX_BASE_TRIPLETS", None)

    if max_base_triplets is not None and max_base_triplets < len(base_ids_all):
        rng = np.random.default_rng(args.seed)
        selected_idx = rng.choice(
            len(base_ids_all),
            size=max_base_triplets,
            replace=False,
        )
        base_ids = [base_ids_all[i] for i in sorted(selected_idx)]
        print(
            f"[SMOKE MODE] Using {len(base_ids)} base triplets "
            f"out of {len(base_ids_all)}"
        )
    else:
        base_ids = base_ids_all

    n_base = len(base_ids)

    print(f"Real triplets (base): {n_base} | Total segments: {len(dataset)}")
    print(
        f"Cross-validation: {args.n_folds} folds "
        f"(~{n_base // args.n_folds} test triplets per fold)"
    )

    kf = KFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.seed,
    )
    excluded_window_keys = set()

    if getattr(cfg, "FILTER_LOCAL_SNR", False):
        excluded_window_keys = load_excluded_window_keys(
            analysis_csv=cfg.LOCAL_SNR_ANALYSIS_CSV,
            threshold_db=cfg.LOCAL_SNR_THRESHOLD_DB,
        )

    fold_results = []
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for fold_idx, (train_base_idx, val_base_idx) in enumerate(
        kf.split(range(n_base))
    ):
        fold_no = fold_idx + 1
        only_fold = getattr(cfg, "ONLY_FOLD", None)

        if only_fold is not None and fold_no != only_fold:
            print(f"Skipping Fold {fold_no}/{args.n_folds}")
            continue

        print(
            f"\n── Fold {fold_no}/{args.n_folds} "
            f"({len(val_base_idx)} test base triplets) ──"
        )

        train_bases = {base_ids[i] for i in train_base_idx}
        val_bases = {base_ids[i] for i in val_base_idx}

        train_seg_idx = [
            i for i, name in enumerate(dataset.names)
            if name.split("_s")[0] in train_bases
        ]

        val_seg_idx = [
            i for i, name in enumerate(dataset.names)
            if name.split("_s")[0] in val_bases
        ]

        train_segments_before_filter = len(train_seg_idx)
        val_segments_before_filter = len(val_seg_idx)

        removed_train_idx = []
        removed_val_idx = []

        if excluded_window_keys:
            train_seg_idx, removed_train_idx = filter_dataset_indices(
                dataset_names=dataset.names,
                indices=train_seg_idx,
                excluded_keys=excluded_window_keys,
            )

            val_seg_idx, removed_val_idx = filter_dataset_indices(
                dataset_names=dataset.names,
                indices=val_seg_idx,
                excluded_keys=excluded_window_keys,
            )

        removed_train_windows = count_unique_windows(
            dataset.names,
            removed_train_idx,
        )

        removed_val_windows = count_unique_windows(
            dataset.names,
            removed_val_idx,
        )

        print(
            f"Train segments: {train_segments_before_filter} -> "
            f"{len(train_seg_idx)} | "
            f"removed items={len(removed_train_idx)} | "
            f"removed original windows={removed_train_windows}"
        )

        print(
            f"Val segments  : {val_segments_before_filter} -> "
            f"{len(val_seg_idx)} | "
            f"removed items={len(removed_val_idx)} | "
            f"removed original windows={removed_val_windows}"
        )

        if len(train_seg_idx) == 0:
            raise RuntimeError(
                f"Fold {fold_no}: no training segments left after filtering."
            )

        if len(val_seg_idx) == 0:
            raise RuntimeError(
                f"Fold {fold_no}: no validation segments left after filtering."
            )

        fold_dataset, gain_info = make_target_dataset_for_fold(
            dataset=dataset,
            train_indices=train_seg_idx,
        )
        print_target_gain_info(fold_no, gain_info)

        train_loader = DataLoader(
            Subset(fold_dataset, train_seg_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        synth_loader = None

        if mixed_training:
            synth_indices = list(range(len(synth_dataset)))
            max_synth_segments = getattr(cfg, "MIXED_SYNTH_MAX_SEGMENTS", None)

            if (
                max_synth_segments is not None
                and int(max_synth_segments) > 0
                and int(max_synth_segments) < len(synth_indices)
            ):
                rng = np.random.default_rng(args.seed + fold_no)
                selected = rng.choice(
                    len(synth_indices),
                    size=int(max_synth_segments),
                    replace=False,
                )
                synth_indices = [synth_indices[i] for i in sorted(selected)]

            synth_loader = DataLoader(
                Subset(synth_dataset, synth_indices),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            print(
                f"Fold {fold_no} synthetic replay segments: "
                f"{len(synth_indices)} | batches={len(synth_loader)}"
            )

        val_loader = DataLoader(
            Subset(fold_dataset, val_seg_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        load_pretrained_weights(model, pretrain_ckpt, device)

        finetune_mode = str(getattr(cfg, "FINETUNE_MODE", "full")).lower().strip()
        if finetune_mode == "two_stage":
            current_scope = "separator_only"
        elif finetune_mode in {"full", "separator_only"}:
            current_scope = finetune_mode
        else:
            raise ValueError(
                "FINETUNE_MODE must be 'full', 'separator_only', or 'two_stage'. "
                f"Got: {finetune_mode}"
            )

        set_trainable_scope(model, current_scope)

        current_lr = (
            float(getattr(cfg, "SEPARATOR_FINETUNE_LR", args.finetune_lr))
            if current_scope == "separator_only"
            else float(getattr(cfg, "FULL_FINETUNE_LR", args.finetune_lr))
        )

        optimizer, scheduler = build_trainable_optimizer_and_scheduler(
            model=model,
            lr=current_lr,
            total_epochs=args.finetune_epochs,
            warmup_epochs=args.warmup_epochs,
        )

        print(
            f"Fine-tuning mode: {finetune_mode} | "
            f"current scope: {current_scope} | "
            f"trainable parameters: {count_trainable_parameters(model):,} | "
            f"lr={current_lr:.2e}"
        )

        best_val_loss = float("inf")
        best_fold_ckpt = os.path.join(
            args.ckpt_dir,
            f"finetune_fold{fold_no}_best.pt",
        )

        patience_count = 0
        best_fold_metrics = {}

        for epoch in range(args.finetune_epochs):
            if (
                finetune_mode == "two_stage"
                and current_scope == "separator_only"
                and epoch == int(getattr(cfg, "TWO_STAGE_SEPARATOR_EPOCHS", 5))
            ):
                current_scope = "full"
                set_trainable_scope(model, current_scope)
                current_lr = float(getattr(cfg, "FULL_FINETUNE_LR", args.finetune_lr))
                optimizer, scheduler = build_trainable_optimizer_and_scheduler(
                    model=model,
                    lr=current_lr,
                    total_epochs=max(1, args.finetune_epochs - epoch),
                    warmup_epochs=0,
                )
                print(
                    f"Two-stage switch at epoch {epoch + 1}: "
                    f"scope={current_scope} | "
                    f"trainable parameters={count_trainable_parameters(model):,} | "
                    f"lr={current_lr:.2e}"
                )

            if epoch < args.warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = 1e-6 + (current_lr - 1e-6) * (
                        epoch / max(1, args.warmup_epochs)
                    )
            elif epoch == args.warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            if mixed_training:
                train_m = train_mixed_one_epoch(
                    model=model,
                    v2_loader=train_loader,
                    synth_loader=synth_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch_idx=epoch,
                    lambda_l1=args.lambda_l1,
                )
            else:
                train_m = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    lambda_l1=args.lambda_l1,
                )

            val_m = validate(
                model,
                val_loader,
                device,
                lambda_l1=args.lambda_l1,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            mixed_log = ""
            if mixed_training:
                mixed_log = (
                    f" | pV2={train_m.get('mixed_p_v2', float('nan')):.2f}"
                    f" | bV2={int(train_m.get('mixed_v2_batches', 0))}"
                    f" | bSyn={int(train_m.get('mixed_synth_batches', 0))}"
                    f" | trainWeighted={train_m.get('loss_total_weighted_by_domain', float('nan')):.4f}"
                )

            print(
                f"Epoch {epoch + 1:3d}/{args.finetune_epochs} | "
                f"scope={current_scope} | "
                f"train={train_m['loss_total']:.4f} | "
                f"val={val_m['loss_total']:.4f} | "
                f"SI-SDR H={val_m['si_sdr_h_dB']:+.2f} dB  "
                f"L={val_m['si_sdr_l_dB']:+.2f} dB | "
                f"L1={val_m['loss_l1']:.5f} "
                f"(weighted={val_m['weighted_loss_l1']:.5f}) | "
                f"RMS={val_m['loss_rms']:.6f} "
                f"(weighted={val_m['weighted_loss_rms']:.6f}) | "
                f"POL={val_m['loss_polarity']:.5f} "
                f"(weighted={val_m['weighted_loss_polarity']:.5f}) | "
                f"MIX={val_m.get('loss_mix', 0.0):.5f} "
                f"(weighted={val_m.get('weighted_loss_mix', 0.0):.5f}) | "
                f"MIXPOL={val_m.get('loss_mix_polarity', 0.0):.5f} "
                f"(weighted={val_m.get('weighted_loss_mix_polarity', 0.0):.5f}) | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                f"{mixed_log}"
            )

            if epoch >= args.warmup_epochs:
                scheduler.step()

            if val_m["loss_total"] < best_val_loss:
                best_val_loss = val_m["loss_total"]
                best_fold_metrics = val_m.copy()

                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold_idx,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_metrics": val_m,
                        "stage": "finetune",
                        "config": {
                            "N_FILTERS": cfg.N_FILTERS,
                            "N_LATENT": cfg.N_LATENT,
                            "ENCODER_R": cfg.ENCODER_R,
                            "DECODER_R": cfg.DECODER_R,
                            "SEP_N_STACKS": cfg.SEP_N_STACKS,
                            "SEP_BLOCKS_PER_STACK": cfg.SEP_BLOCKS_PER_STACK,
                            "SEP_TCN_BOTTLENECK": cfg.SEP_TCN_BOTTLENECK,
                            "ATTN_WINDOW": cfg.ATTN_WINDOW,
                            "MASK_SCALE": cfg.MASK_SCALE,
                            "PRETRAIN_CKPT": pretrain_ckpt,
                            "FINETUNE_MODE": finetune_mode,
                            "CURRENT_SCOPE_AT_SAVE": current_scope,
                            "TWO_STAGE_SEPARATOR_EPOCHS": getattr(cfg, "TWO_STAGE_SEPARATOR_EPOCHS", None),
                            "FINETUNE_LR": cfg.FINETUNE_LR,
                            "SEPARATOR_FINETUNE_LR": getattr(cfg, "SEPARATOR_FINETUNE_LR", cfg.FINETUNE_LR),
                            "FULL_FINETUNE_LR": getattr(cfg, "FULL_FINETUNE_LR", cfg.FINETUNE_LR),
                            "LOSS_OBJECTIVE": (
                                f"SI-SDR + {cfg.LAMBDA_L1} * SmoothL1 "
                                f"+ {getattr(cfg, 'LAMBDA_RMS', 0.0)} * RMS "
                                f"+ {getattr(cfg, 'LAMBDA_POLARITY', 0.0)} * NegativePolarityPenalty "
                                f"+ {getattr(cfg, 'LAMBDA_MIX', 0.0)} * MixtureConsistency "
                                f"+ {getattr(cfg, 'LAMBDA_MIX_POLARITY', 0.0)} * MixturePolarity"
                            ),
                            "LOSS_WEIGHT_H": getattr(cfg, "LOSS_WEIGHT_H", 1.0),
                            "LOSS_WEIGHT_L": getattr(cfg, "LOSS_WEIGHT_L", 1.0),
                            "LAMBDA_L1": cfg.LAMBDA_L1,
                            "LAMBDA_RMS": getattr(cfg, "LAMBDA_RMS", 0.0),
                            "LAMBDA_POLARITY": getattr(cfg, "LAMBDA_POLARITY", 0.0),
                            "DECODER_USE_INPUT_GLN": getattr(cfg, "DECODER_USE_INPUT_GLN", True),
                            "TARGET_GAIN_INFO": gain_info,
                            "FILTER_LOCAL_SNR": getattr(cfg, "FILTER_LOCAL_SNR", False),
                            "LOCAL_SNR_THRESHOLD_DB": getattr(cfg, "LOCAL_SNR_THRESHOLD_DB", None),
                            "LOCAL_SNR_ANALYSIS_CSV": getattr(cfg, "LOCAL_SNR_ANALYSIS_CSV", None),

                            "MIXED_TRAINING": getattr(cfg, "MIXED_TRAINING", False),
                            "SYNTH_SUPERVISED_DIR": getattr(cfg, "SYNTH_SUPERVISED_DIR", None),
                            "MIXED_USE_CURRICULUM": getattr(cfg, "MIXED_USE_CURRICULUM", None),
                            "MIXED_CURRICULUM": list(getattr(cfg, "MIXED_CURRICULUM", ())),
                            "MIXED_V2_PROB": getattr(cfg, "MIXED_V2_PROB", None),
                            "MIXED_WEIGHT_V2": getattr(cfg, "MIXED_WEIGHT_V2", None),
                            "MIXED_WEIGHT_SYNTH": getattr(cfg, "MIXED_WEIGHT_SYNTH", None),
                            "MIXED_STEPS_PER_EPOCH": getattr(cfg, "MIXED_STEPS_PER_EPOCH", None),
                            "MIXED_SYNTH_MAX_SEGMENTS": getattr(cfg, "MIXED_SYNTH_MAX_SEGMENTS", None),
                            "LAMBDA_MIX": getattr(cfg, "LAMBDA_MIX", 0.0),
                            "LAMBDA_MIX_POLARITY": getattr(cfg, "LAMBDA_MIX_POLARITY", 0.0),

                            "USE_BASE_TRIPLET_ALLOWLIST": getattr(cfg, "USE_BASE_TRIPLET_ALLOWLIST", False),
                            "BASE_TRIPLET_ALLOWLIST": list(getattr(cfg, "BASE_TRIPLET_ALLOWLIST", ())),
                            "BASE_TRIPLET_DENYLIST": list(getattr(cfg, "BASE_TRIPLET_DENYLIST", ())),
                            "N_FOLDS": cfg.N_FOLDS,
                            "SPLIT_SEED": cfg.SEED,
                            "MODEL_SEED": getattr(cfg, "MODEL_SEED", cfg.SEED),
                            "TRAIN_SEGMENTS_BEFORE_FILTER": train_segments_before_filter,
                            "TRAIN_SEGMENTS_AFTER_FILTER": len(train_seg_idx),
                            "VAL_SEGMENTS_BEFORE_FILTER": val_segments_before_filter,
                            "VAL_SEGMENTS_AFTER_FILTER": len(val_seg_idx),
                            "REMOVED_TRAIN_ITEMS": len(removed_train_idx),
                            "REMOVED_VAL_ITEMS": len(removed_val_idx),
                            "REMOVED_TRAIN_ORIGINAL_WINDOWS": removed_train_windows,
                            "REMOVED_VAL_ORIGINAL_WINDOWS": removed_val_windows,
                        },
                    },
                    best_fold_ckpt,
                )

                patience_count = 0

            else:
                patience_count += 1

                if patience_count >= args.finetune_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        best_fold_metrics["fold_no"] = fold_no
        best_fold_metrics["target_gain_mode"] = gain_info.get("target_gain_mode")
        best_fold_metrics["target_gain_alpha_h"] = gain_info.get("alpha_h")
        best_fold_metrics["target_gain_alpha_l"] = gain_info.get("alpha_l")
        best_fold_metrics["target_gain_fit_nmse_db"] = gain_info.get("fit_nmse_db")
        best_fold_metrics["target_gain_fit_corr"] = gain_info.get("fit_corr")
        fold_results.append(best_fold_metrics)

        print(
            f"Fold {fold_no} best — "
            f"SI-SDR H: {best_fold_metrics.get('si_sdr_h_dB', float('nan')):+.2f} dB | "
            f"SI-SDR L: {best_fold_metrics.get('si_sdr_l_dB', float('nan')):+.2f} dB"
        )

    return summarize_cv_results(
        fold_results,
        args.ckpt_dir,
        title="Cross-validation results",
    )




# ============================================================
# V1 re-synthesis training-only injection utilities
# ============================================================

def _normalise_six_digit_base_id(value) -> str:
    """Normalize a segment/base identifier to the 6-digit V1/Torabi style."""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""

    stem = Path(text).stem
    if stem.startswith(("M_", "H_", "L_")):
        stem = stem[2:]

    stem = stem.split("_s")[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits.zfill(6) if digits else stem


def _default_v1_allowed_csv() -> str:
    return os.path.join(
        getattr(cfg, "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])),
        "outputs",
        "v1_validation_similarity_audit",
        "v1_injection_allowed_base_ids_by_fold.csv",
    )


def load_v1_allowed_base_ids_for_fold(fold_no: int) -> set[str]:
    """
    Load the V1 base-triplet allowlist produced by
    audit_v1_injection_vs_validation_sources.py.

    The allowlist is fold-specific because a V1 source may be safe for one fold
    but too similar to the validation sources of another fold.
    """
    allowed_csv = getattr(
        cfg,
        "V1_INJECTION_ALLOWED_BASE_IDS_CSV",
        _default_v1_allowed_csv(),
    )

    if allowed_csv is None or str(allowed_csv).strip() == "":
        raise RuntimeError(
            "USE_V1_RESYNTH_TRAIN_INJECTION=True but "
            "V1_INJECTION_ALLOWED_BASE_IDS_CSV is empty."
        )

    if not os.path.exists(allowed_csv):
        raise RuntimeError(
            "V1 injection allowlist not found:\n"
            f"  {allowed_csv}\n"
            "Run audit_v1_injection_vs_validation_sources.py before training."
        )

    df = pd.read_csv(allowed_csv)
    required = {"fold_no", "base_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"V1 injection allowlist is missing columns: {sorted(missing)}"
        )

    df = df[df["fold_no"].astype(int) == int(fold_no)].copy()
    if "allowed_for_training_injection" in df.columns:
        df = df[df["allowed_for_training_injection"].astype(bool)]

    allowed = {
        _normalise_six_digit_base_id(v)
        for v in df["base_id"].tolist()
        if str(v).strip()
    }

    if len(allowed) == 0:
        raise RuntimeError(
            f"Fold {fold_no}: no allowed V1 base triplets found in {allowed_csv}."
        )

    return allowed


def build_train_dataset_with_v1_injection(
    train_subset: Dataset,
    fold_no: int,
    seed: int,
) -> tuple[Dataset, dict]:
    """
    Append V1 re-synth samples to the training set only.

    Validation is intentionally left untouched. This prevents V1 from becoming
    part of the source-disjoint validation benchmark.
    """
    if not bool(getattr(cfg, "USE_V1_RESYNTH_TRAIN_INJECTION", False)):
        return train_subset, {
            "enabled": False,
            "segments": 0,
            "base_triplets": 0,
        }

    v1_dir = getattr(
        cfg,
        "V1_RESYNTH_DIR",
        os.path.join(
            getattr(cfg, "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])),
            "dataset",
            "processed",
            "v1_resynth_300_hop05",
        ),
    )

    v1_dataset = TripletDataset(
        v1_dir,
        sr=cfg.SR,
        segment_samples=cfg.SEG_SAMPLES,
    )

    allowed_bases = load_v1_allowed_base_ids_for_fold(fold_no)

    v1_indices = [
        i for i, name in enumerate(v1_dataset.names)
        if _normalise_six_digit_base_id(name.split("_s")[0]) in allowed_bases
    ]

    if len(v1_indices) == 0:
        raise RuntimeError(
            f"Fold {fold_no}: V1 injection is enabled, but no V1 segments matched "
            "the allowed base IDs. Check V1_RESYNTH_DIR and the audit CSV."
        )

    max_segments = getattr(cfg, "V1_INJECTION_MAX_SEGMENTS", None)
    if max_segments is not None and int(max_segments) > 0 and int(max_segments) < len(v1_indices):
        rng = np.random.default_rng(int(seed) + int(fold_no) + 91001)
        selected = rng.choice(
            len(v1_indices),
            size=int(max_segments),
            replace=False,
        )
        v1_indices = [v1_indices[i] for i in sorted(selected)]

    v1_subset = Subset(v1_dataset, v1_indices)
    merged = ConcatDataset([train_subset, v1_subset])

    info = {
        "enabled": True,
        "v1_dir": str(v1_dir),
        "segments": int(len(v1_indices)),
        "base_triplets": int(len(allowed_bases)),
        "max_segments": max_segments,
    }

    print()
    print("=" * 76)
    print("V1 RE-SYNTH TRAINING-ONLY INJECTION ENABLED")
    print("=" * 76)
    print(f"Fold {fold_no} V1 dir                 : {v1_dir}")
    print(f"Fold {fold_no} allowed V1 base triplets: {info['base_triplets']}")
    print(f"Fold {fold_no} injected V1 segments    : {info['segments']}")
    print("Validation remains unchanged: V1 is NOT added to val_loader.")

    return merged, info

#Train from scratch on real supervised triplets using k-fold CV.
# from BOTH training and validation splits.
def run_finetune_from_scratch(
    args,
    device: torch.device,
) -> dict:

    print("\n" + "=" * 60)
    print("SUPERVISED-ONLY — Training from scratch on supervised triplets")
    print("=" * 60)

    dataset = TripletDataset(
        args.supervised_dir,
        sr=args.sr,
        segment_samples=args.seg_samples,
    )

    base_ids_all = get_base_triplet_ids(args.supervised_dir)
    max_base_triplets = getattr(cfg, "SMOKE_MAX_BASE_TRIPLETS", None)

    # --------------------------------------------------------------------------
    # Source-disjoint split mode.
    #
    # This file is intended for the Torabi synthetic diagnostic where the split is
    # defined at the original HS/LS source level, not at the mixture level.
    # Validation contains only mixtures built from HS and LS sources that are
    # completely absent from training. Combinations with only one unseen source
    # are marked as "unused" in the CSV and are not loaded.
    # --------------------------------------------------------------------------
    use_source_disjoint = bool(getattr(cfg, "USE_SOURCE_DISJOINT_SPLIT", True))

    if use_source_disjoint:
        if max_base_triplets is not None:
            print(
                "[SOURCE-DISJOINT MODE] SMOKE_MAX_BASE_TRIPLETS is ignored. "
                "The split CSV decides train/validation/unused base triplets."
            )

        base_ids = base_ids_all
    else:
        if max_base_triplets is not None and max_base_triplets < len(base_ids_all):
            rng = np.random.default_rng(args.seed)

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
                f"[SMOKE MODE] Using {len(base_ids)} base triplets "
                f"out of {len(base_ids_all)}"
            )
        else:
            base_ids = base_ids_all

    n_base = len(base_ids)

    print(f"Supervised triplets (base): {n_base} | Total dataset items: {len(dataset)}")

    if use_source_disjoint:
        print(
            f"Cross-validation: {args.n_folds} source-disjoint folds "
            "from SOURCE_DISJOINT_SPLIT_CSV"
        )
    else:
        print(
            f"Cross-validation: {args.n_folds} folds "
            f"(~{n_base // args.n_folds} validation triplets per fold)"
        )

    source_split_df = None

    if use_source_disjoint:
        source_split_csv = getattr(
            cfg,
            "SOURCE_DISJOINT_SPLIT_CSV",
            os.path.join(
                getattr(cfg, "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])),
                "outputs",
                "source_disjoint_splits_torabi.csv",
            ),
        )

        if not os.path.exists(source_split_csv):
            raise RuntimeError(
                "Source-disjoint split file not found: "
                f"{source_split_csv}\n"
                "Run make_source_disjoint_folds.py first."
            )

        source_split_df = pd.read_csv(source_split_csv)

        required_cols = {
            "fold_no",
            "base_id",
            "split",
            "hs_source_id",
            "ls_source_id",
        }
        missing_cols = required_cols - set(source_split_df.columns)
        if missing_cols:
            raise RuntimeError(
                f"Missing columns in source-disjoint split CSV: {sorted(missing_cols)}"
            )

        source_split_df = source_split_df.copy()
        source_split_df["base_id"] = (
            source_split_df["base_id"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(6)
        )
        source_split_df["split"] = source_split_df["split"].astype(str).str.lower()

        print()
        print("=" * 76)
        print("SOURCE-DISJOINT SPLIT ENABLED")
        print("=" * 76)
        print(f"Split file: {source_split_csv}")

        available_bases = set(base_ids_all)

        for fold_no in range(1, args.n_folds + 1):
            g = source_split_df[source_split_df["fold_no"] == fold_no]

            if len(g) == 0:
                raise RuntimeError(f"No rows found in split CSV for fold {fold_no}.")

            train_g = g[g["split"] == "train"]
            val_g = g[g["split"] == "val"]
            unused_g = g[g["split"] == "unused"]

            train_h = set(train_g["hs_source_id"].astype(str))
            train_l = set(train_g["ls_source_id"].astype(str))
            val_h = set(val_g["hs_source_id"].astype(str))
            val_l = set(val_g["ls_source_id"].astype(str))

            h_overlap = train_h & val_h
            l_overlap = train_l & val_l

            train_bases_check = set(train_g["base_id"]) & available_bases
            val_bases_check = set(val_g["base_id"]) & available_bases
            unused_bases_check = set(unused_g["base_id"]) & available_bases

            print(
                f"Fold {fold_no}: "
                f"train bases={len(train_bases_check)} | "
                f"val bases={len(val_bases_check)} | "
                f"unused bases={len(unused_bases_check)} | "
                f"HS overlap={len(h_overlap)} | "
                f"LS overlap={len(l_overlap)}"
            )

            if h_overlap or l_overlap:
                raise RuntimeError(f"Source leakage detected in fold {fold_no}.")

            if len(train_bases_check) == 0 or len(val_bases_check) == 0:
                raise RuntimeError(
                    f"Fold {fold_no}: empty train or validation base set after "
                    "intersecting with the supervised dataset."
                )

        kf = None

    else:
        kf = KFold(
            n_splits=args.n_folds,
            shuffle=True,
            random_state=args.seed,
        )

    # --------------------------------------------------------------------------
    # Load local-SNR blacklist once before cross-validation.
    # The blacklist is built on original windows; snr_filter.py also excludes
    # any augmented versions belonging to the same problematic window.
    # --------------------------------------------------------------------------
    excluded_window_keys = set()

    if getattr(cfg, "FILTER_LOCAL_SNR", False):
        excluded_window_keys = load_excluded_window_keys(
            analysis_csv=cfg.LOCAL_SNR_ANALYSIS_CSV,
            threshold_db=cfg.LOCAL_SNR_THRESHOLD_DB,
        )
    else:
        print()

    fold_results = []
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if use_source_disjoint:
        fold_iterator = []
        available_bases = set(base_ids_all)

        for fold_no in range(1, args.n_folds + 1):
            g = source_split_df[source_split_df["fold_no"] == fold_no]

            train_bases_fold = sorted(
                set(g[g["split"] == "train"]["base_id"]) & available_bases
            )
            val_bases_fold = sorted(
                set(g[g["split"] == "val"]["base_id"]) & available_bases
            )

            fold_iterator.append((fold_no - 1, train_bases_fold, val_bases_fold))
    else:
        fold_iterator = []

        for fold_idx, (train_base_idx, val_base_idx) in enumerate(kf.split(range(n_base))):
            train_bases_fold = sorted({base_ids[i] for i in train_base_idx})
            val_bases_fold = sorted({base_ids[i] for i in val_base_idx})
            fold_iterator.append((fold_idx, train_bases_fold, val_bases_fold))

    for fold_idx, train_bases, val_bases in fold_iterator:
        fold_no = fold_idx + 1
        only_fold = getattr(cfg, "ONLY_FOLD", None)

        if only_fold is not None and fold_no != only_fold:
            print(f"Skipping Fold {fold_no}/{args.n_folds}")
            continue

        print(
            f"\n── Fold {fold_no}/{args.n_folds} "
            f"({len(val_bases)} validation base triplets) ──"
        )
        set_seed(args.model_seed, device)
        print(
            f"Fold {fold_no}: model/training seed={args.model_seed} | "
            f"split/subset seed={args.seed} | "
            f"source_disjoint={use_source_disjoint}"
        )
        model = build_model(args, device)

        # Initial split: includes original and augmented versions associated
        # with the base triplets of each fold.
        train_seg_idx = [
            i for i, name in enumerate(dataset.names)
            if name.split("_s")[0] in train_bases
        ]

        val_seg_idx = [
            i for i, name in enumerate(dataset.names)
            if name.split("_s")[0] in val_bases
        ]

        train_segments_before_filter = len(train_seg_idx)
        val_segments_before_filter = len(val_seg_idx)

        removed_train_idx = []
        removed_val_idx = []

        # ----------------------------------------------------------------------
        # CASE A: filter BOTH training and validation.
        # Evaluation is filtered separately in evaluate.py.
        # ----------------------------------------------------------------------
        if excluded_window_keys:
            train_seg_idx, removed_train_idx = filter_dataset_indices(
                dataset_names=dataset.names,
                indices=train_seg_idx,
                excluded_keys=excluded_window_keys,
            )

            val_seg_idx, removed_val_idx = filter_dataset_indices(
                dataset_names=dataset.names,
                indices=val_seg_idx,
                excluded_keys=excluded_window_keys,
            )

        removed_train_windows = count_unique_windows(
            dataset.names,
            removed_train_idx,
        )

        removed_val_windows = count_unique_windows(
            dataset.names,
            removed_val_idx,
        )

        print(
            f"Train segments: {train_segments_before_filter} -> "
            f"{len(train_seg_idx)} | "
            f"removed items={len(removed_train_idx)} | "
            f"removed original windows={removed_train_windows}"
        )

        print(
            f"Val segments  : {val_segments_before_filter} -> "
            f"{len(val_seg_idx)} | "
            f"removed items={len(removed_val_idx)} | "
            f"removed original windows={removed_val_windows}"
        )

        if getattr(cfg, "FILTER_LOCAL_SNR", False):
            if removed_train_windows == 0 and removed_val_windows == 0:
                print(
                    "[WARNING] Filter enabled, but no excluded windows "
                    "occur in this fold split."
                )

        if len(train_seg_idx) == 0:
            raise RuntimeError(
                f"Fold {fold_no}: no training segments left after filtering."
            )

        if len(val_seg_idx) == 0:
            raise RuntimeError(
                f"Fold {fold_no}: no validation segments left after filtering."
            )

        train_dataset = dataset

        if bool(getattr(cfg, "AUGMENT_TRAINING", False)):
            train_dataset = SourceGainJitterDataset(
                base_dataset=dataset,
                gain_jitter_db=getattr(cfg, "AUG_SOURCE_GAIN_JITTER_DB", 3.0),
                prob=getattr(cfg, "AUG_SOURCE_GAIN_PROB", 1.0),
                preserve_peak=getattr(cfg, "AUG_PRESERVE_PEAK", True),
                peak_value=getattr(cfg, "AUG_PEAK_VALUE", 0.95),
            )

            print()
            print(
                "Source gain jitter: "
                f"+/- {getattr(cfg, 'AUG_SOURCE_GAIN_JITTER_DB', 3.0)} dB | "
                f"prob={getattr(cfg, 'AUG_SOURCE_GAIN_PROB', 1.0)} | "
                f"preserve_peak={getattr(cfg, 'AUG_PRESERVE_PEAK', True)}"
            )
        else:
            print()

        train_subset = Subset(train_dataset, train_seg_idx)
        train_loader_dataset = train_subset
        v1_injection_info = {
            "enabled": False,
            "segments": 0,
            "base_triplets": 0,
        }

        if bool(getattr(cfg, "USE_V1_RESYNTH_TRAIN_INJECTION", False)):
            train_loader_dataset, v1_injection_info = build_train_dataset_with_v1_injection(
                train_subset=train_subset,
                fold_no=fold_no,
                seed=args.seed,
            )

        train_loader = DataLoader(
            train_loader_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            Subset(dataset, val_seg_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.finetune_lr,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.finetune_epochs - args.warmup_epochs),
            eta_min=1e-6,
        )

        best_val_loss = float("inf")

        best_fold_ckpt = os.path.join(
            args.ckpt_dir,
            f"scratch_fold{fold_no}_best.pt",
        )

        patience_count = 0
        best_fold_metrics = {}

        for epoch in range(args.finetune_epochs):

            if epoch < args.warmup_epochs:
                lr = 1e-6 + (args.finetune_lr - 1e-6) * (
                    epoch / max(1, args.warmup_epochs)
                )

                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            elif epoch == args.warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.finetune_lr

            train_m = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                lambda_l1=args.lambda_l1,
            )

            val_m = validate(
                model,
                val_loader,
                device,
                lambda_l1=args.lambda_l1,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:3d}/{args.finetune_epochs} | "
                f"train={train_m['loss_total']:.4f} | "
                f"val={val_m['loss_total']:.4f} | "
                f"SI-SDR H={val_m['si_sdr_h_dB']:+.2f} dB  "
                f"L={val_m['si_sdr_l_dB']:+.2f} dB | "
                f"L1={val_m['loss_l1']:.5f} "
                f"(weighted={val_m['weighted_loss_l1']:.5f}) | "
                f"RMS={val_m['loss_rms']:.6f} "
                f"(weighted={val_m['weighted_loss_rms']:.6f}) | "
                f"lr={current_lr:.2e}"
            )

            if epoch >= args.warmup_epochs:
                scheduler.step()

            if val_m["loss_total"] < best_val_loss:
                best_val_loss = val_m["loss_total"]
                best_fold_metrics = val_m.copy()

                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold_idx,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_metrics": val_m,
                        "stage": "supervised_from_scratch",
                        "config": {
                            "N_FILTERS": cfg.N_FILTERS,
                            "N_LATENT": cfg.N_LATENT,
                            "ENCODER_R": cfg.ENCODER_R,
                            "DECODER_R": cfg.DECODER_R,
                            "SEP_N_STACKS": cfg.SEP_N_STACKS,
                            "SEP_BLOCKS_PER_STACK": cfg.SEP_BLOCKS_PER_STACK,
                            "SEP_TCN_BOTTLENECK": cfg.SEP_TCN_BOTTLENECK,
                            "ATTN_WINDOW": cfg.ATTN_WINDOW,
                            "MASK_SCALE": cfg.MASK_SCALE,
                            "LOSS_OBJECTIVE": (
                                f"SI-SDR + {cfg.LAMBDA_L1} * SmoothL1 "
                                f"+ {getattr(cfg, 'LAMBDA_RMS', 0.0)} * RMS "
                                f"+ {getattr(cfg, 'LAMBDA_POLARITY', 0.0)} * NegativePolarityPenalty "
                                f"+ {getattr(cfg, 'LAMBDA_MIX', 0.0)} * MixtureConsistency "
                                f"+ {getattr(cfg, 'LAMBDA_MIX_POLARITY', 0.0)} * MixturePolarity"
                            ),
                            "LOSS_WEIGHT_H": getattr(cfg, "LOSS_WEIGHT_H", 1.0),
                            "LOSS_WEIGHT_L": getattr(cfg, "LOSS_WEIGHT_L", 1.0),
                            "LAMBDA_L1": cfg.LAMBDA_L1,
                            "LAMBDA_RMS": getattr(cfg, "LAMBDA_RMS", 0.0),
                            "LAMBDA_POLARITY": getattr(cfg, "LAMBDA_POLARITY", 0.0),
                            "DECODER_USE_INPUT_GLN": getattr(cfg, "DECODER_USE_INPUT_GLN", True),
                            "FILTER_LOCAL_SNR": getattr(
                                cfg,
                                "FILTER_LOCAL_SNR",
                                False,
                            ),
                            "LOCAL_SNR_THRESHOLD_DB": getattr(
                                cfg,
                                "LOCAL_SNR_THRESHOLD_DB",
                                None,
                            ),
                            "LOCAL_SNR_ANALYSIS_CSV": getattr(
                                cfg,
                                "LOCAL_SNR_ANALYSIS_CSV",
                                None,
                            ),
                            "SMOKE_MAX_BASE_TRIPLETS": getattr(
                                cfg,
                                "SMOKE_MAX_BASE_TRIPLETS",
                                None,
                            ),
                            "USE_SOURCE_DISJOINT_SPLIT": use_source_disjoint,
                            "SOURCE_DISJOINT_SPLIT_CSV": getattr(
                                cfg,
                                "SOURCE_DISJOINT_SPLIT_CSV",
                                None,
                            ),

                            "USE_BASE_TRIPLET_ALLOWLIST": getattr(cfg, "USE_BASE_TRIPLET_ALLOWLIST", False),
                            "BASE_TRIPLET_ALLOWLIST": list(getattr(cfg, "BASE_TRIPLET_ALLOWLIST", ())),
                            "BASE_TRIPLET_DENYLIST": list(getattr(cfg, "BASE_TRIPLET_DENYLIST", ())),
                            "N_FOLDS": cfg.N_FOLDS,
                            "SPLIT_SEED": cfg.SEED,
                            "MODEL_SEED": getattr(cfg, "MODEL_SEED", cfg.SEED),
                            "SEED": cfg.SEED,
                            "TRAIN_SEGMENTS_BEFORE_FILTER": train_segments_before_filter,
                            "TRAIN_SEGMENTS_AFTER_FILTER": len(train_seg_idx),
                            "VAL_SEGMENTS_BEFORE_FILTER": val_segments_before_filter,
                            "VAL_SEGMENTS_AFTER_FILTER": len(val_seg_idx),
                            "USE_V1_RESYNTH_TRAIN_INJECTION": getattr(cfg, "USE_V1_RESYNTH_TRAIN_INJECTION", False),
                            "V1_RESYNTH_DIR": getattr(cfg, "V1_RESYNTH_DIR", None),
                            "V1_INJECTION_ALLOWED_BASE_IDS_CSV": getattr(cfg, "V1_INJECTION_ALLOWED_BASE_IDS_CSV", None),
                            "V1_INJECTION_SEGMENTS": v1_injection_info.get("segments", 0),
                            "V1_INJECTION_BASE_TRIPLETS": v1_injection_info.get("base_triplets", 0),
                            "REMOVED_TRAIN_ITEMS": len(removed_train_idx),
                            "REMOVED_VAL_ITEMS": len(removed_val_idx),
                            "REMOVED_TRAIN_ORIGINAL_WINDOWS": removed_train_windows,
                            "REMOVED_VAL_ORIGINAL_WINDOWS": removed_val_windows,
                        },
                    },
                    best_fold_ckpt,
                )

                patience_count = 0

            else:
                patience_count += 1

                if patience_count >= args.finetune_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        best_fold_metrics["fold_no"] = fold_no
        fold_results.append(best_fold_metrics)

        print(
            f"Fold {fold_no} best — "
            f"SI-SDR H: "
            f"{best_fold_metrics.get('si_sdr_h_dB', float('nan')):+.2f} dB | "
            f"SI-SDR L: "
            f"{best_fold_metrics.get('si_sdr_l_dB', float('nan')):+.2f} dB"
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_title = (
        "Source-disjoint cross-validation results"
        if use_source_disjoint
        else "Cross-validation results"
    )

    return summarize_cv_results(
        fold_results,
        args.ckpt_dir,
        title=summary_title,
    )


# ════════════════════════════════════════════════════════════════════════════
# 8. CV summary
# ════════════════════════════════════════════════════════════════════════════

def summarize_cv_results(
    fold_results: list[dict],
    ckpt_dir: str,
    title: str,
) -> dict:
    """
    Aggregate and save cross-validation results.
    """
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)

    h_scores = [
        r.get("si_sdr_h_dB", float("nan"))
        for r in fold_results
    ]

    l_scores = [
        r.get("si_sdr_l_dB", float("nan"))
        for r in fold_results
    ]
    fold_nos = [
        int(r.get("fold_no", i + 1))
        for i, r in enumerate(fold_results)
    ]
    cv_results = {
        "si_sdr_h_mean": float(np.nanmean(h_scores)),
        "si_sdr_h_std": float(np.nanstd(h_scores)),
        "si_sdr_l_mean": float(np.nanmean(l_scores)),
        "si_sdr_l_std": float(np.nanstd(l_scores)),
        "fold_h_scores": h_scores,
        "fold_l_scores": l_scores,
    }

    print(
        f"Heart SI-SDR : {cv_results['si_sdr_h_mean']:+.2f} ± "
        f"{cv_results['si_sdr_h_std']:.2f} dB"
    )

    print(
        f"Lung  SI-SDR : {cv_results['si_sdr_l_mean']:+.2f} ± "
        f"{cv_results['si_sdr_l_std']:.2f} dB"
    )

    print(
        "Per-fold H   : "
        + str({
            fold_no: f"{score:+.2f}"
            for fold_no, score in zip(fold_nos, h_scores)
        })
    )

    print(
        "Per-fold L   : "
        + str({
            fold_no: f"{score:+.2f}"
            for fold_no, score in zip(fold_nos, l_scores)
        })
    )

    summary_path = os.path.join(ckpt_dir, "cv_results.txt")

    with open(summary_path, "w") as f:
        f.write(title + "\n")
        f.write(
            f"Heart SI-SDR: {cv_results['si_sdr_h_mean']:+.2f} ± "
            f"{cv_results['si_sdr_h_std']:.2f} dB\n"
        )
        f.write(
            f"Lung  SI-SDR: {cv_results['si_sdr_l_mean']:+.2f} ± "
            f"{cv_results['si_sdr_l_std']:.2f} dB\n"
        )

        for fold_no, h, l in zip(fold_nos, h_scores, l_scores):
            f.write(f"Fold {fold_no}: H={h:+.2f} dB  L={l:+.2f} dB\n")

    print(f"\nSummary saved: {summary_path}")

    return cv_results


#################################
#Configuration and main
#################################

class Config:
    pseudo_dir = cfg.PSEUDO_DIR
    supervised_dir = cfg.SUPERVISED_DIR
    pretrain_ckpt = cfg.PRETRAIN_CKPT
    ckpt_dir = cfg.CKPT_DIR
    stage = cfg.STAGE

    sr = cfg.SR
    seg_samples = cfg.SEG_SAMPLES

    batch_size = cfg.BATCH_SIZE
    num_workers = cfg.NUM_WORKERS
    lambda_l1 = cfg.LAMBDA_L1

    pretrain_epochs = cfg.PRETRAIN_EPOCHS
    pretrain_lr = cfg.PRETRAIN_LR
    pretrain_patience = cfg.PRETRAIN_PATIENCE

    finetune_epochs = cfg.FINETUNE_EPOCHS
    finetune_lr = cfg.FINETUNE_LR
    finetune_patience = cfg.FINETUNE_PATIENCE
    warmup_epochs = cfg.WARMUP_EPOCHS

    n_folds = cfg.N_FOLDS
    only_fold = getattr(cfg, "ONLY_FOLD", None)
    seed = cfg.SEED
    model_seed = getattr(cfg, "MODEL_SEED", cfg.SEED)

#Set random seed for reproducibility
def set_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def main():
    args = Config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    eval_post_training = bool(getattr(cfg, "EVALUATE_POST_TRAINING", False))
    print(f"EVALUATE_POST_TRAINING: {'ENABLED' if eval_post_training else 'DISABLED'}")
    set_seed(args.seed, device)

    pretrain_ckpt = args.pretrain_ckpt

    if args.stage == "pretrain":
        model = build_model(args, device)
        run_pretrain(args, model, device)

    elif args.stage == "both":
        model = build_model(args, device)
        pretrain_ckpt = run_pretrain(args, model, device)
        run_finetune(args, model, device, pretrain_ckpt)

    elif args.stage == "finetune":
        if pretrain_ckpt is None:
            raise ValueError("Set pretrain_ckpt or use stage='both'.")

        model = build_model(args, device)
        run_finetune(args, model, device, pretrain_ckpt)

    elif args.stage == "supervised_from_scratch":
        print(f"STARTING EXPERIMENT: {cfg.EXPERIMENT_NAME}")
        run_finetune_from_scratch(args, device)

    else:
        raise ValueError(f"Unknown training stage: {args.stage}")

    print("\nTraining complete.")

    if getattr(cfg, "EVALUATE_POST_TRAINING", False):
        print("\nStarting post-training evaluation...")
        import evaluate_polarity_control
        evaluate_polarity_control.main()

if __name__ == "__main__":
    main()
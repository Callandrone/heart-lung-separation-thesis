"""JASSNet-style waveform decoder.

Each masked latent representation is sent directly through the same decoder
module.  The transposed convolution mirrors the encoder kernel and stride; no
ESD decoder refinement blocks or independent peak normalisation are applied in
training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import model_config as cfg


def peak_normalise(x: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    peak = x.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return x * (target_peak / peak)


class CardiopulmonaryDecoder(nn.Module):
    def __init__(
        self,
        N_latent: int = cfg.N_LATENT,
        N_filters: int = cfg.N_FILTERS,
        kernel_size: int = cfg.KERNEL_SIZE,
        stride: int = cfg.STRIDE,
        R: int = 0,
        dilations: tuple = (),
        dropout_p: float = cfg.DROPOUT_P,
        target_T: int | None = None,
    ) -> None:
        super().__init__()
        if N_filters != N_latent:
            raise ValueError(
                "JASSNet decoder expects the encoder/mask width N to be shared."
            )
        self.target_T = target_T
        self.deconv = nn.ConvTranspose1d(
            N_latent,
            1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, z_src: torch.Tensor, normalise: bool = False) -> torch.Tensor:
        waveform = self.deconv(z_src)
        if self.target_T is not None:
            length = waveform.shape[-1]
            if length > self.target_T:
                waveform = waveform[..., : self.target_T]
            elif length < self.target_T:
                waveform = F.pad(waveform, (0, self.target_T - length))
        if normalise:
            waveform = peak_normalise(waveform)
        return waveform

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    length = (cfg.SEG_SAMPLES - cfg.KERNEL_SIZE) // cfg.STRIDE + 1
    decoder = CardiopulmonaryDecoder(target_T=cfg.SEG_SAMPLES)
    y = decoder(torch.randn(2, cfg.JASSNET_DIM, length))
    assert y.shape == (2, 1, cfg.SEG_SAMPLES), y.shape
    print(f"Decoder output: {tuple(y.shape)} | params={decoder.n_parameters():,}")

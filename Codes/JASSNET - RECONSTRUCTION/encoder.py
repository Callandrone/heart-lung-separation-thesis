"""JASSNet-style waveform encoder.

Zhang et al. describe a single Conv1D encoder followed by ReLU, with kernel
K1 and stride K1/2.  No ESD multi-scale encoder blocks or bottleneck are used.
Shape: [B, 1, T] -> [B, N, S].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import model_config as cfg


class CardiopulmonaryEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        N_filters: int = cfg.N_FILTERS,
        N_latent: int = cfg.N_LATENT,
        kernel_size: int = cfg.KERNEL_SIZE,
        stride: int = cfg.STRIDE,
        R: int = 0,
        dilations: tuple = (),
        dropout_p: float = cfg.DROPOUT_P,
    ) -> None:
        super().__init__()
        if N_filters != N_latent:
            raise ValueError(
                "JASSNet uses one feature width N throughout encoder/masks/decoder; "
                f"received N_filters={N_filters}, N_latent={N_latent}."
            )
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            N_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.activation = nn.ReLU()
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    encoder = CardiopulmonaryEncoder()
    x = torch.randn(2, 1, cfg.SEG_SAMPLES)
    z = encoder(x)
    expected_s = (cfg.SEG_SAMPLES - cfg.KERNEL_SIZE) // cfg.STRIDE + 1
    assert z.shape == (2, cfg.JASSNET_DIM, expected_s), z.shape
    print(f"Encoder output: {tuple(z.shape)} | params={encoder.n_parameters():,}")

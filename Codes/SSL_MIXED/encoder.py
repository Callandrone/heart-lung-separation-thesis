"""
encoder.py
==========
Input: raw waveform x = [B, 1, T] - [batch, 1 bc mono, 2secx4khz=8000]
Output: latent repr. Z = [B, N_latent, L] - [batch, N_Latent, T/stride= 8000/8=1000]
-----------------
1. Conv1D learnable basis
2. GELU activation
3. GlobalLayerNorm
4. Multi-scale depthwise Conv1D block × R
5. Pointwise projection
-----------------------------
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_config as cfg
from normalization import GlobalLayerNorm

##############################
#Learnable Conv1D Basis + GELU
##############################

class LearnableBasisEncoder(nn.Module):
    #Replaces a fixed STFT/filterbank with a learned 1-D convolutional basis

    #constructor with default values from model_config.py
    def __init__(
        self,
        in_channels: int = 1,
        N_filters: int = cfg.N_FILTERS,
        kernel_size: int = cfg.KERNEL_SIZE,
        stride: int = cfg.STRIDE, #stride 8= 2ms at 4 kHz
    ):
        super().__init__() #initialization

        self.kernel_size = kernel_size
        self.stride = stride

        #creating 1D convolution, each filter learns a small temporal form
        self.conv = nn.Conv1d(
            in_channels,
            N_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

        self.activation = nn.GELU() #non linear function that allows small negative values

        # Initialise the learned filterbank weights.
        nn.init.kaiming_uniform_(
            self.conv.weight,
            nonlinearity="linear", #neautral initialization
        )

    # Encode the waveform using the learned basis and GELU activation.
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #manual padding with 4 zeros on each side, ensuring same output length
        pad_total = self.kernel_size - self.stride
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = F.pad(x, (pad_left, pad_right))  #applying the padding
        x = self.conv(x) #apply the convolution, here we have the transformation from waveform to filter (built in pytorch)
        x = self.activation(x) #gelu application

        return x #[B, N_filters, L]

#########################################################
#Multi-Scale Dilated Depthwise Conv Block
##########################################################

class MultiScaleDWConvBlock(nn.Module):
    #it applies three parallel depthwise temporal convolutions with different dilation factors
    #access to multiple local temporal scales

    def __init__(
        self,
        N_filters: int = cfg.N_FILTERS,
        kernel_size: int = 3,
        dilations: tuple = cfg.DILATIONS,
        dropout_p: float = cfg.DROPOUT_P,
    ):
        super().__init__()

        #creating a convolutional list with 3 parallel depthwise convolutions with different dilation factors
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                N_filters,
                N_filters,
                kernel_size=kernel_size,
                dilation=d,
                groups=N_filters, #each channel is convolved separately, this is what makes it depthwise
                padding=d * (kernel_size - 1) // 2, #same padding to keep the output length the same as input length, adjusted for dilation
                bias=False,
            )
            for d in dilations
        ])

        #pointwise convolution to mix the outputs of the parallel convolutions, it has kernel size 1 and no bias
        self.pw_conv = nn.Conv1d(
            N_filters,
            N_filters,
            kernel_size=1,
            bias=False,
        )

        self.prelu = nn.PReLU(num_parameters=N_filters)
        self.norm = GlobalLayerNorm(N_filters) #normalization after the pointwise convolution
        self.dropout = nn.Dropout(p=dropout_p) #dropout to reduce overfitting, applied after normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x #save the original input
        branch_sum = sum(conv(x) for conv in self.dw_convs) #sum the outputs of the parallel depthwise convolutions, this gives us a multi-scale representation

        # Pointwise projection, activation, normalisation and dropout.
        out = self.pw_conv(branch_sum)
        out = self.prelu(out)
        out = self.norm(out)
        out = self.dropout(out)

        return out + residual

##############################
#Bottleneck Projection
##############################

class BottleneckProjection(nn.Module):
    #reduces the dimensionality passed to the separator and helps control model size

    def __init__(
        self,
        N_filters: int = cfg.N_FILTERS,
        N_latent: int = cfg.N_LATENT,
    ):
        super().__init__()

        self.proj = nn.Conv1d(
            N_filters,
            N_latent,
            kernel_size=1,
            bias=False,
        )

        self.norm = GlobalLayerNorm(N_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #output=Z : [B, N_latent, L]
        return self.norm(self.proj(x))

##############################
#Main encoder
##############################

class CardiopulmonaryEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        N_filters: int = cfg.N_FILTERS,
        N_latent: int = cfg.N_LATENT,
        kernel_size: int = cfg.KERNEL_SIZE,
        stride: int = cfg.STRIDE,
        R: int = cfg.ENCODER_R,
        dilations: tuple = cfg.DILATIONS,
        dropout_p: float = cfg.DROPOUT_P,
    ):
        super().__init__()
        self.stride = stride

        self.basis = LearnableBasisEncoder(
            in_channels=in_channels,
            N_filters=N_filters,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.input_norm = GlobalLayerNorm(N_filters)

        self.dw_blocks = nn.Sequential(*[
            MultiScaleDWConvBlock(
                N_filters=N_filters,
                kernel_size=3,
                dilations=dilations,
                dropout_p=dropout_p,
            )
            for _ in range(R)
        ])

        self.bottleneck = BottleneckProjection(
            N_filters=N_filters,
            N_latent=N_latent,
        )

        self.config = dict(
            in_channels=in_channels,
            N_filters=N_filters,
            N_latent=N_latent,
            kernel_size=kernel_size,
            stride=stride,
            R=R,
            dilations=dilations,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.basis(x) #Learnable Encoder, .forward -> padding + conv1D + gelu
        x = self.input_norm(x) #Global Layer, .forward -> normalization
        x = self.dw_blocks(x) #MultiScaleDW, .forward -> 3 blocks and parallel convolution
        Z = self.bottleneck(x) #BottleNeck
        return Z

    def receptive_field_ms(self, sr: int = cfg.SR) -> float:
        # Compute the convolutional path's local receptive field in milliseconds.
        # GlobalLayerNorm also introduces dependence across the full segment.

        dw_kernel = 3
        max_dilation = max(self.config["dilations"])
        R = self.config["R"]
        stride = self.config["stride"]
        basis_kernel = self.config["kernel_size"]
        rf_latent_frames = 1 + R * (dw_kernel - 1) * max_dilation
        rf_input_samples = basis_kernel + (rf_latent_frames - 1) * stride

        return rf_input_samples / sr * 1000

    def n_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def __repr__(self):
        cfg_local = self.config
        rf = self.receptive_field_ms()

        return (
            f"CardiopulmonaryEncoder(\n"
            f"  N_filters={cfg_local['N_filters']}, "
            f"N_latent={cfg_local['N_latent']},\n"
            f"  kernel={cfg_local['kernel_size']}, "
            f"stride={cfg_local['stride']}, "
            f"R={cfg_local['R']},\n"
            f"  dilations={cfg_local['dilations']}, "
            f"dropout={cfg_local['dropout_p']},\n"
            f"  approx_local_receptive_field={rf:.1f} ms,\n"
            f"  trainable_params={self.n_parameters():,}\n"
            f")"
        )

##############################
#Sanity check
##############################
if __name__ == "__main__":
    SR = cfg.SR
    SEG_S = cfg.SEG_SECONDS
    BATCH = cfg.BATCH_SIZE
    T = cfg.SEG_SAMPLES

    print("=" * 60)
    print("CardiopulmonaryEncoder — sanity check")
    print("=" * 60)

    encoder = CardiopulmonaryEncoder(
        in_channels=1,
        N_filters=cfg.N_FILTERS,
        N_latent=cfg.N_LATENT,
        kernel_size=cfg.KERNEL_SIZE,
        stride=cfg.STRIDE,
        R=cfg.ENCODER_R,
        dilations=cfg.DILATIONS,
        dropout_p=cfg.DROPOUT_P,
    )

    print(encoder)
    print()

    x = torch.randn(BATCH, 1, T)

    print(f"Input  x : {list(x.shape)}   [{SR} Hz × {SEG_S}s]")

    encoder.eval()
    with torch.no_grad():
        Z = encoder(x)

    print(f"Output Z : {list(Z.shape)}")
    print()

    expected_L = Z.shape[-1]
    expected_shape = (BATCH, cfg.N_LATENT, expected_L)

    assert Z.shape == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {Z.shape}"
    )

    rf = encoder.receptive_field_ms(SR)

    assert rf > 50, (
        f"Receptive field {rf:.1f} ms too short for local HS morphology"
    )

    print("Shape check      : PASSED")
    print(f"Latent length    : {expected_L} frames")
    print(f"Receptive field  : {rf:.1f} ms")
    print(f"Trainable params : {encoder.n_parameters():,}")
    print()

    z_bytes = Z.element_size() * Z.nelement()

    print(f"Latent Z size (fp32)   : {z_bytes / 1024:.1f} KB per batch of {BATCH}")
    print(f"Latent Z per sample    : {z_bytes / BATCH / 1024:.2f} KB")
    print()

    encoder.train()

    x_grad = torch.randn(BATCH, 1, T)
    Z_train = encoder(x_grad)
    loss = Z_train.mean()
    loss.backward()

    grad_ok = all(
        p.grad is not None
        for p in encoder.parameters()
        if p.requires_grad
    )

    print(f"Gradient flow: {'PASSED' if grad_ok else 'FAILED'}")
    print("=" * 60)
    print()
    print("Next step → pass Z to the Separator module.")
    print("Z shape [B, N_latent, L] =", list(Z.shape))
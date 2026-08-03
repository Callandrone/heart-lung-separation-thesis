"""
decoder.py
==========
The decoder is called twice independently:
    Z_hs → decoder → H_hat
    Z_ls → decoder → L_hat

Key design choices
------------------
- No weight tying with the encoder.
- No activation after ConvTranspose1D because audio is bipolar.
- DW-Conv blocks before deconvolution refine the masked latent representation.
- During training, normalise=False is recommended so the loss sees the raw
  reconstructed waveform.
- For listening/export, normalise=True can be used to keep the waveform within
  a controlled amplitude range.
------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import GlobalLayerNorm
import model_config as cfg

###################################
#Sub-module 1: Bottleneck expansion
###################################
class BottleneckExpansion(nn.Module):

    def __init__(
        self,
        N_latent: int = cfg.N_LATENT,
        N_filters: int = cfg.N_FILTERS,
    ):
        super().__init__()

        self.expand = nn.Conv1d(
            N_latent,
            N_filters,
            kernel_size=1,
            bias=False,
        )

        self.use_input_gln = getattr(
            cfg,
            "DECODER_USE_INPUT_GLN",
            True,
        )

        self.norm = (
            GlobalLayerNorm(N_filters)
            if self.use_input_gln
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.expand(x))


########################################
#Sub-module 2: DW-Conv refinement block
########################################

class DecoderDWConvBlock(nn.Module):
    """
    Multi-scale dilated depthwise Conv1D refinement block.

    It refines the masked latent representation before waveform reconstruction.
    This can reduce frame-boundary artifacts caused by sending masked latent
    features directly into ConvTranspose1D.

    The structure is similar to the encoder DW block, but parameters are
    independent from the encoder.

    Parameters
    ----------
    N_filters : int
        Decoder feature channels.
    kernel_size : int
        Depthwise convolution kernel size.
    dilations : tuple[int]
        Parallel dilation factors.
    dropout_p : float
        Dropout probability.
    """

    def __init__(
        self,
        N_filters: int = cfg.N_FILTERS,
        kernel_size: int = 3,
        dilations: tuple = cfg.DILATIONS,
        dropout_p: float = cfg.DROPOUT_P,
    ):
        super().__init__()

        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                N_filters,
                N_filters,
                kernel_size=kernel_size,
                dilation=d,
                groups=N_filters,
                padding=d * (kernel_size - 1) // 2,
                bias=False,
            )
            for d in dilations
        ])

        self.pw_conv = nn.Conv1d(
            N_filters,
            N_filters,
            kernel_size=1,
            bias=False,
        )

        self.prelu = nn.PReLU(num_parameters=N_filters)
        self.norm = GlobalLayerNorm(N_filters)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, N_filters, L]

        Returns
        -------
        [B, N_filters, L]
        """
        residual = x

        branch_sum = sum(conv(x) for conv in self.dw_convs)

        out = self.pw_conv(branch_sum)
        out = self.prelu(out)
        out = self.norm(out)
        out = self.dropout(out)

        return out + residual


######################################################################
#Sub-module 3: ConvTranspose1D waveform reconstruction
######################################################################

#reconstruct waveform from refined latent features
class WaveformReconstructor(nn.Module):

    def __init__(
        self,
        N_filters: int = cfg.N_FILTERS,
        kernel_size: int = cfg.KERNEL_SIZE,
        stride: int = cfg.STRIDE,
    ):
        super().__init__()

        self.stride = stride

        self.deconv = nn.ConvTranspose1d(
            N_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size - stride,
            bias=False,
        ) #it mirrors the encoder's learned Conv1D basis, upsampling and filtering in 1 operation

        #small initial weights keep the decoder close to zero output at init.
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.01)
        #no activation function after ConvTranspose since audio is bipolar and the output must not be clipped
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


######################################################
#Sub-module 4: Optional per-sample peak normalisation
######################################################

# Peak-normalise each sample independently.
def peak_normalise(
    x: torch.Tensor,
    target_peak: float = 0.95,
) -> torch.Tensor:

    peak = x.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return x * (target_peak / peak)


#################
#Main decoder
#################

class CardiopulmonaryDecoder(nn.Module):

    def __init__(
        self,
        N_latent: int = cfg.N_LATENT,
        N_filters: int = cfg.N_FILTERS,
        kernel_size: int = cfg.KERNEL_SIZE,
        stride: int = cfg.STRIDE,
        R: int = cfg.DECODER_R,
        dilations: tuple = cfg.DILATIONS,
        dropout_p: float = cfg.DROPOUT_P,
        target_T: int | None = None,
    ):
        super().__init__()
        self.target_T = target_T
        self.stride = stride

        self.expansion = BottleneckExpansion( N_latent=N_latent, N_filters=N_filters,)

        self.refine_blocks = nn.Sequential(*[
            DecoderDWConvBlock( N_filters=N_filters, kernel_size=3, dilations=dilations, dropout_p=dropout_p,)
            for _ in range(R)
        ])

        self.reconstructor = WaveformReconstructor(N_filters=N_filters, kernel_size=kernel_size, stride=stride,)

        self.config = dict(
            N_latent=N_latent,
            N_filters=N_filters,
            kernel_size=kernel_size,
            stride=stride,
            R=R,
            dilations=dilations,
            dropout_p=dropout_p,
            target_T=target_T,
        )

    def forward( self,  Z_src: torch.Tensor,  normalise: bool = True,  ) -> torch.Tensor:

        x = self.expansion(Z_src)
        x = self.refine_blocks(x)
        waveform = self.reconstructor(x) #generate the waveform

        if self.target_T is not None:
            T_out = waveform.shape[-1]
            #this ensures the output is exactly T=8000
            if T_out > self.target_T:
                waveform = waveform[..., :self.target_T]
            elif T_out < self.target_T:
                pad = self.target_T - T_out
                waveform = F.pad(waveform, (0, pad))

        if normalise:
            waveform = peak_normalise(waveform)

        return waveform

    def n_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def __repr__(self):
        local_cfg = self.config

        return (
            f"CardiopulmonaryDecoder(\n"
            f"  N_latent={local_cfg['N_latent']}, "
            f"N_filters={local_cfg['N_filters']},\n"
            f"  kernel={local_cfg['kernel_size']}, "
            f"stride={local_cfg['stride']},\n"
            f"  R={local_cfg['R']} DW-Conv refinement blocks,\n"
            f"  dilations={local_cfg['dilations']},\n"
            f"  target_T={local_cfg['target_T']},\n"
            f"  trainable_params={self.n_parameters():,}\n"
            f")"
        )



##############################
#Sanity check
##############################
if __name__ == "__main__":
    import math
    SR = cfg.SR
    SEG_S = cfg.SEG_SECONDS
    T = cfg.SEG_SAMPLES
    B = 4
    D = cfg.N_LATENT
    L = math.ceil(T / cfg.STRIDE)

    print("=" * 60)
    print("CardiopulmonaryDecoder — sanity check")
    print("=" * 60)

    decoder = CardiopulmonaryDecoder(
        N_latent=cfg.N_LATENT,
        N_filters=cfg.N_FILTERS,
        kernel_size=cfg.KERNEL_SIZE,
        stride=cfg.STRIDE,
        R=cfg.DECODER_R,
        dilations=cfg.DILATIONS,
        dropout_p=cfg.DROPOUT_P,
        target_T=T,
    )

    print(decoder)
    print()

    #Simulate separator output.
    Z_hs = torch.randn(B, D, L)
    Z_ls = torch.randn(B, D, L)

    print(f"Input  Z_hs: {list(Z_hs.shape)}")
    print(f"Input  Z_ls : {list(Z_ls.shape)}")

    decoder.eval()
    with torch.no_grad():
        H_hat = decoder(Z_hs, normalise=True)
        L_hat = decoder(Z_ls, normalise=True)

    print(f"Output H_hat: {list(H_hat.shape)}")
    print(f"Output L_hat: {list(L_hat.shape)}")
    print()

    assert H_hat.shape == (B, 1, T), f"H_hat shape error: {H_hat.shape}"
    assert L_hat.shape == (B, 1, T), f"L_hat shape error: {L_hat.shape}"

    print("Shape check: PASSED")

    assert H_hat.abs().max() <= 1.0 + 1e-5, "H_hat exceeds ±1"
    assert L_hat.abs().max() <= 1.0 + 1e-5, "L_hat exceeds ±1"

    print(f"Amplitude range H: [{H_hat.min():.3f}, {H_hat.max():.3f}]")
    print(f"Amplitude range L: [{L_hat.min():.3f}, {L_hat.max():.3f}]")
    print(f"Trainable params: {decoder.n_parameters():,}")
    print()

    decoder.train()

    Zh = torch.randn(B, D, L)
    Zl = torch.randn(B, D, L)

    Hh = decoder(Zh, normalise=False)
    Lh = decoder(Zl, normalise=False)

    loss = Hh.mean() + Lh.mean()
    loss.backward()

    grad_ok = all(
        p.grad is not None
        for p in decoder.parameters()
        if p.requires_grad
    )

    print(f"Gradient flow: {'PASSED' if grad_ok else 'FAILED'}")

    print()
    print("─" * 60)
    print("Full pipeline test (Encoder → Separator → Decoder)")
    print("─" * 60)

    try:
        from encoder import CardiopulmonaryEncoder
        from separator import CardiopulmonarySeparator

        enc = CardiopulmonaryEncoder(
            in_channels=1,
            N_filters=cfg.N_FILTERS,
            N_latent=cfg.N_LATENT,
            kernel_size=cfg.KERNEL_SIZE,
            stride=cfg.STRIDE,
            R=cfg.ENCODER_R,
            dilations=cfg.DILATIONS,
            dropout_p=cfg.DROPOUT_P,
        )

        sep = CardiopulmonarySeparator(
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

        enc.eval()
        sep.eval()
        decoder.eval()

        M = torch.randn(2, 1, T)

        with torch.no_grad():
            Z = enc(M)
            Z_hs, Z_ls, _ = sep(Z)
            H_out = decoder(Z_hs, normalise=True)
            L_out = decoder(Z_ls, normalise=True)

        assert Z.shape == (2, cfg.N_LATENT, L), f"Z shape error: {Z.shape}"
        assert Z_hs.shape == (2, cfg.N_LATENT, L), f"Z_hs shape error: {Z_hs.shape}"
        assert Z_ls.shape == (2, cfg.N_LATENT, L), f"Z_ls shape error: {Z_ls.shape}"
        assert H_out.shape == (2, 1, T), f"H_out shape error: {H_out.shape}"
        assert L_out.shape == (2, 1, T), f"L_out shape error: {L_out.shape}"

        total_params = (
            enc.n_parameters()
            + sep.n_parameters()
            + decoder.n_parameters()
        )

        print(f"Input M: {list(M.shape)}")
        print(f"Latent Z: {list(Z.shape)}")
        print(f"Output H_hat: {list(H_out.shape)}")
        print(f"Output L_hat: {list(L_out.shape)}")
        print(f"Total params: {total_params:,}")
        print("Full pipeline: PASSED")

    except ImportError as e:
        print(f"Skipped full pipeline test (missing module): {e}")
        print("Run from the directory containing encoder.py and separator.py")

    print("=" * 60)
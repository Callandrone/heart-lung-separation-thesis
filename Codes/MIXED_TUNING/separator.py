"""JASSNet-style masking network and joint-attention separator.

Implemented from the architecture and equations described in Zhang et al.
(2025):
- input LayerNorm, positional encoding and pointwise convolution;
- R stacked separation modules;
- convolutional modules using LayerNorm, depthwise Conv1D, linear/ReLU,
  second depthwise Conv1D, skip connection and dropout;
- attentive gating with U and V expanded to 2N;
- local chunked attention using ReLU^2((Q K^T) / P);
- global linearised attention Q ((K^T V) / S);
- gated output layer and non-negative ReLU source masks.

The article does not provide every numerical hyperparameter needed to recreate
an official implementation, so configured values are recorded in model_config.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_config as cfg


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic absolute positional information to [B, S, N]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected positional dim={self.dim}, got {dim}")
        device, dtype = x.device, x.dtype
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)


class ConvolutionModule(nn.Module):
    """
    This block captures local features for the global and local attention algorithm
    Depthwise convolution works channel by channel and Relu introduces non linearity
    The second depthwise block refines the sequence
    Dropout reduces overfitting
        LayerNorm
        -> Depthwise Conv1D
        -> Linear projection
        -> ReLU
        -> residual branch
        -> second Depthwise Conv1D
        -> residual sum
        -> Dropout

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = cfg.JASSNET_CONV_KERNEL,
        dropout_p: float = cfg.DROPOUT_P,
    ) -> None:

        super().__init__()
        pad = kernel_size // 2

        self.norm = nn.LayerNorm(in_dim) #LayerNorm to start

        self.dw_in = nn.Conv1d(
            in_dim,
            in_dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=in_dim,
            bias=False,
        ) #DepthWise convolution for temporal information with kernel equal to 3

        self.linear = nn.Linear(
            in_dim,
            out_dim,
            bias=False,
        ) #Linear Projection, from 128 to 256

        self.activation = nn.ReLU() #Non linearity to U and V

        self.dw_out = nn.Conv1d(
            out_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=out_dim,
            bias=False,
        ) #Second depthwise convolution

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x) #LayerNorm
        out = self.dw_in(out.transpose(1, 2)).transpose(1, 2) #DW
        out = self.activation(self.linear(out)) #Linear Projection before relu
        residual = out
        out = self.dw_out(out.transpose(1, 2)).transpose(1, 2) #DW
        out = out + residual
        return self.dropout(out)


class RotaryPositionEmbedding(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                f"RotaryPositionEmbedding requires an even dimension, got {dim}."
            )

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]

        length = x.shape[1]

        positions = torch.arange(
            length,
            device=x.device,
            dtype=self.inv_freq.dtype,
        )

        angles = torch.einsum(
            "s,d->sd",
            positions,
            self.inv_freq,
        ).to(dtype=x.dtype)

        cos = torch.repeat_interleave(
            angles.cos(),
            repeats=2,
            dim=-1,
        ).unsqueeze(0)

        sin = torch.repeat_interleave(
            angles.sin(),
            repeats=2,
            dim=-1,
        ).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.stack(
            (-x_odd, x_even),
            dim=-1,
        ).flatten(-2)

        return x * cos + x_rotated * sin


class ScaleOffsetRotary(nn.Module):
    """
    Applica scale, offset e Rotary Positional Embedding alla
    rappresentazione condivisa Z per ottenere Q, K, Q' e K'.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.offset = nn.Parameter(torch.zeros(1, 1, dim))
        self.rotary = RotaryPositionEmbedding(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.rotary(self.scale * z + self.offset)


class JointLocalGlobalAttention(nn.Module):
    """
    Joint local-global attention
    Shared representation:
        Z = ConvM(X'') in R^(S x D), con D << N

    It is used to build
        Q, K   -> local attention
        Q', K' -> linearised gloabal attention
    """

    def __init__(
        self,
        dim: int,
        value_dim: int,
        attn_dim: int,
        local_chunk: int,
        rpe_kernel: int,
        dropout_p: float,
    ) -> None:
        super().__init__()

        if local_chunk <= 0:
            raise ValueError("local_chunk must be positive")

        self.local_chunk = local_chunk

        #Z, third shared convolutional block
        self.shared_z_conv = ConvolutionModule(
            in_dim=dim,
            out_dim=attn_dim,
            dropout_p=dropout_p,
        )

        #Scale + offset + Rotary Positional Embedding
        self.local_q = ScaleOffsetRotary(attn_dim)
        self.local_k = ScaleOffsetRotary(attn_dim)
        self.global_q = ScaleOffsetRotary(attn_dim)
        self.global_k = ScaleOffsetRotary(attn_dim)

    def _global_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
            V'_global = Q' ( beta K'^T V )
            U'_global = Q' ( beta K'^T U )
        with beta = 1 / S.
        """

        length = max(1, values.shape[1])

        context = torch.einsum(
            "bsd,bsv->bdv",
            k,
            values,
        ) / float(length)

        return torch.einsum(
            "bsd,bdv->bsv",
            q,
            context,
        )

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
            ReLU^2( gamma Q_h K_h^T ) V_h
        with gamma = 1 / P.
        """

        batch, length, qdim = q.shape
        value_dim = values.shape[-1]
        chunk = self.local_chunk

        pad = (-length) % chunk

        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            values = F.pad(values, (0, 0, 0, pad))

        n_chunks = q.shape[1] // chunk

        q = q.view(batch, n_chunks, chunk, qdim)
        k = k.view(batch, n_chunks, chunk, qdim)
        values = values.view(batch, n_chunks, chunk, value_dim)

        scores = torch.einsum(
            "bhpd,bhqd->bhpq",
            q,
            k,
        ) / float(chunk)

        weights = F.relu(scores).pow(2)

        output = torch.einsum(
            "bhpq,bhqv->bhpv",
            weights,
            values,
        )

        output = output.reshape(
            batch,
            n_chunks * chunk,
            value_dim,
        )

        return output[:, :length, :]

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        #Shared low-dimensional representation:
        z = self.shared_z_conv(x)

        #Local Q, K and global Q', K'
        q_local = self.local_q(z)
        k_local = self.local_k(z)
        q_global = self.global_q(z)
        k_global = self.global_k(z)

        u_prime = (
            self._local_attention(q_local, k_local, u)
            + self._global_attention(q_global, k_global, u)
        )

        v_prime = (
            self._local_attention(q_local, k_local, v)
            + self._global_attention(q_global, k_global, v)
        )

        return u_prime, v_prime


class SeparationModule(nn.Module):
    """Attentive gating module using the paper's U/V formulation."""

    def __init__(
        self,
        dim: int,
        expansion: int,
        attn_dim: int,
        local_chunk: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        value_dim = expansion * dim
        self.u_conv = ConvolutionModule(dim, value_dim, dropout_p=dropout_p)
        self.v_conv = ConvolutionModule(dim, value_dim, dropout_p=dropout_p)
        self.attention = JointLocalGlobalAttention(
            dim=dim,
            value_dim=value_dim,
            attn_dim=attn_dim,
            local_chunk=local_chunk,
            rpe_kernel=cfg.JASSNET_RPE_KERNEL,
            dropout_p=dropout_p,
        )
        self.merge = ConvolutionModule(value_dim, dim, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u_conv(x)
        v = self.v_conv(x)
        u_prime, v_prime = self.attention(x, u, v)
        o_prime = torch.sigmoid(u * v_prime)
        o_double_prime = u_prime * v
        return x + self.merge(o_prime * o_double_prime)


class CardiopulmonarySeparator(nn.Module):
    """Generate heart/lung masks over the encoded mixture features."""

    def __init__(
        self,
        dim: int = cfg.N_LATENT,
        n_sources: int = cfg.N_SOURCES,
        n_stacks: int = 1,
        S: int = cfg.JASSNET_NUM_MODULES,
        tcn_bottleneck: int = cfg.JASSNET_ATTN_DIM,
        attn_heads: int = 1,
        attn_window: int = cfg.JASSNET_LOCAL_CHUNK,
        n_global: int = 0,
        dropout_p: float = cfg.DROPOUT_P,
        mask_scale: object = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_sources = n_sources
        self.num_modules = int(S) if S is not None else cfg.JASSNET_NUM_MODULES
        self.input_norm = nn.LayerNorm(dim)
        self.position = SinusoidalPositionalEncoding(dim)
        self.input_pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.modules_stack = nn.ModuleList(
            [
                SeparationModule(
                    dim=dim,
                    expansion=cfg.JASSNET_EXPANSION,
                    attn_dim=tcn_bottleneck,
                    local_chunk=attn_window,
                    dropout_p=dropout_p,
                )
                for _ in range(self.num_modules)
            ]
        )
        self.prelu = nn.PReLU(num_parameters=dim)
        output_dim = n_sources * dim
        self.output_expand = nn.Conv1d(dim, output_dim, kernel_size=1)
        self.output_tanh = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        self.output_sigmoid = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        self.mask_head = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        self.mask_activation = nn.ReLU()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = z.transpose(1, 2) #LayerNorm works on the last dimension
        x = self.input_norm(x) #LayerNorm
        if cfg.JASSNET_POSITIONAL_ENCODING:
            x = self.position(x) #Sinusoidal Positional Encoding
        x = self.input_pointwise(x.transpose(1, 2)).transpose(1, 2)
        for module in self.modules_stack:
            x = module(x)
        x = self.prelu(x.transpose(1, 2))
        x = self.output_expand(x)
        gated = torch.tanh(self.output_tanh(x)) * torch.sigmoid(self.output_sigmoid(x))
        masks = self.mask_activation(self.mask_head(gated))
        masks = masks.view(z.shape[0], self.n_sources, self.dim, z.shape[-1])
        z_hs = z * masks[:, 0]
        z_ls = z * masks[:, 1]
        return z_hs, z_ls, masks

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    length = (cfg.SEG_SAMPLES - cfg.KERNEL_SIZE) // cfg.STRIDE + 1
    separator = CardiopulmonarySeparator()
    z = torch.randn(2, cfg.JASSNET_DIM, length, requires_grad=True)
    z_hs, z_ls, masks = separator(z)
    assert z_hs.shape == z.shape
    assert z_ls.shape == z.shape
    assert masks.shape == (2, cfg.N_SOURCES, cfg.JASSNET_DIM, length)
    assert masks.min() >= 0
    (z_hs.mean() + z_ls.mean()).backward()
    print(
        f"Separator outputs: {tuple(z_hs.shape)}, {tuple(z_ls.shape)}, "
        f"masks={tuple(masks.shape)} | params={separator.n_parameters():,}"
    )

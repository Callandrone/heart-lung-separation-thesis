import torch
import torch.nn as nn

"""
  Tensor shape -> [B, C, T].
  Unlike Batch Norm (which normalizes across the batch dimension) or standard
  Layer Norm (which normalizes across channels only), gLN computes a single
  mean and variance per sample by collapsing both the channel (C) and time (T)
  dimensions together. This makes the statistics global with respect to the
  entire feature map of each sample, while remaining independent across the
  batch.
  """

class GlobalLayerNorm(nn.Module):

    #dim is the number of channels (C), eps avoids division by zero
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1)) #learnable scale parameter, initialized to ones, updated in training
        self.beta = nn.Parameter(torch.zeros(1, dim, 1)) #learnable shift parameter, initialized to zeros, updated in training

    #forward receives x of shape [B, C, T] and returns the normalized tensor with the same shape
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True) #mean across channel and time dimensions
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False) #it measures spread of activations
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta #output
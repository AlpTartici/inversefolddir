# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
time_embed.py

This script defines the Gaussian Fourier Projection layer, a technique for
creating rich, fixed (non-learned) embeddings of a scalar value, in this case,
time `t`. This is a crucial component in diffusion and flow-matching models.
"""

import numpy as np
import torch
import torch.nn as nn


class GaussianFourierProjection(nn.Module):
    """
    Implements Gaussian Fourier Projection for encoding time.

    This layer maps a scalar time `t` (representing the noise level) to a
    higher-dimensional feature vector. The transformation is fixed and not
    learned during training. It works by projecting `t` onto a set of random
    frequencies `W` sampled from a Gaussian distribution, and then passing the
    result through sine and cosine functions.

    This provides the model with a periodic, multi-scale representation of time,
    which is more expressive than a simple scalar input.
    """

    def __init__(self, embed_dim: int, scale: float):
        """
        Args:
            embed_dim (int): The desired dimension of the time embedding. Must be even.
            scale (float): A scaling factor for the random frequencies. It controls
                           the range of frequencies in the embedding.
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, but got {embed_dim}")

        self.embed_dim = embed_dim

        # Sample the random frequencies `W` from a standard normal distribution
        # and scale them. These frequencies are fixed after initialization.
        W = torch.randn(embed_dim // 2) * scale

        # Register `W` as a buffer. This means it's part of the module's state
        # but is not considered a trainable parameter.
        self.register_buffer("W", W)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Applies the Gaussian Fourier Projection to a batch of time values.

        Args:
            t (torch.Tensor): A tensor of shape [B] or [B,1] containing the
                              time steps (noise levels) for each sample in the batch.

        Returns:
            A tensor of shape [B, embed_dim] representing the time embeddings.
        """
        alpha = t + 1
        # Ensure t is a column vector for broadcasting
        t_proj = alpha.view(-1, 1) * self.W.view(1, -1) * 2 * np.pi

        # Concatenate the sine and cosine projections to form the final embedding
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

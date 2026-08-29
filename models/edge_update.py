# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
SE(3)-equivariant edge feature update module.

This module provides edge feature updates that preserve SE(3) equivariance:
- Scalar edge features updated via MLP on rotation-invariant inputs
- Vector edge features updated via scalar-gated linear combinations + cross products

Design considerations:
- Cross products computed element-wise across multi-channel vectors, then aggregated
- Cross product MAGNITUDES included as invariant features (provides sin(θ) directly)
- Near-parallel vectors handled via soft gating (sigmoid on norm) rather than hard mask
- Original edge_v (relative displacement direction) explicitly preserved via residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorLayerNorm(nn.Module):
    """
    LayerNorm for vector features that preserves direction.

    Normalizes by RMS of vector magnitudes across channels, preserving
    the relative directions and SE(3) equivariance.
    """
    def __init__(self, num_vectors: int, eps: float = 1e-8):
        super().__init__()
        self.num_vectors = num_vectors
        self.eps = eps

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: Vector features [*, num_vectors, 3]
        Returns:
            Normalized vectors [*, num_vectors, 3]
        """
        assert v.dim() >= 2, f"Expected at least 2D tensor, got {v.dim()}D"
        assert v.shape[-1] == 3, f"Expected last dim to be 3, got {v.shape[-1]}"

        # Compute norms and RMS normalize
        # Add eps inside sqrt to prevent gradient explosion when squared norms are tiny
        vn = torch.norm(v, dim=-1, keepdim=True)  # [*, num_vectors, 1]
        vn_rms = torch.sqrt(torch.mean(vn ** 2, dim=-2, keepdim=True) + self.eps)  # [*, 1, 1]

        # Gradient-safe normalization: for near-zero RMS (degenerate geometry),
        # detach to prevent gradient explosion through 1/eps^2
        safe_mask = (vn_rms > 1e-4)  # Increased threshold for safety
        # Normal case: full gradient flow
        result_normal = v / vn_rms
        # Degenerate case: detach to prevent gradient explosion
        result_detached = v.detach() / vn_rms.detach()
        return torch.where(safe_mask, result_normal, result_detached)


class SE3EdgeUpdate(nn.Module):
    """
    SE(3)-equivariant edge feature update module.

    Updates edge features using:
    - Scalar update: MLP on invariant features + residual connection
    - Vector update: Scalar-gated linear combination of vectors + cross products + LayerNorm

    Equivariance is preserved by:
    1. Computing only rotation-invariant features (norms, dot products, cross product magnitudes) for scalar MLP
    2. Using scalar weights (from invariants) to combine equivariant vectors
    3. Cross products of equivariant vectors produce equivariant vectors

    Invariant features include:
    - Node scalar features (already invariant)
    - Vector norms (||v||)
    - Dot products (u·v = ||u|| ||v|| cos θ)
    - Cross product magnitudes (||u×v|| = ||u|| ||v|| sin θ)
      - Computed cheaply since we already compute cross product for vector update
      - Provides sin(θ) directly, reducing nonlinear burden on MLP

    Args:
        node_s_dim: Scalar dimension of node features
        node_v_dim: Number of vector channels in node features
        edge_s_dim: Scalar dimension of edge features
        edge_v_dim: Number of vector channels in edge features (typically 1)
        hidden_dim: Hidden dimension for MLPs
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_s_dim: int,
        node_v_dim: int,
        edge_s_dim: int,
        edge_v_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Store dimensions for validation
        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.edge_s_dim = edge_s_dim
        self.edge_v_dim = edge_v_dim

        # Calculate invariant feature dimension
        # Components:
        # - node_s_src, node_s_dst: node_s_dim * 2
        # - node_v norms (src, dst): node_v_dim * 2
        # - edge_v norms: edge_v_dim
        # - dot products (src-dst per channel): node_v_dim
        # - dot products (src-edge, dst-edge): edge_v_dim * 2
        # - cross product magnitude (src×dst): 1 (aggregated across channels)
        # - cross product magnitudes (src×edge, dst×edge): edge_v_dim * 2
        # - edge_s: edge_s_dim
        self.invariant_dim = (
            node_s_dim * 2 +           # Node scalars
            node_v_dim * 2 +           # Node vector norms
            edge_v_dim +               # Edge vector norms
            node_v_dim +               # Dot products src-dst
            edge_v_dim * 2 +           # Dot products node-edge
            1 +                        # Cross product magnitude src×dst
            edge_v_dim * 2 +           # Cross product magnitudes node×edge
            edge_s_dim                 # Edge scalars
        )

        # Scalar edge update MLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.invariant_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_s_dim),
        )

        # Vector weight MLP - outputs 4 weights for:
        # w1: existing edge_v (relative displacement - important to preserve!)
        # w2: aggregated node_v[src]
        # w3: aggregated node_v[dst]
        # w4: cross product of node vectors
        self.vec_weight_mlp = nn.Sequential(
            nn.Linear(self.invariant_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

        # Learnable scale for soft gating of cross products
        # Controls how quickly the cross product contribution drops for parallel vectors
        self.cross_scale = nn.Parameter(torch.tensor(1.0))

        # Vector LayerNorm for stability
        self.vector_layer_norm = VectorLayerNorm(edge_v_dim)

        # Initialize final layers close to zero for stable training
        # This means initial updates are small, preserving original features
        nn.init.zeros_(self.scalar_mlp[-1].weight)
        nn.init.zeros_(self.scalar_mlp[-1].bias)
        nn.init.zeros_(self.vec_weight_mlp[-1].weight)
        nn.init.zeros_(self.vec_weight_mlp[-1].bias)

    def _validate_inputs(
        self,
        edge_index: torch.Tensor,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> None:
        """Validate input tensor shapes and values."""
        num_nodes = node_s.shape[0]
        num_edges = edge_index.shape[1]

        # Edge index validation
        assert edge_index.dim() == 2, f"edge_index should be 2D, got {edge_index.dim()}D"
        assert edge_index.shape[0] == 2, f"edge_index should have shape [2, E], got {edge_index.shape}"
        assert edge_index.max() < num_nodes, (
            f"edge_index contains invalid node index {edge_index.max().item()}, "
            f"but only {num_nodes} nodes exist"
        )
        assert edge_index.min() >= 0, f"edge_index contains negative index {edge_index.min().item()}"

        # Node feature validation
        assert node_s.dim() == 2, f"node_s should be 2D [N, s_dim], got {node_s.dim()}D"
        assert node_s.shape[1] == self.node_s_dim, (
            f"node_s dim mismatch: expected {self.node_s_dim}, got {node_s.shape[1]}"
        )
        assert node_v.dim() == 3, f"node_v should be 3D [N, v_dim, 3], got {node_v.dim()}D"
        assert node_v.shape[1] == self.node_v_dim, (
            f"node_v dim mismatch: expected {self.node_v_dim}, got {node_v.shape[1]}"
        )
        assert node_v.shape[2] == 3, f"node_v last dim should be 3, got {node_v.shape[2]}"

        # Edge feature validation
        assert edge_s.dim() == 2, f"edge_s should be 2D [E, s_dim], got {edge_s.dim()}D"
        assert edge_s.shape[0] == num_edges, (
            f"edge_s has {edge_s.shape[0]} edges but edge_index has {num_edges}"
        )
        assert edge_s.shape[1] == self.edge_s_dim, (
            f"edge_s dim mismatch: expected {self.edge_s_dim}, got {edge_s.shape[1]}"
        )
        assert edge_v.dim() == 3, f"edge_v should be 3D [E, v_dim, 3], got {edge_v.dim()}D"
        assert edge_v.shape[0] == num_edges, (
            f"edge_v has {edge_v.shape[0]} edges but edge_index has {num_edges}"
        )
        assert edge_v.shape[1] == self.edge_v_dim, (
            f"edge_v dim mismatch: expected {self.edge_v_dim}, got {edge_v.shape[1]}"
        )
        assert edge_v.shape[2] == 3, f"edge_v last dim should be 3, got {edge_v.shape[2]}"

    def _compute_invariants(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> tuple:
        """
        Compute rotation-invariant features for scalar MLP.

        All computed features are rotation-invariant:
        - Scalar features: invariant by definition
        - Vector norms: ||v|| is unchanged by rotation
        - Dot products: u·v = ||u|| ||v|| cos(θ) is invariant
        - Cross product magnitudes: ||u×v|| = ||u|| ||v|| sin(θ) is invariant

        Args:
            src: Source node indices [E]
            dst: Destination node indices [E]
            node_s: Node scalar features [N, node_s_dim]
            node_v: Node vector features [N, node_v_dim, 3]
            edge_s: Edge scalar features [E, edge_s_dim]
            edge_v: Edge vector features [E, edge_v_dim, 3]

        Returns:
            Tuple of (invariants tensor, cross product vector, cross product norm)
            - invariants: [E, invariant_dim]
            - cross: [E, 1, 3] - cross product of mean node vectors
            - cross_norm: [E, 1, 1] - magnitude of cross product
        """
        # Gather node features for source and destination
        node_s_src = node_s[src]  # [E, node_s_dim]
        node_s_dst = node_s[dst]  # [E, node_s_dim]
        node_v_src = node_v[src]  # [E, node_v_dim, 3]
        node_v_dst = node_v[dst]  # [E, node_v_dim, 3]

        # Vector norms (rotation-invariant)
        node_v_src_norms = torch.norm(node_v_src, dim=-1)  # [E, node_v_dim]
        node_v_dst_norms = torch.norm(node_v_dst, dim=-1)  # [E, node_v_dim]
        edge_v_norms = torch.norm(edge_v, dim=-1)          # [E, edge_v_dim]

        # Dot products between corresponding vector channels (rotation-invariant)
        # This captures angle information: cos(θ) between node vectors
        dots_src_dst = (node_v_src * node_v_dst).sum(dim=-1)  # [E, node_v_dim]

        # Mean-aggregate node vectors for computing edge-related features
        node_v_src_mean = node_v_src.mean(dim=1, keepdim=True)  # [E, 1, 3]
        node_v_dst_mean = node_v_dst.mean(dim=1, keepdim=True)  # [E, 1, 3]

        # Dot products between node vectors and edge vectors
        dots_src_edge = (node_v_src_mean * edge_v).sum(dim=-1)  # [E, edge_v_dim]
        dots_dst_edge = (node_v_dst_mean * edge_v).sum(dim=-1)  # [E, edge_v_dim]

        # Cross products and their magnitudes (provides sin(θ) directly)
        # This is cheap to compute since we need the cross product anyway for vector update
        cross_src_dst = torch.cross(node_v_src_mean, node_v_dst_mean, dim=-1)  # [E, 1, 3]
        cross_src_edge = torch.cross(node_v_src_mean, edge_v, dim=-1)          # [E, edge_v_dim, 3]
        cross_dst_edge = torch.cross(node_v_dst_mean, edge_v, dim=-1)          # [E, edge_v_dim, 3]

        # Cross product magnitudes (rotation-invariant, provides sin(θ))
        cross_src_dst_norm = torch.norm(cross_src_dst, dim=-1)  # [E, 1]
        cross_src_edge_norm = torch.norm(cross_src_edge, dim=-1)  # [E, edge_v_dim]
        cross_dst_edge_norm = torch.norm(cross_dst_edge, dim=-1)  # [E, edge_v_dim]

        # Concatenate all invariant features
        invariants = torch.cat([
            node_s_src,
            node_s_dst,
            node_v_src_norms,
            node_v_dst_norms,
            edge_v_norms,
            dots_src_dst,
            dots_src_edge,
            dots_dst_edge,
            cross_src_dst_norm,      # sin(θ) between node vectors
            cross_src_edge_norm,     # sin(θ) between src node and edge
            cross_dst_edge_norm,     # sin(θ) between dst node and edge
            edge_s,
        ], dim=-1)

        # Validate invariant dimension
        assert invariants.shape[-1] == self.invariant_dim, (
            f"Invariant dimension mismatch: expected {self.invariant_dim}, "
            f"got {invariants.shape[-1]}"
        )

        # Check for NaN/Inf
        if torch.isnan(invariants).any():
            raise ValueError("NaN detected in invariant features")
        if torch.isinf(invariants).any():
            raise ValueError("Inf detected in invariant features")

        # Return invariants and cross product info (reuse in vector update)
        return invariants, cross_src_dst, cross_src_dst_norm.unsqueeze(-1)

    def _update_vectors(
        self,
        node_v_src_mean: torch.Tensor,
        node_v_dst_mean: torch.Tensor,
        edge_v: torch.Tensor,
        cross: torch.Tensor,
        cross_norm: torch.Tensor,
        invariants: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update edge vectors equivariantly.

        Uses scalar-gated linear combination of:
        1. Existing edge vector (relative displacement direction)
        2. Aggregated source node vector
        3. Aggregated destination node vector
        4. Cross product of node vectors (provides new geometric basis)

        All weights are computed from invariants, ensuring equivariance.

        Args:
            node_v_src_mean: Mean source node vectors [E, 1, 3]
            node_v_dst_mean: Mean destination node vectors [E, 1, 3]
            edge_v: Edge vector features [E, edge_v_dim, 3]
            cross: Cross product of mean node vectors [E, 1, 3]
            cross_norm: Magnitude of cross product [E, 1, 1]
            invariants: Rotation-invariant features [E, invariant_dim]

        Returns:
            Updated edge vectors [E, edge_v_dim, 3]
        """
        eps = 1e-8

        # Compute scalar weights from invariants (preserves equivariance)
        weights = self.vec_weight_mlp(invariants)  # [E, 4]

        # Validate weights
        if torch.isnan(weights).any():
            raise ValueError("NaN in vector weights before activation")

        # Split into individual weights
        w1_raw, w2_raw, w3_raw, w4_raw = weights.split(1, dim=-1)

        # Apply sigmoid for soft gating in [0, 1]
        # These are scalar weights that will gate vector contributions
        w1 = torch.sigmoid(w1_raw)  # [E, 1] - weight for edge_v
        w2 = torch.sigmoid(w2_raw)  # [E, 1] - weight for node_v[src]
        w3 = torch.sigmoid(w3_raw)  # [E, 1] - weight for node_v[dst]

        # Soft gating for cross product based on its magnitude
        # When vectors are nearly parallel, ||cross|| → 0, and we smoothly reduce contribution
        # sigmoid(norm * scale) gives smooth transition
        cross_gate = torch.sigmoid(cross_norm * F.softplus(self.cross_scale))  # [E, 1, 1]

        # Normalize cross product with gradient-safe handling for degenerate cases
        # For edges where cross_norm is very small (parallel/zero vectors), detach from
        # gradient graph to prevent backward explosion. These cases have no meaningful
        # geometric gradient anyway.
        safe_mask = (cross_norm > 1e-4)  # [E, 1, 1] - threshold raised for gradient stability
        # Compute normalized cross product
        cross_norm_safe = cross_norm + eps
        cross_normalized_raw = cross / cross_norm_safe
        # For degenerate cases, detach to prevent gradient explosion (0/eps^2 -> huge)
        cross_normalized_detached = cross.detach() / cross_norm_safe.detach()
        cross_normalized = torch.where(safe_mask, cross_normalized_raw, cross_normalized_detached)

        # Apply both learned weight and geometry-based gate
        w4 = torch.sigmoid(w4_raw).unsqueeze(-1) * cross_gate  # [E, 1, 1]

        # Expand scalar weights for broadcasting with 3D vectors
        w1 = w1.unsqueeze(-1)  # [E, 1, 1]
        w2 = w2.unsqueeze(-1)  # [E, 1, 1]
        w3 = w3.unsqueeze(-1)  # [E, 1, 1]

        # Weighted combination (all operations preserve equivariance)
        # edge_v is the original relative displacement direction - important geometric info!
        delta = w1 * edge_v + w2 * node_v_src_mean + w3 * node_v_dst_mean + w4 * cross_normalized

        # Residual connection for stable training
        # This ensures original edge direction is preserved as a baseline
        edge_v_new = edge_v + delta

        # LayerNorm for stability across layers
        # Prevents magnitude drift while preserving direction
        edge_v_new = self.vector_layer_norm(edge_v_new)

        # Validate output
        assert edge_v_new.shape == edge_v.shape, (
            f"Output shape {edge_v_new.shape} doesn't match input {edge_v.shape}"
        )
        if torch.isnan(edge_v_new).any():
            raise ValueError("NaN in updated edge vectors")
        if torch.isinf(edge_v_new).any():
            raise ValueError("Inf in updated edge vectors")

        return edge_v_new

    def forward(
        self,
        edge_index: torch.Tensor,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> tuple:
        """
        Forward pass: update edge features equivariantly.

        Args:
            edge_index: Edge connectivity [2, E]
            node_s: Node scalar features [N, node_s_dim]
            node_v: Node vector features [N, node_v_dim, 3]
            edge_s: Edge scalar features [E, edge_s_dim]
            edge_v: Edge vector features [E, edge_v_dim, 3]

        Returns:
            Tuple of (updated_edge_s, updated_edge_v)
        """
        # Input validation
        self._validate_inputs(edge_index, node_s, node_v, edge_s, edge_v)

        src, dst = edge_index

        # Compute mean node vectors (needed for multiple features)
        node_v_src_mean = node_v[src].mean(dim=1, keepdim=True)  # [E, 1, 3]
        node_v_dst_mean = node_v[dst].mean(dim=1, keepdim=True)  # [E, 1, 3]

        # 1. Compute rotation-invariant features (also returns cross product for reuse)
        invariants, cross, cross_norm = self._compute_invariants(
            src, dst, node_s, node_v, edge_s, edge_v
        )

        # 2. Update scalar edge features (with residual)
        edge_s_delta = self.scalar_mlp(invariants)
        edge_s_new = edge_s + edge_s_delta

        # Validate scalar output
        if torch.isnan(edge_s_new).any():
            raise ValueError("NaN in updated edge scalars")
        if torch.isinf(edge_s_new).any():
            raise ValueError("Inf in updated edge scalars")

        # 3. Update vector edge features (equivariant)
        # Reuse mean node vectors and cross product computed in invariants
        edge_v_new = self._update_vectors(
            node_v_src_mean, node_v_dst_mean, edge_v, cross, cross_norm, invariants
        )

        return edge_s_new, edge_v_new

    def get_magnitude_stats(
        self,
        edge_v: torch.Tensor
    ) -> dict:
        """
        Get statistics about vector magnitudes for monitoring.

        Useful for debugging magnitude drift across layers.

        Args:
            edge_v: Edge vector features [E, edge_v_dim, 3]

        Returns:
            Dict with magnitude statistics
        """
        norms = torch.norm(edge_v, dim=-1)  # [E, edge_v_dim]
        return {
            'mean': norms.mean().item(),
            'std': norms.std().item(),
            'min': norms.min().item(),
            'max': norms.max().item(),
        }

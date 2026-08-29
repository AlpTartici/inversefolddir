# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
graph_builder.py

This module provides functionality to convert protein backbone coordinates into
graph representations suitable for geometric deep learning models. It constructs
k-nearest neighbor graphs with edge features based on 3D distances, sequence
distances, and geometric features like dihedral angles and orientations.

The GraphBuilder class is the main interface for creating protein graphs from
either CIF files or pre-processed coordinate tensors        # Scalar node feature                if not self._is_standard_amino_acid(aa):
                    seq_list[i] = 'X'  # Replace with unknown residue
                    if self.smoke_test:
                        print(f"  Replaced non-standard amino acid '{aa}' with 'X' at position {i}")r real no        # Vector: zeros for orientation directions and sidechain
        virtual_v = torch.cat([
            torch.zeros(1, 2, 3, device=coords4.device),  # Zero orientation directions
            torch.zeros(1, 1, 3, device=coords4.device),  # Zero sidechain
        ], dim=1)  # [1, 3, 3]

        # Combine real and virtual nodes
        node_s = torch.cat([node_s_real, virtual_s], dim=0)  # [L+1, 9]
        node_v = torch.cat([node_v_real, virtual_v], dim=0)  # [L+1, 3, 3]   node_s_real = torch.cat([
            dihedrals,                    # Dihedral angles [L, 6]
            scores_norm.unsqueeze(-1),    # Normalized scores [L, 1]
            source_indicator,             # Data source indicator [L, 2]
        ], dim=-1)  # [L, 9]

        # Vector node features for real nodes
        print(f"orientations shape: {orientations.shape}, sidechains shape: {sidechains.shape}", flush=True)
        node_v_real = torch.cat([
            orientations,                 # Local coordinate frames [L, 3, 3]
            sidechains,                   # Side chain directions [L, 1, 3]
        ], dim=1)  # [L, 4, 3] both CATH
dataset structures and AlphaFold2 predictions.

Key features:
- Virtual super node connected to all residues for global connectivity
- Exponential RBF spacing for distance features
- Proper handling of B-factors (PDB) and pLDDT scores (AlphaFold2)
- Data source indicators (AlphaFold2 vs PDB vs CATH)
"""

import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .cif_parser import parse_cif_backbone_auto

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rbf_cache'))
from rbf_lookup_manager import RBFLookupManager


def normalize_uncertainty(input_tensor: torch.Tensor, uncertainty_type: str = 'b_factor', geom_missing: Optional[torch.Tensor] = None, blur_uncertainty=False) -> torch.Tensor:
    """
    Normalizes uncertainty values (B-factors or pLDDT scores) within each protein using mean-center unit-norm + sigmoid.
    NMR structures (all-zero B-factors) are set to neutral uncertainty (0.5).

    Parameters:
    - input_tensor: PyTorch tensor of uncertainty values [L]
    - uncertainty_type: 'b_factor' for B-factors, 'plddt' for pLDDT scores
    - geom_missing: Optional boolean mask [L] indicating missing geometry nodes

    Returns:
    - normalized: PyTorch tensor [L] with normalized uncertainty values
    """
    # FAIL-FAST validation: Check input tensor type and properties
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"input_tensor must be a PyTorch tensor, got {type(input_tensor)}")

    if input_tensor.dim() != 1:
        raise ValueError(f"input_tensor must be 1D, got shape {input_tensor.shape}")

    if len(input_tensor) == 0:
        raise ValueError("input_tensor cannot be empty")

    # FAIL-FAST validation: Check uncertainty type
    if uncertainty_type not in ['b_factor', 'plddt']:
        raise ValueError(f"uncertainty_type must be 'b_factor' or 'plddt', got '{uncertainty_type}'")

    # FAIL-FAST validation: Check geom_missing mask
    if geom_missing is not None:
        if not isinstance(geom_missing, torch.Tensor):
            raise TypeError(f"geom_missing must be a PyTorch tensor, got {type(geom_missing)}")

        if geom_missing.dtype != torch.bool:
            raise TypeError(f"geom_missing must be boolean tensor, got {geom_missing.dtype}")

        if geom_missing.shape != input_tensor.shape:
            raise ValueError(f"geom_missing shape {geom_missing.shape} must match input_tensor shape {input_tensor.shape}")

    # Input validation for NaN/Inf with fail-fast on critical errors
    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        print(f"WARNING: NaN/Inf detected in {uncertainty_type} scores, setting to fallback values")

        # Count problematic values for validation
        nan_count = torch.isnan(input_tensor).sum().item()
        inf_count = torch.isinf(input_tensor).sum().item()
        total_bad = nan_count + inf_count

        # FAIL-FAST: If too many values are bad, something is seriously wrong
        if total_bad >= len(input_tensor):
            raise ValueError(f"All {len(input_tensor)} values in {uncertainty_type} scores are NaN/Inf - data is completely corrupted")

        # Replace NaN/Inf with reasonable defaults
        if uncertainty_type == 'plddt':
            input_tensor = torch.where(torch.isnan(input_tensor) | torch.isinf(input_tensor),
                                     torch.tensor(50.0, device=input_tensor.device), input_tensor)
        else:  # b_factor
            input_tensor = torch.where(torch.isnan(input_tensor) | torch.isinf(input_tensor),
                                     torch.tensor(20.0, device=input_tensor.device), input_tensor)

    # FAIL-FAST validation: Check for negative values where they shouldn't be
    if uncertainty_type == 'plddt' and (input_tensor < 0).any():
        raise ValueError(f"pLDDT scores cannot be negative, found min value: {input_tensor.min().item()}")

    # Create a mask for valid (non-missing) nodes
    if geom_missing is not None:
        valid_mask = ~geom_missing
    else:
        valid_mask = torch.ones_like(input_tensor, dtype=torch.bool)

    # FAIL-FAST validation: Ensure we have at least some valid data to work with
    valid_count = valid_mask.sum().item()
    if valid_count == 0:
        # Special case: no valid scores available, return 0.5 for all valid nodes, 0 for missing
        result = torch.full_like(input_tensor, 0.5, dtype=torch.float)
        if geom_missing is not None:
            result[geom_missing] = 0.0
        return result

    # Get valid scores for statistics calculation
    valid_scores = input_tensor[valid_mask]

    # FAIL-FAST validation: Check that valid scores are reasonable
    if uncertainty_type == 'plddt':
        if (valid_scores > 100).any():
            raise ValueError(f"pLDDT scores cannot exceed 100, found max value: {valid_scores.max().item()}")
    elif uncertainty_type == 'b_factor':
        if (valid_scores > 1000).any():
            print(f"WARNING: Unusually high B-factors detected (max: {valid_scores.max().item()})")

    # Check for NMR structures (all-zero B-factors) and handle specially
    if uncertainty_type == 'b_factor' and torch.all(valid_scores == 0):
        # NMR structure: set all to neutral uncertainty (0.5)
        result = torch.full_like(input_tensor, 0.5, dtype=torch.float)
        if geom_missing is not None:
            result[geom_missing] = 0.0
        return result

    # Calculate per-protein mean and std from valid scores only (within-protein normalization)
    mean = torch.mean(valid_scores)
    std = torch.std(valid_scores)

    # FAIL-FAST validation: Check computed statistics
    if torch.isnan(mean) or torch.isinf(mean):
        raise ValueError(f"Computed mean is invalid: {mean}")

    if torch.isnan(std) or torch.isinf(std):
        raise ValueError(f"Computed std is invalid: {std}")

    # Handle zero std case (all values are identical)
    if std == 0 or blur_uncertainty:
        # If all values are the same, return neutral uncertainty
        if hasattr(torch, '_debug_mode') and torch._debug_mode:
            print(f"    FALLBACK REASON: Zero standard deviation (all B-factors identical: {mean.item():.3f})")
        result = torch.full_like(input_tensor, 0.5, dtype=torch.float)
        if geom_missing is not None:
            result[geom_missing] = 0.0
        return result

    # Standard within-protein normalization
    norm = (input_tensor - mean) / std

    # Apply sigmoid
    sigmoid = torch.sigmoid(norm)

    # FAIL-FAST validation: Check sigmoid output
    if torch.isnan(sigmoid).any() or torch.isinf(sigmoid).any():
        raise ValueError(f"Sigmoid output contains NaN/Inf values")

    # Return based on uncertainty type
    if uncertainty_type == 'b_factor':
        normalized = 1 - sigmoid  # Higher B-factors = more uncertainty = lower values
    else:  # plddt
        normalized = sigmoid      # Higher pLDDT = more confidence = higher values

    # Set missing geometry nodes to 0
    if geom_missing is not None:
        normalized[geom_missing] = 0.0

    # FAIL-FAST validation: Check final output
    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
        raise ValueError(f"Final normalized output contains NaN/Inf values")

    if (normalized < 0).any() or (normalized > 1).any():
        raise ValueError(f"Normalized values must be in [0,1], got range [{normalized.min().item()}, {normalized.max().item()}]")

    return normalized

def _is_standard_amino_acid(aa):
    """
    Check if an amino acid is one of the 20 standard amino acids or 'X' (unknown).

    Args:
        aa: Single amino acid (1-letter or 3-letter code)

    Returns:
        bool: True if it's a standard amino acid or 'X', False otherwise
    """
    # Standard 20 amino acids (1-letter codes) + X for unknown
    standard_1letter = {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X'}

    # Standard 20 amino acids (3-letter codes) + XXX for unknown
    standard_3letter = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                       'THR', 'TRP', 'TYR', 'VAL', 'XXX'}

    if len(aa) == 1:
        return aa.upper() in standard_1letter
    elif len(aa) == 3:
        return aa.upper() in standard_3letter
    else:
        return False

def _normalize(x, eps=1e-8, dim=-1):
    """
    Normalizes a tensor along the specified dimension.

    Args:
        x: Input tensor
        eps: Small value to prevent division by zero
        dim: Dimension to normalize along

    Returns:
        Normalized tensor
    """
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)

def _rbf(dist, D_min=0.0, D_max=500.0, D_count=8, device='cpu', spacing='exponential'):
    """
    Converts distances to radial basis function (RBF) features.

    This function creates a smooth representation of distances by projecting
    them onto a set of basis functions. This is useful for neural networks
    as it provides a continuous representation of discrete distance values.

    Args:
        dist: Distance tensor
        D_min: Minimum distance for RBF centers
        D_max: Maximum distance for RBF centers
        D_count: Number of RBF centers
        device: Device to place tensors on
        spacing: 'linear' or 'exponential' spacing for RBF centers

    Returns:
        RBF features tensor
    """
    if spacing == 'linear':
        centers = torch.linspace(D_min, D_max, D_count, device=device)
    elif spacing == 'exponential':
        log_min = math.log(D_min + 1e-3)
        log_max = math.log(D_max + 1e-3)
        centers = torch.exp(torch.linspace(log_min, log_max, D_count, device=device))
    else:
        raise ValueError(f"Unsupported spacing type: {spacing}")

    width = (D_max - D_min) / D_count
    diff = dist.unsqueeze(-1) - centers
    return torch.exp(- (diff ** 2) / (2 * width**2))

def _dihedrals(X, eps=1e-7):
    """
    Computes backbone dihedral angles (phi, psi, omega) from coordinates.

    This is the proper GVP-GNN dihedral computation that returns 6 features:
    cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)

    Args:
        X: Backbone coordinates tensor [L, 4, 3] (N, CA, C, O)
        eps: Small value to prevent numerical issues

    Returns:
        Dihedral angles tensor [L, 6] (cos/sin of phi, psi, omega)
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    # Reshape to get all N, CA, C atoms in sequence
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.linalg.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = _normalize(torch.linalg.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle (cos, sin for each angle)
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)  # [L, 6]
    return D_features

def _orientations(X):
    """
    Computes local coordinate frames for each residue.

    Creates a local coordinate system for each residue based on the
    CA coordinates. This is the exact GVP-GNN implementation.

    Args:
        X: CA coordinates tensor [L, 3]

    Returns:
        Orientation tensors [L, 2, 3] (forward and backward directions)
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _positional_embeddings(edge_index, num_embeddings=16, period_range=[2, 1000], device='cpu'):
    """
    Computes positional embeddings for edges based on sequence distance.

    Args:
        edge_index: Edge indices [2, E]
        num_embeddings: Number of embedding dimensions
        period_range: Range for sinusoidal periods
        device: Device to create tensors on

    Returns:
        Positional embeddings [E, num_embeddings]
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def _sidechains(X):
    """
    Computes an imputed direction vector for the C-beta atom of the sidechain.
    This is calculated from the backbone atoms (N, C-alpha, C) using tetrahedral
    geometry, providing a consistent orientation for the sidechain.
    Returns a tensor of shape [L, 1, 3].
    """
    N, origin, C = X[:,0], X[:,1], X[:,2]
    c = _normalize(C - origin)
    n = _normalize(N - origin)
    bis = _normalize(c + n)
    perp = _normalize(torch.linalg.cross(c, n, dim=-1))
    v = -bis * math.sqrt(1/3) - perp * math.sqrt(2/3)
    return v.unsqueeze(1)

# NaN-safe versions for smoke_test mode

def _normalize_safe(x, eps=1e-8, dim=-1, smoke_test=False):
    """NaN-safe version of _normalize"""
    if smoke_test:
        if torch.isnan(x).any():
            print(f"WARNING: NaN detected in normalize input tensor", flush=True)
            # Replace NaN with zeros
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        if torch.isinf(x).any():
            print(f"WARNING: Inf detected in normalize input tensor", flush=True)
            x = torch.clamp(x, -1000, 1000)

    norm = torch.norm(x, dim=dim, keepdim=True) + eps
    result = x / norm

    if smoke_test:
        if torch.isnan(result).any():
            print(f"WARNING: NaN detected in normalize output tensor", flush=True)
            # Replace NaN with unit vector in first dimension
            nan_mask = torch.isnan(result)
            if nan_mask.any():
                unit_vec = torch.zeros_like(result)
                unit_vec[..., 0] = 1.0  # Set first component to 1
                result = torch.where(nan_mask, unit_vec, result)
        if torch.isinf(result).any():
            print(f"WARNING: Inf detected in normalize output tensor", flush=True)
            result = torch.clamp(result, -1, 1)

    return result

def _rbf_safe(dist, D_min=0.0, D_max=500.0, D_count=8, device='cpu', spacing='exponential', smoke_test=False):
    """NaN-safe version of _rbf"""
    if smoke_test:
        if torch.isnan(dist).any():
            print(f"WARNING: NaN detected in RBF input distances", flush=True)
            # Replace NaN distances with a reasonable default (10.0 Angstrom)
            dist = torch.where(torch.isnan(dist), torch.tensor(10.0, device=dist.device), dist)
        if torch.isinf(dist).any():
            print(f"WARNING: Inf detected in RBF input distances", flush=True)
            dist = torch.clamp(dist, D_min, D_max)
        if (dist < 0).any():
            print(f"WARNING: Negative distances detected in RBF input", flush=True)
            dist = torch.abs(dist)

    try:
        result = _rbf(dist, D_min, D_max, D_count, device, spacing)

        if smoke_test:
            if torch.isnan(result).any():
                print(f"WARNING: NaN detected in RBF output", flush=True)
                # Replace NaN with zeros
                result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
            if torch.isinf(result).any():
                print(f"WARNING: Inf detected in RBF output", flush=True)
                result = torch.clamp(result, -10, 10)

        return result
    except Exception as e:
        if smoke_test:
            print(f"ERROR in _rbf: {e}", flush=True)
        raise

def _dihedrals_safe(X, eps=1e-7, smoke_test=False):
    """NaN-safe version of _dihedrals"""
    if smoke_test:
        if torch.isnan(X).any():
            print(f"WARNING: NaN detected in dihedral input coordinates", flush=True)
        if torch.isinf(X).any():
            print(f"WARNING: Inf detected in dihedral input coordinates", flush=True)

    try:
        result = _dihedrals(X, eps)

        if smoke_test:
            if torch.isnan(result).any():
                print(f"WARNING: NaN detected in dihedral output", flush=True)
            if torch.isinf(result).any():
                print(f"WARNING: Inf detected in dihedral output", flush=True)

        return result
    except Exception as e:
        if smoke_test:
            print(f"ERROR in _dihedrals: {e}", flush=True)
        raise

def _orientations_safe(X, smoke_test=False):
    """NaN-safe version of _orientations"""
    if smoke_test:
        if torch.isnan(X).any():
            print(f"WARNING: NaN detected in orientation input coordinates", flush=True)
        if torch.isinf(X).any():
            print(f"WARNING: Inf detected in orientation input coordinates", flush=True)

    try:
        result = _orientations(X)

        if smoke_test:
            if torch.isnan(result).any():
                print(f"WARNING: NaN detected in orientation output", flush=True)
            if torch.isinf(result).any():
                print(f"WARNING: Inf detected in orientation output", flush=True)

        return result
    except Exception as e:
        if smoke_test:
            print(f"ERROR in _orientations: {e}", flush=True)
        raise

def _sidechains_safe(X, smoke_test=False):
    """NaN-safe version of _sidechains"""
    if smoke_test:
        if torch.isnan(X).any():
            print(f"WARNING: NaN detected in sidechain input coordinates", flush=True)
        if torch.isinf(X).any():
            print(f"WARNING: Inf detected in sidechain input coordinates", flush=True)

    try:
        result = _sidechains(X)

        if smoke_test:
            if torch.isnan(result).any():
                print(f"WARNING: NaN detected in sidechain output", flush=True)
            if torch.isinf(result).any():
                print(f"WARNING: Inf detected in sidechain output", flush=True)

        return result
    except Exception as e:
        if smoke_test:
            print(f"ERROR in _sidechains: {e}", flush=True)
        raise

# GraphBuilder

class GraphBuilder:
    """
    Builds graph representations from protein backbone coordinates.

    This class converts protein structures into graph representations suitable
    for geometric deep learning. It creates k-nearest neighbor graphs with
    rich edge features including 3D distances, sequence distances, and
    geometric features like dihedral angles and orientations.

    The graph can be built from either CIF files or pre-processed coordinate
    tensors, supporting both CATH dataset structures and AlphaFold2 predictions.

    Key features:
    - Virtual super node connected to all residues
    - Exponential RBF spacing for distance features
    - Proper handling of B-factors and pLDDT scores
    - Data source indicators
    """

    def __init__(self, k=None, k_farthest=None, k_random=None, max_edge_dist=None,
                 num_rbf_3d=None, num_rbf_seq=None,
                 verbose=False, smoke_test=False,
                 use_virtual_node=True, no_source_indicator=False,
                 # RBF distance range parameters
                 rbf_3d_min=2.0, rbf_3d_max=350.0, rbf_3d_spacing='exponential',
                 # Structure noise parameters
                 structure_noise_mag_std=0.0,
                 time_based_struct_noise='fixed',
                 uncertainty_struct_noise_scaling=False,
                 rigid_residue_noise=False,
                 # Uncertainty processing parameters
                 blur_uncertainty=False,
                 # Node degree feature
                 include_node_degree=False):
        """
        Initialize the GraphBuilder.

        Args:
            k: Number of nearest neighbors per node (REQUIRED - no default unless max_edge_dist is used)
            k_farthest: Number of farthest neighbors to include (REQUIRED - no default unless max_edge_dist is used)
            k_random: Number of random neighbors to include (REQUIRED - no default unless max_edge_dist is used)
            max_edge_dist: Maximum distance cutoff (Angstroms) for edge creation. If set, overrides k, k_farthest, k_random. Max 80 neighbors per node for safety.
            num_rbf_3d: Number of RBF features for 3D distances
            num_rbf_seq: Number of RBF features for sequence distances
            verbose: Whether to print progress information
            smoke_test: Whether to enable extensive NaN/error checking
            use_virtual_node: Whether to include a virtual super node in the graph
            no_source_indicator: If True, use [1.0, 1.0] for all data sources instead of distinctive indicators
            rbf_3d_min: Minimum distance (Angstroms) for 3D RBF centers (default: 2.0)
            rbf_3d_max: Maximum distance (Angstroms) for 3D RBF centers (default: 350.0)
            rbf_3d_spacing: Spacing type for 3D RBF centers - 'linear' or 'exponential' (default: exponential)
            structure_noise_mag_std: Standard deviation for Gaussian noise on coordinates (0.0 = disabled)
            time_based_struct_noise: Time-based noise scaling ('fixed', 'increasing', 'decreasing')
            uncertainty_struct_noise_scaling: Whether to scale noise by uncertainty (1 - normalized_uncertainty)
            rigid_residue_noise: If True, apply the same displacement to all 4 backbone atoms per residue
                                 (rigid translation), preserving intra-residue geometry (bond lengths,
                                 angles, Cβ direction). If False (default), each atom gets independent noise.
            blur_uncertainty: Whether to blur uncertainty values during normalization
        """
        # Validate edge building parameters - fail-fast approach
        if max_edge_dist is not None:
            if max_edge_dist <= 0:
                raise ValueError(f"max_edge_dist must be positive, got {max_edge_dist}")

            # When max_edge_dist is set, k parameters are ignored
            if k is not None or k_farthest is not None or k_random is not None:
                if verbose:
                    print(f"max_edge_dist={max_edge_dist} specified: ignoring k={k}, k_farthest={k_farthest}, k_random={k_random}")

            # Set k parameters to None to indicate distance-based mode
            k = None
            k_farthest = None
            k_random = None

        else:
            # Original k-neighbor mode - validate that critical parameters are provided
            if k is None:
                raise ValueError("k (number of nearest neighbors) must be explicitly provided when max_edge_dist is not used")
            if k_farthest is None:
                raise ValueError("k_farthest (number of farthest neighbors) must be explicitly provided when max_edge_dist is not used")
            if k_random is None:
                raise ValueError("k_random (number of random neighbors) must be explicitly provided when max_edge_dist is not used")

        self.k = k
        self.k_farthest = k_farthest
        self.k_random = k_random
        self.max_edge_dist = max_edge_dist
        self.num_rbf_3d = num_rbf_3d
        self.num_rbf_seq = num_rbf_seq
        self.verbose = verbose
        self.smoke_test = smoke_test
        self.use_virtual_node = use_virtual_node
        self.no_source_indicator = no_source_indicator

        # Store RBF parameters
        self.rbf_3d_min = rbf_3d_min
        self.rbf_3d_max = rbf_3d_max
        self.rbf_3d_spacing = rbf_3d_spacing

        # Store RBF parameters
        self.rbf_3d_min = rbf_3d_min
        self.rbf_3d_max = rbf_3d_max
        self.rbf_3d_spacing = rbf_3d_spacing

        # Print edge building mode for clarity
        if self.max_edge_dist is not None:
            if verbose:
                print(f"GraphBuilder: Distance-based edge building (max_edge_dist={self.max_edge_dist}Å, max 80 neighbors per node)")
        else:
            if verbose:
                print(f"GraphBuilder: K-neighbor based edge building (k={self.k}, k_farthest={self.k_farthest}, k_random={self.k_random})")
                print(f"GraphBuilder: RBF parameters (min={self.rbf_3d_min}, max={self.rbf_3d_max}, spacing={self.rbf_3d_spacing})")

        # Structure noise parameters
        self.structure_noise_mag_std = structure_noise_mag_std
        self.time_based_struct_noise = time_based_struct_noise
        self.uncertainty_struct_noise_scaling = uncertainty_struct_noise_scaling
        self.rigid_residue_noise = rigid_residue_noise

        # Uncertainty processing parameters
        self.blur_uncertainty = blur_uncertainty

        # Node degree feature flag
        self.include_node_degree = include_node_degree

        # Initialize RBF lookup manager with configurable parameters
        self.rbf_manager = RBFLookupManager(
            verbose=True,  # Enable verbose output to see which RBF files are used
            rbf_3d_min=rbf_3d_min,
            rbf_3d_max=rbf_3d_max,
            rbf_3d_spacing=rbf_3d_spacing
        )

        # Don't load tables here - they'll be loaded lazily when needed
        # This prevents every DataLoader worker from loading tables during initialization

    def update_noise_params(self, structure_noise_mag_std=None, time_based_struct_noise=None,
                           uncertainty_struct_noise_scaling=None, rigid_residue_noise=None):
        """
        Update noise parameters dynamically. This allows the training loop to
        modify noise settings without recreating the GraphBuilder.

        Args:
            structure_noise_mag_std: New noise magnitude (None = no change)
            time_based_struct_noise: New time-based scaling (None = no change)
            uncertainty_struct_noise_scaling: New uncertainty scaling (None = no change)
            rigid_residue_noise: New rigid noise mode (None = no change)
        """
        if structure_noise_mag_std is not None:
            self.structure_noise_mag_std = structure_noise_mag_std
        if time_based_struct_noise is not None:
            self.time_based_struct_noise = time_based_struct_noise
        if rigid_residue_noise is not None:
            self.rigid_residue_noise = rigid_residue_noise
        if uncertainty_struct_noise_scaling is not None:
            self.uncertainty_struct_noise_scaling = uncertainty_struct_noise_scaling

    def _add_coordinate_noise(self, coords4: torch.Tensor, scores_norm: torch.Tensor,
                             geom_missing: torch.Tensor, t: Optional[float] = None) -> torch.Tensor:
        """
        Add Gaussian noise to coordinate tensors based on configuration.

        Args:
            coords4: Backbone coordinates [L, 4, 3] (N, CA, C, O)
            scores_norm: Normalized uncertainty scores [L] (range 0-1)
            geom_missing: Boolean mask [L] indicating residues with missing geometry
            t: Current time step for time-based noise scaling (None for sampling)

        Returns:
            Noisy coordinates [L, 4, 3]
        """
        # Skip noise addition if disabled
        if self.structure_noise_mag_std <= 0:
            return coords4

        device = coords4.device
        L = coords4.shape[0]

        # Calculate time coefficient
        if self.time_based_struct_noise == 'increasing' and t is not None:
            time_coeff = torch.sqrt(torch.tensor(t, device=device)) + 0.1
        elif self.time_based_struct_noise == 'decreasing' and t is not None:
            time_coeff = torch.sqrt(torch.clamp(torch.tensor(8.0 - t, device=device), 0.0)) + 0.1
        else:  # 'fixed' or sampling mode (t=None)
            time_coeff = 1.0

        # Calculate uncertainty coefficient
        if self.uncertainty_struct_noise_scaling:
            # More flexible parts (lower uncertainty scores) get more noise
            uncert_coeff = 1.0 - scores_norm  # [L]
            uncert_coeff = uncert_coeff.unsqueeze(1).unsqueeze(2)  # [L, 1, 1] for broadcasting
            if self.verbose:
                print(f"  scores_norm range: {scores_norm.min():.3f}-{scores_norm.max():.3f}")
                print(f"  uncert_coeff range: {uncert_coeff.min():.3f}-{uncert_coeff.max():.3f}")
        else:
            uncert_coeff = 1.0

        # Debug logging for structure noise verification (verbose mode only)
        if self.verbose and self.structure_noise_mag_std > 0:
            print(f"[STRUCTURE NOISE] t={t}, time_coeff={time_coeff:.3f}, noise_type={self.time_based_struct_noise}, uncert_scaling={self.uncertainty_struct_noise_scaling}")

        # Generate Gaussian noise: [L, 4, 3]
        if self.rigid_residue_noise:
            # Rigid translation: same displacement for all 4 backbone atoms per residue.
            # Preserves intra-residue geometry (bond lengths, angles, Cβ direction).
            noise_ca = torch.randn(L, 1, 3, device=device, dtype=coords4.dtype) * self.structure_noise_mag_std
            noise = noise_ca.expand(-1, 4, -1).contiguous()
        else:
            noise = torch.randn_like(coords4) * self.structure_noise_mag_std

        # Apply coefficients: Noise = time_coeff * uncert_coeff * Normal(0, structure_noise_mag_std)
        scaled_noise = time_coeff * uncert_coeff * noise

        # Don't add noise to residues with missing geometry (preserve placeholder coordinates)
        scaled_noise[geom_missing] = 0.0

        # Add noise to coordinates
        noisy_coords = coords4 + scaled_noise

        if self.verbose and self.structure_noise_mag_std > 0:
            noise_magnitude = torch.norm(scaled_noise).item()
            print(f"Added coordinate noise: std={self.structure_noise_mag_std}, "
                  f"time_coeff={time_coeff}, magnitude={noise_magnitude:.3f}")

        return noisy_coords

    def preload_rbf_tables(self, device='cpu'):
        """
        Pre-load RBF tables into memory cache. Call this once at the start of training
        to avoid loading messages during the first graph construction.

        Args:
            device: Device to load tables to ('cpu' or 'cuda:0', etc.)
        """
        self.rbf_manager.load_3d_rbf_table(device)
        self.rbf_manager.load_seq_rbf_table(device)

    def build(self, cif_path: str, time_param: float) -> Data:
        """
        Builds a graph from a CIF file.

        This method automatically detects whether the CIF file is from
        AlphaFold2 or a traditional PDB source and parses accordingly.

        Args:
            cif_path: Path to the CIF file
            time_param: Time parameter for time-dependent coordinate noise (None = no time dependence)

        Returns:
            PyTorch Geometric Data object containing the graph
        """
        # Parse the CIF file with automatic source detection
        coords, scores, residue_types, source = parse_cif_backbone_auto(cif_path)

        # Build graph from the parsed data
        return self.build_from_tensors(coords, scores, source, cif_path, filtered_seq=''.join(residue_types), t=time_param)

    def build_from_dict(self, entry: dict, time_param: float, af_filter_mode = True,
                        context_mask: Optional[torch.Tensor] = None) -> Data:
        """
        Builds a graph from a dictionary entry containing pre-parsed protein data.

        Args:
            entry: Dictionary with keys like 'coords', 'seq', 'name', etc.
                   Optionally includes '_precomputed_coords' and '_precomputed_dist' for batch optimization
            context_mask: Optional boolean tensor [L]. True = context/fixed chain residue.
                          Forwarded to build_from_tensors; see its docstring for details.

        Returns:
            PyTorch Geometric Data object
        """
        # 'seq', 'coords', 'num_chains', 'name', 'CATH', 'b_factors', 'dssp']
        if 'plddt' in entry:  # AF2 data
            seq_len = len(entry['seq'])

            if seq_len % 50 != 0 and seq_len > 20 and af_filter_mode:
                # Check if coords is a dict with atom keys (C, CA, N, O)
                for atom_key in entry['coords']:
                        entry['coords'][atom_key] = entry['coords'][atom_key][3:-3]

                entry['seq'] = entry['seq'][3:-3]
                entry['plddt'] = entry['plddt'][3:-3]
                if 'dssp' in entry:
                    entry['dssp'] = entry['dssp'][3:-3]

        # Check for precomputed coordinates from batch processing
        if '_precomputed_coords' in entry:
            coords4 = entry['_precomputed_coords']
            if self.smoke_test:
                print(f"Using precomputed coordinates for {entry.get('name', 'unknown')}")
        else:
            # Convert numpy arrays to torch tensors (original logic)
            coords_np = entry['coords']
            if isinstance(coords_np, dict) and 'N' in coords_np:
                # Get sequence length for validation
                seq_len = len(entry['seq'])

                # Check that all coordinate arrays have the expected length
                coord_lens = {atom: coords_np[atom].shape[0] for atom in ['N', 'CA', 'C', 'O'] if atom in coords_np}
                if not all(length == seq_len for length in coord_lens.values()):
                    raise ValueError(f"Coordinate array lengths {coord_lens} don't match sequence length {seq_len}")

                # Reconstruct full 4-atom backbone
                coords4 = torch.tensor(np.stack([
                    coords_np.get(atom, coords_np['CA'])
                    for atom in ['N', 'CA', 'C', 'O']
                ], axis=1), dtype=torch.float32)
            else:
                coords4 = torch.tensor(coords_np, dtype=torch.float32)

        # NEW APPROACH: Masked-geometry instead of filtering
        # Keep all sequence positions but mark those with missing/invalid geometry
        original_seq = entry['seq']
        L = coords4.shape[0]

        # Check for NaN/Inf coordinates
        has_coord_issues = torch.isnan(coords4).any() or torch.isinf(coords4).any()

        # Check for non-standard amino acids
        seq_list = list(original_seq)
        has_seq_issues = any(not _is_standard_amino_acid(aa) for aa in seq_list)

        # Create geometry missing mask
        geom_missing = torch.zeros(L, dtype=torch.bool)

        if has_coord_issues or has_seq_issues:
            if self.smoke_test:
                issues = []
                if has_coord_issues:
                    issues.append("NaN/Inf coordinates")
                if has_seq_issues:
                    issues.append("non-standard amino acids")
                print(f"WARNING: {' and '.join(issues)} detected in {entry.get('name', 'unknown')}, masking problematic residues")

            # Mark residues with coordinate issues
            for atom_idx in range(4):  # N, CA, C, O
                atom_coords = coords4[:, atom_idx, :]  # [L, 3]
                atom_has_nan_inf = torch.isnan(atom_coords).any(dim=1) | torch.isinf(atom_coords).any(dim=1)
                geom_missing = geom_missing | atom_has_nan_inf

            # Mark residues with non-standard amino acids (excluding 'X' which is now supported)
            for i, aa in enumerate(seq_list):
                if not _is_standard_amino_acid(aa):
                    geom_missing[i] = True
                    if self.smoke_test:
                        print(f"  Non-standard amino acid '{aa}' at position {i}, masking geometry")

            # Count how many residues have missing geometry
            num_masked = geom_missing.sum().item()
            if self.smoke_test and num_masked > 0:
                print(f"Masking geometry for {num_masked} out of {L} residues")

            # Replace problematic coordinates with zeros (placeholder)
            # This preserves sequence length while indicating missing geometry
            coords4[geom_missing] = 0.0

            # For non-standard amino acids (except 'X'), replace with 'G' (glycine) in sequence
            # 'X' is preserved as it's now part of our extended alphabet
            for i, aa in enumerate(seq_list):
                if not _is_standard_amino_acid(aa):
                    seq_list[i] = 'G'  # Replace with glycine as safe fallback
                    if self.smoke_test:
                        print(f"  Replaced non-standard amino acid '{aa}' with 'G' at position {i}")
                elif aa.upper() == 'X':
                    # Ensure 'X' is in consistent format
                    seq_list[i] = 'X'

            # Update sequence with sanitized version
            filtered_seq = ''.join(seq_list)
            entry = entry.copy()
            # Preserve original sequence for ground truth
            if 'original_seq' not in entry:
                entry['original_seq'] = entry['seq']  # Save original before overwriting
            entry['seq'] = filtered_seq  # Keep filtered for graph building

            # If all residues have missing geometry, skip this protein
            if geom_missing.all():
                if self.smoke_test:
                    print(f"WARNING: All residues have missing geometry in {entry.get('name', 'unknown')}, skipping")
                raise ValueError(f"All residues have missing geometry in {entry.get('name', 'unknown')}")

        # Store the geometry missing mask for use in node features and training
        # This will be passed to build_from_tensors to include in the graph
        filtered_seq = entry.get('seq', original_seq)

        # Extract B-factor scores from entry data if available
        if 'b_factors' in entry:
            # CATH dataset with B-factor data
            b_factors = entry['b_factors']
            if isinstance(b_factors, np.ndarray):
                scores = torch.from_numpy(b_factors).float()
            else:
                scores = torch.tensor(b_factors, dtype=torch.float32)

            # Verify length matches coordinates
            if len(scores) != coords4.shape[0]:
                print(f"WARNING: B-factor length mismatch for {entry.get('name', 'unknown')}: "
                      f"coords={coords4.shape[0]}, b_factors={len(scores)}, using dummy values")
                raise Exception('shape mismatch with b factors')
                #scores = torch.ones(coords4.shape[0], dtype=torch.float32)
        elif 'plddt' in entry:
            # AlphaFold2 dataset with pLDDT data
            plddt_scores = entry['plddt']
            if isinstance(plddt_scores, np.ndarray):
                scores = torch.from_numpy(plddt_scores).float()
            else:
                scores = torch.tensor(plddt_scores, dtype=torch.float32)

            # Verify length matches coordinates
            if len(scores) != coords4.shape[0]:
                print(f"WARNING: pLDDT length mismatch for {entry.get('name', 'unknown')}: "
                      f"coords={coords4.shape[0]}, plddt={len(scores)}, using dummy values")
                raise Exception('shape mismatch with plddt in graph builder')
                #scores = torch.ones(coords4.shape[0], dtype=torch.float32)
        else:
            # Fallback: dummy scores if no uncertainty data present
            scores = torch.ones(coords4.shape[0], dtype=torch.float32)
            if self.smoke_test:
                print(f"WARNING: No B-factor or pLDDT data found for {entry.get('name', 'unknown')}, using dummy scores")

        # Store the filtered sequence for dataset use
        filtered_seq = entry.get('seq', '')

        # Determine source type based on available data
        if 'plddt' in entry:
            source = 'alphafold2'
        elif 'b_factors' in entry:
            source = 'pdb'  # CATH dataset with B-factors treated as PDB source
        else:
            source = 'pdb'  # Default to PDB source instead of CATH

        # Check for precomputed distance matrix from batch processing
        precomputed_dist = entry.get('_precomputed_dist', None)

        # Apply coordinate noise if enabled (before building the graph)
        if self.structure_noise_mag_std > 0:
            # Normalize scores before applying noise (same as in build_from_tensors)
            uncertainty_type = 'plddt' if 'plddt' in entry else 'b_factor'
            scores_norm = normalize_uncertainty(scores, uncertainty_type, geom_missing, blur_uncertainty=self.blur_uncertainty)

            # Apply noise with normalized scores and time parameter
            coords4 = self._add_coordinate_noise(coords4, scores_norm, geom_missing, t=time_param)

            if self.smoke_test:
                time_info = f", t={time_param:.3f}" if time_param is not None else ""
                print(f"Applied coordinate noise: std={self.structure_noise_mag_std}{time_info}")

        result = self.build_from_tensors(coords4, scores, source=source, cif_path=entry.get('name', 'unknown'),
                                       geom_missing=geom_missing, filtered_seq=filtered_seq,
                                       precomputed_dist=precomputed_dist,
                                       context_mask=context_mask)

        return result
    def build_from_tensors(self,
                          coords4: torch.Tensor,
                          scores: torch.Tensor,
                          source: str,
                          cif_path: str = "unknown",
                          geom_missing: Optional[torch.Tensor] = None,
                          filtered_seq: str = "",
                          precomputed_dist: Optional[torch.Tensor] = None,
                          t: Optional[float] = None,
                          context_mask: Optional[torch.Tensor] = None) -> Data:
        """
        Builds a graph from pre-processed coordinate tensors.

        This method creates a graph representation from backbone coordinates
        and either B-factors (for PDB structures) or pLDDT scores (for
        AlphaFold2 predictions).

        Args:
            coords4: Backbone coordinates [L, 4, 3] (N, CA, C, O)
            scores: B-factors (PDB) or pLDDT scores (AlphaFold2) [L]
            source: Data source ('pdb' or 'alphafold2')
            cif_path: Original file path for reference
            geom_missing: Boolean mask [L] indicating residues with missing geometry
            filtered_seq: Filtered sequence string for reference
            precomputed_dist: Optional precomputed distance matrix [L, L]
            t: Current time step for time-based noise scaling (None for sampling)
            context_mask: Optional boolean tensor [L]. True = context/fixed chain residue
                          (e.g. GFP chain in a multi-chain conditioning setup). These
                          residues participate in edge construction and message passing
                          normally, but are excluded from the packing/burial score
                          computation so they do not artificially inflate the burial
                          signal of target residues. Stored on the returned Data object.

        Returns:
            PyTorch Geometric Data object containing the graph
        """
        # FAIL-FAST validation: Check all input parameters
        if not isinstance(coords4, torch.Tensor):
            raise TypeError(f"coords4 must be a PyTorch tensor, got {type(coords4)}")

        if coords4.dim() != 3:
            raise ValueError(f"coords4 must be 3D tensor [L, 4, 3], got shape {coords4.shape}")

        if coords4.shape[1] != 4:
            raise ValueError(f"coords4 must have 4 atoms (N, CA, C, O), got {coords4.shape[1]} atoms")

        if coords4.shape[2] != 3:
            raise ValueError(f"coords4 must have 3D coordinates, got {coords4.shape[2]}D")

        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"scores must be a PyTorch tensor, got {type(scores)}")

        if source not in ['pdb', 'alphafold2']:
            raise ValueError(f"source must be 'pdb' or 'alphafold2', got '{source}'")

        device = coords4.device
        L = coords4.shape[0]

        # FAIL-FAST validation: Check tensor shapes match
        if scores.shape[0] != L:
            raise ValueError(f"scores length {scores.shape[0]} must match coords4 length {L}")

        # Initialize geometry missing mask if not provided
        if geom_missing is None:
            geom_missing = torch.zeros(L, dtype=torch.bool, device=device)

        # FAIL-FAST validation: Check geom_missing tensor
        if not isinstance(geom_missing, torch.Tensor):
            raise TypeError(f"geom_missing must be a PyTorch tensor, got {type(geom_missing)}")

        if geom_missing.dtype != torch.bool:
            raise TypeError(f"geom_missing must be boolean tensor, got {geom_missing.dtype}")

        if geom_missing.shape[0] != L:
            raise ValueError(f"geom_missing length {geom_missing.shape[0]} must match coords4 length {L}")

        # FAIL-FAST validation: Check for completely empty protein
        if L == 0:
            raise ValueError("Cannot build graph from empty protein (L=0)")

        # FAIL-FAST validation: Check for reasonable protein size
        if L > 10000:
            print(f"WARNING: Very large protein with {L} residues, this may cause memory issues")

        # Validate coordinates for NaN/Inf values after initial filtering
        # Most NaN/Inf should already be removed by build_from_dict, but check again for safety
        if torch.isnan(coords4).any() or torch.isinf(coords4).any():
            if self.smoke_test:
                print(f"WARNING: Unexpected NaN/Inf coordinates in {cif_path} after initial filtering")
            # Replace remaining NaN/Inf with reasonable default values as fallback
            # Use CA coordinate as fallback for missing atoms
            ca_coords = coords4[:, 1, :]  # CA is at index 1

            # For each residue, replace NaN/Inf atoms with CA coordinate
            for i in range(4):  # N, CA, C, O
                mask = torch.isnan(coords4[:, i, :]) | torch.isinf(coords4[:, i, :])
                if mask.any():
                    coords4[:, i, :] = torch.where(mask, ca_coords, coords4[:, i, :])

            # If CA itself has issues, set to origin
            ca_mask = torch.isnan(ca_coords) | torch.isinf(ca_coords)
            if ca_mask.any():
                coords4[:, 1, :] = torch.where(ca_mask, torch.zeros_like(ca_coords), ca_coords)
                # Update other atoms to match cleaned CA
                for i in [0, 2, 3]:
                    coords4[:, i, :] = torch.where(ca_mask, torch.zeros_like(ca_coords), coords4[:, i, :])

        # FAIL-FAST validation: After cleaning, ensure coordinates are still reasonable
        if torch.isnan(coords4).any():
            raise ValueError(f"NaN coordinates still detected in {cif_path} after all cleanup attempts")

        if torch.isinf(coords4).any():
            raise ValueError(f"Inf coordinates detected in {cif_path} after cleanup")

        if self.smoke_test:
            if torch.isnan(coords4).any():
                print(f"WARNING: NaN coordinates still detected in {cif_path} after all cleanup attempts")
            if torch.isinf(coords4).any():
                print(f"WARNING: Inf coordinates detected in {cif_path}")

        if self.verbose:
            print(f"Building graph for {cif_path} ({source}) with {L} residues")

        # Calculate normalized uncertainty scores early for noise scaling
        # This is needed before adding coordinate noise
        if source == 'alphafold2':
            uncertainty_type = 'plddt'
        else:  # pdb or cath - both use B-factors
            uncertainty_type = 'b_factor'

        try:
            scores_norm = normalize_uncertainty(scores, uncertainty_type, geom_missing, blur_uncertainty=self.blur_uncertainty)
        except Exception as e:
            raise RuntimeError(f"Failed to normalize {uncertainty_type} scores for {source} data: {e}")

        # Note: Coordinate noise is applied in build_from_dict before calling this method
        # No noise application here to avoid duplicate noise

        # Extract CA coordinates for distance calculations (using potentially noisy coordinates)
        CA = coords4[:, 1, :]  # [L, 3]

        # Use precomputed distances if available (batch optimization)
        if precomputed_dist is not None:
            if self.smoke_test:
                print(f"Using precomputed distance matrix for efficiency")
            dist = precomputed_dist
            if dist.shape != (L, L):
                raise ValueError(f"Precomputed distance matrix shape {dist.shape} doesn't match coordinates {(L, L)}")
        else:
            # Compute pairwise distances
            dist = torch.cdist(CA, CA)  # [L, L]

        # For residues with missing geometry, set distances to a large value (1000.0)
        # This prevents them from being selected as nearest neighbors based on geometry
        for i in range(L):
            if geom_missing[i]:
                dist[i, :] = 1000.0  # Distance FROM missing geometry node
                dist[:, i] = 1000.0  # Distance TO missing geometry node
                dist[i, i] = 0.0     # Self-distance should be 0

        # Create edge indices including virtual node and sequence neighbors
        edge_index = self._create_edges_with_virtual(dist, geom_missing)

        # Compute edge features
        edge_s, edge_v = self._compute_edge_features(coords4, edge_index, dist, geom_missing)

        # Compute node features including virtual node and geometry missing flag
        # Pass the already-computed scores_norm to avoid redundant calculation
        node_s, node_v = self._compute_node_features(
            coords4, scores_norm, source, geom_missing,
            edge_index=edge_index, seq=filtered_seq,
            context_mask=context_mask,
        )

        # Create PyTorch Geometric Data object, adjust num_nodes based on virtual node usage
        data = Data(
            x_s=node_s,           # Scalar node features [L+1, F] or [L, F]
            x_v=node_v,           # Vector node features [L+1, 3, 3] or [L, 3, 3]
            edge_index=edge_index, # Edge connectivity [2, E]
            edge_s=edge_s,        # Scalar edge features [E, F]
            edge_v=edge_v,        # Vector edge features [E, 1, 3]
            pos=coords4[:, 1, :], # CA coordinates for positions [L, 3]
            num_nodes=L + 1 if self.use_virtual_node else L,
            geom_missing=geom_missing,  # Store geometry missing mask for training
            filtered_seq=filtered_seq,  # Store filtered sequence for masking (NEW)
            source=source,         # Data source for reference
            cif_path=cif_path,     # Original file path
            _original_coords=coords4.clone(),  # Store original coordinates for noise application
            context_mask=context_mask,  # Optional: marks context/fixed chain residues
        )
        # Record virtual node usage for downstream processing
        data.use_virtual_node = self.use_virtual_node
        return data

    def _create_distance_based_edges(self, dist: torch.Tensor, geom_missing: torch.Tensor) -> torch.Tensor:
        """
        Creates edges based on distance cutoff instead of fixed k-neighbors.

        All residue pairs within max_edge_dist are connected, with a safety limit
        of 80 neighbors per residue (taking the nearest 80 if more exist).

        Args:
            dist: Pairwise distance matrix [L, L]
            geom_missing: Boolean mask [L] indicating residues with missing geometry

        Returns:
            Edge index tensor [2, E] including sequence edges for missing geometry
        """
        L = dist.shape[0]
        device = dist.device

        if self.verbose:
            print(f"Building distance-based edges with cutoff {self.max_edge_dist}Å (max 80 neighbors per residue)")

        # 1. Distance-based edges (exclude self-connections)
        # Create mask for edges within distance cutoff
        dist_mask = (dist <= self.max_edge_dist) & (dist > 0)  # Exclude self (dist=0)

        # Get all edge candidates
        edge_candidates = dist_mask.nonzero()  # [num_edges, 2]

        if edge_candidates.numel() == 0:
            # No edges found - this should be very rare with reasonable cutoffs
            ei_dist = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_count_stats = torch.zeros(L, dtype=torch.long, device=device)
        else:
            # Apply safety limit: max 80 neighbors per residue
            MAX_NEIGHBORS = 80

            # Group edges by source residue and apply limit
            edge_list = []
            edge_count_stats = torch.zeros(L, dtype=torch.long, device=device)

            for i in range(L):
                # Find all neighbors within distance for residue i
                neighbors_mask = dist_mask[i]
                neighbor_indices = neighbors_mask.nonzero().flatten()

                if len(neighbor_indices) > MAX_NEIGHBORS:
                    # Too many neighbors - select nearest MAX_NEIGHBORS
                    neighbor_distances = dist[i, neighbor_indices]
                    _, nearest_indices = torch.topk(neighbor_distances, k=MAX_NEIGHBORS, largest=False)
                    selected_neighbors = neighbor_indices[nearest_indices]
                else:
                    selected_neighbors = neighbor_indices

                edge_count_stats[i] = len(selected_neighbors)

                # Add edges from residue i to selected neighbors
                if len(selected_neighbors) > 0:
                    sources = torch.full((len(selected_neighbors),), i, device=device)
                    edge_list.append(torch.stack([sources, selected_neighbors]))

            # Combine all edges
            if edge_list:
                ei_dist = torch.cat(edge_list, dim=1)
            else:
                ei_dist = torch.empty((2, 0), dtype=torch.long, device=device)

        # 2. Add sequence edges to ensure all residues stay connected to immediate neighbors
        # Use efficient tensor operations for parallelizability

        # Create all possible sequence edges (adjacent residues)
        if L > 1:
            # Forward edges: 0->1, 1->2, ..., (L-2)->(L-1)
            seq_forward = torch.stack([
                torch.arange(L-1, device=device),      # sources: [0, 1, 2, ..., L-2]
                torch.arange(1, L, device=device)      # targets: [1, 2, 3, ..., L-1]
            ])

            # Backward edges: 1->0, 2->1, ..., (L-1)->(L-2)
            seq_backward = torch.stack([
                torch.arange(1, L, device=device),     # sources: [1, 2, 3, ..., L-1]
                torch.arange(L-1, device=device)       # targets: [0, 1, 2, ..., L-2]
            ])

            # Combine forward and backward sequence edges
            ei_seq_all = torch.cat([seq_forward, seq_backward], dim=1)  # [2, 2*(L-1)]

            # Remove sequence edges that are already covered by distance edges
            if ei_dist.shape[1] > 0:
                # Create boolean mask for sequence edges that are NOT in distance edges
                # For each sequence edge, check if it exists in distance edges
                seq_src, seq_dst = ei_seq_all[0], ei_seq_all[1]
                dist_src, dist_dst = ei_dist[0], ei_dist[1]

                # Use broadcasting to check if each sequence edge exists in distance edges
                # Shape: [num_seq_edges, num_dist_edges]
                src_match = seq_src.unsqueeze(1) == dist_src.unsqueeze(0)  # [2*(L-1), num_dist_edges]
                dst_match = seq_dst.unsqueeze(1) == dist_dst.unsqueeze(0)  # [2*(L-1), num_dist_edges]
                edge_match = src_match & dst_match  # [2*(L-1), num_dist_edges]

                # Keep sequence edges that don't match any distance edge
                keep_mask = ~edge_match.any(dim=1)  # [2*(L-1)]
                ei_seq = ei_seq_all[:, keep_mask]
            else:
                # No distance edges, keep all sequence edges
                ei_seq = ei_seq_all
        else:
            # Single residue, no sequence edges
            ei_seq = torch.empty((2, 0), dtype=torch.long, device=device)

        # 3. Combine distance-based and sequence edges (no duplicates)
        ei_base = torch.cat([ei_dist, ei_seq], dim=1)

        # Optional: Log edge statistics
        MAX_NEIGHBORS = 80  # Safety limit for neighbors per residue
        if self.verbose:
            avg_neighbors = edge_count_stats.float().mean().item()
            max_neighbors = edge_count_stats.max().item()
            min_neighbors = edge_count_stats.min().item()
            over_limit_count = (edge_count_stats >= MAX_NEIGHBORS).sum().item()

            print(f"Distance-based graph building stats for {L} residues:")
            print(f"  Distance cutoff: {self.max_edge_dist}Å")
            print(f"  Distance-based edges: {ei_dist.shape[1]}")
            print(f"  Sequence edges: {ei_seq.shape[1]}")
            print(f"  Total real-real edges: {ei_base.shape[1]}")
            print(f"  Neighbors per residue - avg: {avg_neighbors:.1f}, min: {min_neighbors}, max: {max_neighbors}")
            if over_limit_count > 0:
                print(f"  Residues at neighbor limit ({MAX_NEIGHBORS}): {over_limit_count}/{L}")

        return ei_base

    def _create_edges_with_virtual(self, dist: torch.Tensor, geom_missing: torch.Tensor) -> torch.Tensor:
        """
        Creates edge indices including virtual node connections and sequence neighbors.

        Uses either distance-based or k-neighbor based edge building depending on configuration.

        Args:
            dist: Pairwise distance matrix [L, L]
            geom_missing: Boolean mask [L] indicating residues with missing geometry

        Returns:
            Edge index tensor [2, E] including virtual node edges and sequence edges
        """
        L = dist.shape[0]

        # Choose edge building strategy based on configuration
        if self.max_edge_dist is not None:
            # Distance-based edge building
            ei_base = self._create_distance_based_edges(dist, geom_missing)
        else:
            # Original k-neighbor based edge building
            ei_base = self._create_kneighbor_based_edges(dist, geom_missing)

        # Add virtual node edges if enabled (same for both strategies)
        if self.use_virtual_node:
            idx = torch.arange(L, device=dist.device)
            to_v   = torch.stack([idx, torch.full((L,), L, device=dist.device)], dim=0)
            from_v = torch.stack([torch.full((L,), L, device=dist.device), idx], dim=0)
            ei = torch.cat([ei_base, to_v, from_v], dim=1)
            return ei

        # Return only real-real edges when virtual node disabled
        return ei_base

    def _create_kneighbor_based_edges(self, dist: torch.Tensor, geom_missing: torch.Tensor) -> torch.Tensor:
        """
        Creates edges using the original k-neighbor approach.

        Args:
            dist: Pairwise distance matrix [L, L]
            geom_missing: Boolean mask [L] indicating residues with missing geometry

        Returns:
            Edge index tensor [2, E] for real-real edges and sequence edges
        """
        L = dist.shape[0]

        # 1. K-nearest neighbors (exclude self-connections)
        dist_for_knn = dist.clone()
        dist_for_knn.fill_diagonal_(float('inf'))  # Exclude self from nearest neighbors
        _, indices_knn = torch.topk(dist_for_knn, k=min(self.k, L-1), dim=-1, largest=False)
        row_knn = torch.arange(L, device=dist.device).unsqueeze(1).expand(-1, indices_knn.shape[1])
        ei_knn = torch.stack([row_knn.flatten(), indices_knn.flatten()], dim=0)

        # 2. Farthest neighbors in 3D space (exclude self-connections)
        dist_for_far = dist.clone()
        dist_for_far.fill_diagonal_(0.0)  # Exclude self from farthest neighbors
        _, indices_far = torch.topk(dist_for_far, k=min(self.k_farthest, L-1), dim=-1, largest=True)
        row_far = torch.arange(L, device=dist.device).unsqueeze(1).expand(-1, indices_far.shape[1])
        ei_far = torch.stack([row_far.flatten(), indices_far.flatten()], dim=0)

        # 3. OPTIMIZED: Random connections (5.7x faster than original!)
        if self.k_random > 0:
            # Create existing connections mask efficiently (vectorized)
            existing_mask = torch.zeros(L, L, dtype=torch.bool, device=dist.device)

            # Mark existing connections in batch operations
            row_indices = torch.arange(L, device=dist.device).unsqueeze(1)
            existing_mask[row_indices, indices_knn] = True
            existing_mask[row_indices, indices_far] = True
            existing_mask.fill_diagonal_(True)  # No self-connections

            # Vectorized random sampling
            rand_edges = []
            for i in range(L):
                # Get available targets (vectorized)
                available = (~existing_mask[i]).nonzero().flatten()
                if len(available) > 0:
                    # Efficient sampling without replacement
                    k_sample = min(self.k_random, len(available))
                    if k_sample == len(available):
                        selected = available
                    else:
                        perm = torch.randperm(len(available), device=dist.device)[:k_sample]
                        selected = available[perm]

                    # Create edges efficiently
                    sources = torch.full((k_sample,), i, device=dist.device)
                    rand_edges.append(torch.stack([sources, selected]))

            ei_rand = torch.cat(rand_edges, dim=1) if rand_edges else torch.empty((2, 0), dtype=torch.long, device=dist.device)
        else:
            ei_rand = torch.empty((2, 0), dtype=torch.long, device=dist.device)

        # 4. Add sequence edges to ensure all residues stay connected to immediate neighbors
        # Use efficient tensor operations for parallelizability

        # Create all possible sequence edges (adjacent residues)
        if L > 1:
            # Forward edges: 0->1, 1->2, ..., (L-2)->(L-1)
            seq_forward = torch.stack([
                torch.arange(L-1, device=dist.device),      # sources: [0, 1, 2, ..., L-2]
                torch.arange(1, L, device=dist.device)      # targets: [1, 2, 3, ..., L-1]
            ])

            # Backward edges: 1->0, 2->1, ..., (L-1)->(L-2)
            seq_backward = torch.stack([
                torch.arange(1, L, device=dist.device),     # sources: [1, 2, 3, ..., L-1]
                torch.arange(L-1, device=dist.device)       # targets: [0, 1, 2, ..., L-2]
            ])

            # Combine forward and backward sequence edges
            ei_seq_all = torch.cat([seq_forward, seq_backward], dim=1)  # [2, 2*(L-1)]

            # Remove sequence edges that are already covered by existing edges
            all_existing = torch.cat([ei_knn, ei_far, ei_rand], dim=1)

            if all_existing.shape[1] > 0:
                # Create boolean mask for sequence edges that are NOT in existing edges
                seq_src, seq_dst = ei_seq_all[0], ei_seq_all[1]
                exist_src, exist_dst = all_existing[0], all_existing[1]

                # Use broadcasting to check if each sequence edge exists in existing edges
                src_match = seq_src.unsqueeze(1) == exist_src.unsqueeze(0)
                dst_match = seq_dst.unsqueeze(1) == exist_dst.unsqueeze(0)
                edge_match = src_match & dst_match

                # Keep sequence edges that don't match any existing edge
                keep_mask = ~edge_match.any(dim=1)
                ei_seq = ei_seq_all[:, keep_mask]
            else:
                # No existing edges, keep all sequence edges
                ei_seq = ei_seq_all
        else:
            # Single residue, no sequence edges
            ei_seq = torch.empty((2, 0), dtype=torch.long, device=dist.device)

        # 5. Combine all edge types for real nodes
        ei_base = torch.cat([ei_knn, ei_far, ei_rand, ei_seq], dim=1)

        # Optional: Log edge counts for debugging
        if self.verbose:
            print(f"K-neighbor graph building stats for {L} residues:")
            print(f"  K-nearest edges: {ei_knn.shape[1]}")
            print(f"  Farthest edges: {ei_far.shape[1]}")
            print(f"  Random edges: {ei_rand.shape[1]}")
            print(f"  Sequence edges: {ei_seq.shape[1]}")
            print(f"  Total real-real edges: {ei_base.shape[1]}")

        return ei_base

    def _compute_edge_features(self, coords4: torch.Tensor,
                             edge_index: torch.Tensor,
                             dist: torch.Tensor,
                             geom_missing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes edge features for the graph including virtual node edges.

        Args:
            coords4: Backbone coordinates [L, 4, 3]
            edge_index: Edge indices [2, E] (including virtual node)
            dist: Pairwise distance matrix [L, L]
            geom_missing: Boolean mask [L] indicating residues with missing geometry

        Returns:
            Tuple of (edge_s, edge_v) where:
            - edge_s: Scalar edge features [E, F_s]
            - edge_v: Vector edge features [E, 1, 3]
        """
        device = coords4.device
        E = edge_index.shape[1]
        L = coords4.shape[0]

        # Get source and target node indices
        src, dst = edge_index[0], edge_index[1]

        # Mask to separate real-real edges from virtual edges
        mask = (src != L) & (dst != L)

        # Mask for edges involving geometry-missing nodes
        # These edges get special handling (zero features for geometry-based components)
        # For virtual edges, we skip the geometry check since virtual node doesn't have geometry state
        geom_valid_mask = torch.zeros_like(mask, dtype=torch.bool)
        # Only check geometry for real-real edges
        real_real_mask = mask
        if real_real_mask.any():
            real_src = src[real_real_mask]
            real_dst = dst[real_real_mask]
            geom_valid_mask[real_real_mask] = (~geom_missing[real_src]) & (~geom_missing[real_dst])

        # Initialize edge features with proper GVP-GNN dimensions
        # Scalar: RBF (16) + positional embeddings (16) = 32 total
        # Vector: direction vectors (1 vector of 3D)
        scalar_dim = 16 + 16  # RBF + positional embeddings = 32
        edge_s = torch.zeros(E, scalar_dim, device=device)
        edge_v = torch.zeros(E, 1, 3, device=device)

        # Compute positional embeddings for all edges (using lookup table)
        try:
            seq_distances = edge_index[0] - edge_index[1]  # Sequence distance
            pos_embeddings = self.rbf_manager.lookup_seq_rbf(seq_distances, device=device)
        except Exception as e:
            print(f"ERROR in positional embeddings lookup computation: {e}")
            print(f"edge_index shape: {edge_index.shape}")
            raise

        # For real-real edges with valid geometry, compute full geometric features
        if geom_valid_mask.sum() > 0:
            src_valid, dst_valid = src[geom_valid_mask], dst[geom_valid_mask]

            # 3D distance features with 16 RBF centers (using lookup table)
            try:
                dist_3d = dist[src_valid, dst_valid]
                if self.smoke_test:
                    print(f"3D distance range: {dist_3d.min():.3f} - {dist_3d.max():.3f}")

                # Use lookup table for RBF computation
                rbf_3d = self.rbf_manager.lookup_3d_rbf(dist_3d, device=device)
            except Exception as e:
                print(f"ERROR in RBF lookup computation: {e}")
                print(f"dist_3d shape: {dist_3d.shape}")
                raise

            # Direction vector (unit vector from source to target)
            try:
                CA_src = coords4[src_valid, 1, :]  # CA coordinates of source nodes
                CA_dst = coords4[dst_valid, 1, :]  # CA coordinates of target nodes
                direction = CA_dst - CA_src
                if self.smoke_test:
                    direction = _normalize_safe(direction, dim=-1, smoke_test=True)
                else:
                    direction = _normalize(direction, dim=-1)
            except Exception as e:
                print(f"ERROR in direction computation: {e}")
                print(f"CA_src shape: {CA_src.shape}, CA_dst shape: {CA_dst.shape}")
                raise

            # Assign scalar features (RBF + positional embeddings) for geometry-valid edges
            try:
                edge_s_concat = torch.cat([rbf_3d, pos_embeddings[geom_valid_mask]], dim=-1)
                edge_s[geom_valid_mask] = edge_s_concat
            except Exception as e:
                print(f"ERROR in edge_s concatenation: {e}")
                raise

            # Assign vector features (direction) for geometry-valid edges
            try:
                edge_v[geom_valid_mask, 0, :] = direction
            except Exception as e:
                print(f"ERROR in edge_v assignment: {e}")
                raise

        # For edges involving geometry-missing nodes, use sequence-based features only
        geom_missing_mask = mask & ~geom_valid_mask
        if geom_missing_mask.sum() > 0:
            # For geometry-missing edges, set RBF features to zero (no geometric distance info)
            # but keep positional embeddings (sequence distance info)
            zero_rbf = torch.zeros(geom_missing_mask.sum(), 16, device=device)
            edge_s_missing = torch.cat([zero_rbf, pos_embeddings[geom_missing_mask]], dim=-1)
            edge_s[geom_missing_mask] = edge_s_missing

            # Vector features are zero for geometry-missing edges (no spatial direction)
            edge_v[geom_missing_mask] = 0.0

        # For virtual edges, use small random values
        try:
            virt_mask = ~mask
            if virt_mask.sum() > 0:
                edge_s[virt_mask] = torch.randn(virt_mask.sum(), scalar_dim, device=device) * 0.01
                edge_v[virt_mask] = torch.randn(virt_mask.sum(), 1, 3, device=device) * 0.01
        except Exception as e:
            print(f"ERROR in virtual edge assignment: {e}")
            print(f"virt_mask sum: {virt_mask.sum()}")
            raise


        # Ensure edge_s has correct shape [E, 32] - flatten if needed
        if edge_s.dim() > 2:
            edge_s = edge_s.view(E, -1)
            print(f"[DEBUG] Reshaped edge_s to: {edge_s.shape}", flush=True)

        return edge_s, edge_v

    def _compute_weighted_packing_score(self, coords4: torch.Tensor, seq: str,
                                        context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a Gaussian-weighted local packing score for each residue and return
        both the log-scaled raw score and a within-structure percentile rank.

        Uses virtual Cβ as the anchor atom (Cα for glycine), which better reflects
        side-chain packing geometry than Cα alone. Virtual Cβ is reconstructed from
        backbone N, Cα, C via standard geometry (Bhattacharya et al. formula).

        For each residue i the raw score is:
            score_i = sum_{j: |i-j|>2, d_ij < 14 Å} exp(-(d_ij / 8.0)^2)

        Two complementary features are returned:
          - log1p(raw_score): captures absolute local density across structures.
            High in compact/globular proteins, low in IDRs globally.
          - within-structure percentile rank in [0, 1]: captures relative burial
            regardless of overall chain compactness. Robust to PDB vs AF2 distribution
            differences. Set to 0.5 (neutral) for chains with negligible burial
            gradient (max - min < 1.0) to avoid percentile noise on flat distributions.

        Together they let the model distinguish:
          high raw + high percentile  → genuinely buried, compact protein
          low raw  + high percentile  → "most buried" residue in a disordered chain
          high raw + low percentile   → outlier packed site in a mostly-surface chain
          low raw  + low percentile   → surface residue in any context

        Args:
            coords4:      Backbone coordinates [L, 4, 3] in order N, CA, C, O.
            seq:          Amino acid sequence string of length L.
            context_mask: Optional boolean tensor [L]. When provided (multi-chain
                          conditioning), True marks context/fixed residues (e.g. GFP)
                          that contribute edges and message passing but must NOT affect
                          the burial score of target residues. Packing scores for target
                          residues are computed using only intra-target neighbors,
                          preserving the single-chain training distribution. Context
                          residues receive a fixed neutral burial signal (log=0,
                          percentile=0.5) matching the virtual-node convention.

        Returns:
            log_scores:   Float tensor [L], log1p of raw packing score.
            percentiles:  Float tensor [L], within-structure percentile rank in [0, 1].
        """
        OUTER_CUTOFF      = 14.0
        SIGMA             = 8.0
        SEQ_EXCL          = 2
        FLAT_SCORE_THRESH = 1.0   # chains with max-min below this get neutral percentile

        coords_np = coords4.cpu().numpy()
        N_coords  = coords_np[:, 0, :]   # [L, 3]
        CA_coords = coords_np[:, 1, :]   # [L, 3]
        C_coords  = coords_np[:, 2, :]   # [L, 3]
        L = len(CA_coords)

        # Resolve which residues are context-only (should not be scored)
        if context_mask is not None:
            is_context = context_mask.cpu().numpy().astype(bool)  # [L]
        else:
            is_context = np.zeros(L, dtype=bool)
        is_target = ~is_context  # residues we actually want to score

        # Identify residues with complete backbone coords
        valid = ~(np.any(np.isnan(N_coords),  axis=1) |
                  np.any(np.isnan(CA_coords), axis=1) |
                  np.any(np.isnan(C_coords),  axis=1))

        # Build anchor array: virtual Cβ for non-Gly, Cα for Gly / missing
        anchors = np.full((L, 3), np.nan)
        if valid.any():
            b  = CA_coords[valid] - N_coords[valid]
            c_ = C_coords[valid]  - CA_coords[valid]
            a  = np.cross(b, c_)
            cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_ + CA_coords[valid]
            anchors[valid] = cb

            # Glycine has no Cβ — use Cα instead
            valid_idx = np.where(valid)[0]
            for pos, aa in zip(valid_idx, [seq[i] for i in valid_idx]):
                if aa == 'G':
                    anchors[pos] = CA_coords[pos]

        raw_scores = np.zeros(L, dtype=np.float32)

        # Target residues that have valid coordinates
        target_valid = valid & is_target
        target_valid_idx = np.where(target_valid)[0]

        if context_mask is not None:
            # ----------------------------------------------------------------
            # Multi-chain mode: compute burial independently per chain so that
            # neither chain inflates the other's packing score.
            #
            # Target residues (NB): scored using only intra-target neighbors.
            # Context residues (GFP): scored using only intra-context neighbors.
            # Rationale: both chains were seen by the model only as monomers
            # during training. Cross-chain density would place residues
            # out-of-distribution for the burial features. GFP context enters
            # the model exclusively through message-passing edge features.
            # ----------------------------------------------------------------
            context_valid_idx = np.where(valid & is_context)[0]
            anch_context = anchors[context_valid_idx]                                  # [C, 3]

            # Target burial — intra-target only
            if len(target_valid_idx) >= 2:
                anch_target = anchors[target_valid_idx]                                # [T, 3]
                diff_tt  = anch_target[:, None, :] - anch_target[None, :, :]          # [T, T, 3]
                dist_tt  = np.sqrt((diff_tt ** 2).sum(axis=-1))                        # [T, T]
                seq_d_tt = np.abs(target_valid_idx[:, None] - target_valid_idx[None, :])
                elig_tt  = (dist_tt < OUTER_CUTOFF) & (seq_d_tt > SEQ_EXCL) & (dist_tt > 0)
                w_tt     = np.where(elig_tt, np.exp(-(dist_tt / SIGMA) ** 2), 0.0)
                raw_scores[target_valid_idx] = w_tt.sum(axis=1).astype(np.float32)

            # Context burial — intra-context only
            if len(context_valid_idx) >= 2:
                diff_cc  = anch_context[:, None, :] - anch_context[None, :, :]        # [C, C, 3]
                dist_cc  = np.sqrt((diff_cc ** 2).sum(axis=-1))                        # [C, C]
                seq_d_cc = np.abs(context_valid_idx[:, None] - context_valid_idx[None, :])
                elig_cc  = (dist_cc < OUTER_CUTOFF) & (seq_d_cc > SEQ_EXCL) & (dist_cc > 0)
                w_cc     = np.where(elig_cc, np.exp(-(dist_cc / SIGMA) ** 2), 0.0)
                raw_scores[context_valid_idx] = w_cc.sum(axis=1).astype(np.float32)

        else:
            # Original single-chain path: all residues are targets
            if len(target_valid_idx) >= 2:
                anch_target = anchors[target_valid_idx]                                # [L, 3]
                diff  = anch_target[:, None, :] - anch_target[None, :, :]             # [L, L, 3]
                dist  = np.sqrt((diff ** 2).sum(axis=-1))                              # [L, L]
                seq_d = np.abs(target_valid_idx[:, None] - target_valid_idx[None, :])
                eligible = (dist < OUTER_CUTOFF) & (seq_d > SEQ_EXCL) & (dist > 0)
                weights  = np.where(eligible, np.exp(-(dist / SIGMA) ** 2), 0.0)
                raw_scores[target_valid_idx] = weights.sum(axis=1).astype(np.float32)

        log_scores  = np.log1p(raw_scores).astype(np.float32)
        percentiles = np.zeros(L, dtype=np.float32)

        # Percentile rank computed independently per chain
        for valid_idx_group in ([target_valid_idx] if context_mask is None
                                else [target_valid_idx,
                                      np.where(valid & is_context)[0]]):
            if len(valid_idx_group) >= 2:
                group_raw   = raw_scores[valid_idx_group]
                score_range = group_raw.max() - group_raw.min()
                if score_range < FLAT_SCORE_THRESH:
                    percentiles[valid_idx_group] = 0.5
                else:
                    order = group_raw.argsort().argsort()
                    percentiles[valid_idx_group] = (order / (len(valid_idx_group) - 1)).astype(np.float32)
            elif len(valid_idx_group) == 1:
                percentiles[valid_idx_group] = 0.5

        return torch.from_numpy(log_scores), torch.from_numpy(percentiles)

    def _compute_node_features(self, coords4: torch.Tensor,
                             scores_norm: torch.Tensor,
                             source: str,
                             geom_missing: torch.Tensor,
                             edge_index: Optional[torch.Tensor] = None,
                             seq: str = "",
                             context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes node features including virtual node and geometry missing flag.

        Args:
            coords4: Backbone coordinates [L, 4, 3]
            scores_norm: Normalized uncertainty scores [L] (already processed)
            source: Data source ('pdb', 'alphafold2', or 'cath')
            geom_missing: Boolean mask [L] indicating residues with missing geometry
            edge_index: Edge index tensor [2, E] used to compute node degree when
                        include_node_degree=True. Must be provided when that flag is set.
            seq: Amino acid sequence string used for virtual-Cβ packing score when
                 include_node_degree=True.
            context_mask: Optional boolean tensor [L]. True = context/fixed chain residue
                          (e.g. GFP). Passed to _compute_weighted_packing_score so those
                          atoms are excluded from the burial score of target residues.

        Returns:
            Tuple of (scalar_features, vector_features) for L+1 nodes
        """
        L = coords4.shape[0]

        # Geometric features for real nodes
        # For nodes with missing geometry, these will be zeros (placeholder values)
        try:
            if self.smoke_test:
                dihedrals = _dihedrals_safe(coords4, smoke_test=True)  # [L, 6] - cos/sin of phi, psi, omega
            else:
                dihedrals = _dihedrals(coords4)  # [L, 6] - cos/sin of phi, psi, omega
        except Exception as e:
            print(f"[ERROR] Failed to compute dihedrals", flush=True)
            print(f"[ERROR] coords4 shape: {coords4.shape}", flush=True)
            raise e

        try:
            if self.smoke_test:
                orientations = _orientations_safe(coords4[:, 1], smoke_test=True)  # [L, 2, 3] - forward/backward directions from CA
            else:
                orientations = _orientations(coords4[:, 1])  # [L, 2, 3] - forward/backward directions from CA
        except Exception as e:
            print(f"[ERROR] Failed to compute orientations", flush=True)
            print(f"[ERROR] coords4[:, 1] shape: {coords4[:, 1].shape}", flush=True)
            raise e

        try:
            if self.smoke_test:
                sidechains = _sidechains_safe(coords4, smoke_test=True)  # [L, 3] - sidechain directions
            else:
                sidechains = _sidechains(coords4)  # [L, 3] - sidechain directions
        except Exception as e:
            print(f"[ERROR] Failed to compute sidechains", flush=True)
            print(f"[ERROR] coords4 shape: {coords4.shape}", flush=True)
            raise e

        # Use the already-computed normalized scores (passed as parameter)
        # This avoids redundant computation since normalization was done earlier for noise scaling

        # Data source indicators
        if self.no_source_indicator:
            # When no_source_indicator is True, use [1.0, 1.0] for all sources
            source_indicator = torch.tensor([1.0, 1.0], device=coords4.device)  # [1, 1]
        else:
            # Use distinctive indicators for each source (only alphafold2 and pdb)
            if source == 'alphafold2':
                source_indicator = torch.tensor([0.0, 1.0], device=coords4.device)  # [0, 1]
            elif source == 'pdb':
                source_indicator = torch.tensor([1.0, 0.0], device=coords4.device)  # [1, 0]
            else:
                raise ValueError(f"Unsupported source '{source}'. Source must be either 'alphafold2' or 'pdb'.")

        # Expand source indicator to all nodes
        source_indicator = source_indicator.unsqueeze(0).expand(L, -1)

        # Scalar node features for real nodes
        try:
            # Add geometry missing flag as an additional scalar feature
            # This tells the model which residues have missing geometric information
            scalar_parts = [
                dihedrals,                           # Dihedral angles [L, 6] (zeros for missing geometry)
                scores_norm.unsqueeze(-1),           # Normalized scores [L, 1]
                source_indicator,                    # Data source indicator [L, 2]
                geom_missing.float().unsqueeze(-1),  # Geometry missing flag [L, 1]
            ]

            if self.include_node_degree:
                # Two complementary burial features derived from Gaussian-weighted
                # virtual-Cβ packing score (Cα for Gly):
                #   1. log1p(raw score): absolute local density signal. High for buried
                #      residues in compact proteins, low for IDRs globally.
                #   2. within-structure percentile rank [0,1]: relative burial robust
                #      to PDB vs AF2 compactness differences. Neutral (0.5) for chains
                #      with negligible burial gradient to avoid noise on flat distributions.
                # Together they let the model distinguish genuinely buried residues from
                # "most packed in an otherwise disordered chain".
                # context_mask (if set) prevents context/fixed chain atoms from inflating
                # the burial score of target residues in multi-chain conditioning.
                log_packing, percentile = self._compute_weighted_packing_score(
                    coords4, seq, context_mask=context_mask)
                log_packing = log_packing.to(coords4.device).unsqueeze(-1)   # [L, 1]
                percentile  = percentile.to(coords4.device).unsqueeze(-1)    # [L, 1]
                scalar_parts.append(log_packing)
                scalar_parts.append(percentile)

            node_s_real = torch.cat(scalar_parts, dim=-1)
            # dim: 10 without degree feature, 11 with it
        except Exception as e:
            print(f"[ERROR] Failed to concatenate scalar node features", flush=True)
            print(f"[ERROR] dihedrals: {dihedrals.shape}", flush=True)
            print(f"[ERROR] scores_norm.unsqueeze(-1): {scores_norm.unsqueeze(-1).shape}", flush=True)
            print(f"[ERROR] source_indicator: {source_indicator.shape}", flush=True)
            print(f"[ERROR] geom_missing: {geom_missing.shape}", flush=True)
            raise e

        # Vector node features for real nodes (following GVP format)
        try:
            # Need to make sidechains have shape [L, 1, 3] to match orientations [L, 2, 3]
            if sidechains.dim() == 2:  # [L, 3] -> [L, 1, 3]
                sidechains = sidechains.unsqueeze(1)
                print(f"[DEBUG] Reshaped sidechains to: {sidechains.shape}", flush=True)

            node_v_real = torch.cat([
                orientations,                 # Forward/backward directions [L, 2, 3]
                sidechains,                   # Side chain directions [L, 1, 3]
            ], dim=-2)  # [L, 3, 3]
        except Exception as e:
            print(f"[ERROR] Failed to concatenate vector node features", flush=True)
            print(f"[ERROR] orientations: {orientations.shape}", flush=True)
            print(f"[ERROR] sidechains: {sidechains.shape}", flush=True)
            raise e

        # Combine real and optional virtual nodes
        if self.use_virtual_node:
            # Virtual node features
            try:
                # Scalar: zeros for geometric features, ZERO uncertainty (extremely flexible), source indicator, geometry flag
                virtual_parts = [
                    torch.zeros(1, 6, device=coords4.device),  # No dihedrals (6 dims)
                    torch.zeros(1, 1, device=coords4.device),  # Zero uncertainty for virtual node (1 dim)
                    source_indicator[:1],                       # Same source indicator (2 dims)
                    torch.zeros(1, 1, device=coords4.device),  # Virtual node has no missing geometry (1 dim)
                ]
                if self.include_node_degree:
                    # Virtual node has no physical position:
                    # log_packing = 0 (log1p(0)), percentile = 0.5 (neutral)
                    virtual_parts.append(torch.zeros(1, 1, device=coords4.device))          # log_packing
                    virtual_parts.append(torch.full((1, 1), 0.5, device=coords4.device))    # percentile
                virtual_s = torch.cat(virtual_parts, dim=-1)  # [1, 10] or [1, 11]
            except Exception as e:
                print(f"[ERROR] Failed to create virtual scalar features", flush=True)
                print(f"[ERROR] scores_norm.mean(): {scores_norm.mean()}", flush=True)
                print(f"[ERROR] source_indicator[:1]: {source_indicator[:1].shape}", flush=True)
                raise e

            try:
                # Vector: zeros matching the real node vector structure [1, 3, 3]
                virtual_v = torch.zeros(1, 3, 3, device=coords4.device)
            except Exception as e:
                print(f"[ERROR] Failed to create virtual vector features", flush=True)
                raise e

            # Combine real and virtual nodes
            try:
                node_s = torch.cat([node_s_real, virtual_s], dim=0)  # [L+1, 10]
            except Exception as e:
                print(f"[ERROR] Failed to concatenate final scalar features", flush=True)
                print(f"[ERROR] node_s_real: {node_s_real.shape}, virtual_s: {virtual_s.shape}", flush=True)
                raise e

            try:
                node_v = torch.cat([node_v_real, virtual_v], dim=0)  # [L+1, 3, 3]
            except Exception as e:
                print(f"[ERROR] Failed to concatenate final vector features", flush=True)
                print(f"[ERROR] node_v_real: {node_v_real.shape}, virtual_v: {virtual_v.shape}", flush=True)
                raise e

            return node_s, node_v

        # When virtual node disabled, return only real node features
        return node_s_real, node_v_real


# Flank filtering utilities for sampling mode

def detect_missing_flanks(coords4: torch.Tensor, verbose: bool = False) -> Tuple[int, int]:
    """
    Find indices of first and last residues with valid coordinates.

    This function identifies the valid residue range by finding the first and last
    residues with non-missing coordinates. Residues in the middle with missing
    coordinates are included in the range.

    Args:
        coords4: Backbone coordinates [L, 4, 3] (N, CA, C, O)
        verbose: If True, print diagnostic information

    Returns:
        Tuple of (start_idx, end_idx) where:
        - start_idx: Index of first residue with valid coordinates (inclusive)
        - end_idx: Index of last residue with valid coordinates (inclusive)
        - Returns (0, -1) if no valid residues found

    Examples:
        >>> coords = torch.zeros(10, 4, 3)
        >>> coords[3:7] = torch.randn(4, 4, 3)  # Valid residues at 3-6
        >>> start, end = detect_missing_flanks(coords)
        >>> print(start, end)  # Output: 3, 6
    """
    L = coords4.shape[0]

    # Identify residues with valid coordinates
    # A residue is valid if it has no NaN/Inf and is not all zeros
    valid = torch.zeros(L, dtype=torch.bool, device=coords4.device)

    for i in range(L):
        residue_coords = coords4[i]  # [4, 3] - all 4 backbone atoms

        # Check for NaN or Inf
        has_nan_inf = torch.isnan(residue_coords).any() or torch.isinf(residue_coords).any()

        # Check if all coordinates are zero (placeholder for missing)
        # Use small epsilon to handle floating point precision
        is_all_zero = (residue_coords.abs().sum() < 1e-6)

        # Valid if neither condition is true
        valid[i] = not (has_nan_inf or is_all_zero)

    # Find indices of all valid residues
    valid_indices = valid.nonzero().flatten()

    if len(valid_indices) == 0:
        # No valid residues found
        if verbose:
            print(f"WARNING: No valid residues found in structure")
        return 0, -1

    # First and last valid residues define the range
    start_idx = valid_indices[0].item()
    end_idx = valid_indices[-1].item()

    if verbose:
        num_missing_start = start_idx
        num_missing_end = L - end_idx - 1
        num_missing_middle = L - end_idx + start_idx - 1 - len(valid_indices)

        print(f"Flank detection results:")
        print(f"  Total residues: {L}")
        print(f"  Valid range: {start_idx} to {end_idx} (inclusive)")
        print(f"  Missing at start: {num_missing_start}")
        print(f"  Missing at end: {num_missing_end}")
        print(f"  Missing in middle: {num_missing_middle}")

    return start_idx, end_idx


def filter_missing_flanks(entry: dict, verbose: bool = False) -> Tuple[dict, int, int]:
    """
    Remove residues with missing coordinates from start and end of a protein entry.

    This function filters out flanking residues with missing coordinates while
    preserving residues in the middle that may have missing coordinates. It maintains
    alignment between sequence, coordinates, B-factors/pLDDT, and DSSP annotations.

    Args:
        entry: Protein data dictionary with keys:
            - 'seq': Amino acid sequence string
            - 'coords': Coordinates (dict of arrays or [L, 4, 3] array)
            - 'b_factors' or 'plddt': Uncertainty scores
            - 'dssp': (optional) DSSP annotations
            - Other metadata preserved as-is
        verbose: If True, print filtering information

    Returns:
        Tuple of (filtered_entry, start_idx, end_idx) where:
        - filtered_entry: New dict with filtered data
        - start_idx: Index of first residue in original sequence (for output mapping)
        - end_idx: Index of last residue in original sequence (for output mapping)

    Raises:
        ValueError: If no valid residues found or data alignment issues detected
    """
    import numpy as np

    # Convert coordinates to standardized tensor format
    coords_np = entry['coords']

    if isinstance(coords_np, dict) and 'N' in coords_np:
        # Dict format: {'N': array, 'CA': array, 'C': array, 'O': array}
        # Convert to [L, 4, 3] format
        coords4 = torch.tensor(np.stack([
            coords_np.get(atom, coords_np['CA'])
            for atom in ['N', 'CA', 'C', 'O']
        ], axis=1), dtype=torch.float32)
    else:
        # Already in array format
        coords4 = torch.tensor(coords_np, dtype=torch.float32)

    # Validate coordinate shape
    if coords4.dim() != 3 or coords4.shape[1] != 4 or coords4.shape[2] != 3:
        raise ValueError(f"Expected coords shape [L, 4, 3], got {coords4.shape}")

    L = coords4.shape[0]
    original_seq_len = len(entry['seq'])

    # Validate sequence-coordinate alignment
    if L != original_seq_len:
        raise ValueError(f"Coordinate length ({L}) does not match sequence length ({original_seq_len})")

    # Detect missing flanks
    start_idx, end_idx = detect_missing_flanks(coords4, verbose=verbose)

    # Handle edge case: no valid residues
    if end_idx < start_idx:
        raise ValueError(f"No valid residues found in {entry.get('name', 'unknown')}")

    # Edge case: no filtering needed
    if start_idx == 0 and end_idx == L - 1:
        if verbose:
            print(f"No flanking residues to filter for {entry.get('name', 'unknown')}")
        return entry, 0, L - 1

    # Perform filtering
    if verbose:
        print(f"Filtering {entry.get('name', 'unknown')}:")
        print(f"  Original length: {L}")
        print(f"  Filtered length: {end_idx - start_idx + 1}")
        print(f"  Removing {start_idx} residues from start, {L - end_idx - 1} from end")

    # Create filtered entry (shallow copy to preserve metadata)
    filtered_entry = entry.copy()

    # Filter sequence
    filtered_entry['seq'] = entry['seq'][start_idx:end_idx + 1]

    # Filter coordinates
    if isinstance(coords_np, dict) and 'N' in coords_np:
        # Filter each atom type separately
        filtered_coords = {}
        for atom in ['N', 'CA', 'C', 'O']:
            if atom in coords_np:
                filtered_coords[atom] = coords_np[atom][start_idx:end_idx + 1]
        filtered_entry['coords'] = filtered_coords
    else:
        # Filter array format
        filtered_entry['coords'] = coords_np[start_idx:end_idx + 1]

    # Filter B-factors or pLDDT scores
    if 'b_factors' in entry:
        b_factors = entry['b_factors']
        if isinstance(b_factors, np.ndarray):
            filtered_entry['b_factors'] = b_factors[start_idx:end_idx + 1]
        elif isinstance(b_factors, list):
            filtered_entry['b_factors'] = b_factors[start_idx:end_idx + 1]
        else:
            filtered_entry['b_factors'] = b_factors  # Preserve if unknown type

    if 'plddt' in entry:
        plddt = entry['plddt']
        if isinstance(plddt, np.ndarray):
            filtered_entry['plddt'] = plddt[start_idx:end_idx + 1]
        elif isinstance(plddt, list):
            filtered_entry['plddt'] = plddt[start_idx:end_idx + 1]
        else:
            filtered_entry['plddt'] = plddt  # Preserve if unknown type

    # Filter DSSP annotations if present
    if 'dssp' in entry:
        dssp = entry['dssp']
        if isinstance(dssp, str):
            filtered_entry['dssp'] = dssp[start_idx:end_idx + 1]
        elif isinstance(dssp, (list, np.ndarray)):
            filtered_entry['dssp'] = dssp[start_idx:end_idx + 1]

    # Validate filtered data alignment
    filtered_len = len(filtered_entry['seq'])
    expected_len = end_idx - start_idx + 1

    if filtered_len != expected_len:
        raise ValueError(f"Filtered sequence length ({filtered_len}) does not match "
                        f"expected length ({expected_len})")

    # Validate coordinates
    if isinstance(filtered_entry['coords'], dict):
        coord_len = len(filtered_entry['coords']['CA'])
    else:
        coord_len = filtered_entry['coords'].shape[0]

    if coord_len != filtered_len:
        raise ValueError(f"Filtered coordinate length ({coord_len}) does not match "
                        f"filtered sequence length ({filtered_len})")

    # Validate B-factors/pLDDT if present
    if 'b_factors' in filtered_entry:
        if isinstance(filtered_entry['b_factors'], (list, np.ndarray)):
            bf_len = len(filtered_entry['b_factors'])
            if bf_len != filtered_len:
                raise ValueError(f"Filtered B-factor length ({bf_len}) does not match "
                               f"filtered sequence length ({filtered_len})")

    if 'plddt' in filtered_entry:
        if isinstance(filtered_entry['plddt'], (list, np.ndarray)):
            plddt_len = len(filtered_entry['plddt'])
            if plddt_len != filtered_len:
                raise ValueError(f"Filtered pLDDT length ({plddt_len}) does not match "
                               f"filtered sequence length ({filtered_len})")

    # Validate DSSP if present
    if 'dssp' in filtered_entry:
        if isinstance(filtered_entry['dssp'], (str, list, np.ndarray)):
            dssp_len = len(filtered_entry['dssp'])
            if dssp_len != filtered_len:
                raise ValueError(f"Filtered DSSP length ({dssp_len}) does not match "
                               f"filtered sequence length ({filtered_len})")

    if verbose:
        print(f"Filtering successful - all data aligned at length {filtered_len}")

    return filtered_entry, start_idx, end_idx

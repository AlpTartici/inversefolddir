#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
LDDT (Local Distance Difference Test) Calculator

This module computes LDDT scores between predicted and reference protein structures
following the canonical CASP/AlphaFold protocol.

LDDT Metric (Standard Implementation):
    - For each residue i, identifies neighbors j where:
        * Distance in reference < 15Å (inclusion radius)
        * |i - j| >= 2 (excludes covalently bonded neighbors)
    - For each neighbor pair, checks distance preservation at 4 thresholds (0.5, 1, 2, 4 Å)
    - Computes per-residue LDDT: mean fraction preserved across thresholds
    - Final LDDT: equal-weight average across all residues with neighbors
    - Reported as 0-100 (percentage)

Standard Protocol:
    - Uses Cα atoms only (Cα-LDDT)
    - No global superposition (alignment-free local metric)
    - Excludes sequence neighbors |i-j| < 2 to avoid bias from local geometry
    - Per-residue averaging (not pair-weighted)
    - Proper residue mapping by (chain, resSeq, iCode)
    - Evaluates per chain
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Import residue mapper
try:
    from .residue_mapper import map_residues
except ImportError:
    from residue_mapper import map_residues

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class LDDTCalculator:
    """
    Calculator for LDDT (Local Distance Difference Test) scores.

    Implements canonical Cα-LDDT following CASP/AlphaFold protocol:
    - 15 Å inclusion radius for local neighborhoods
    - Four distance preservation thresholds: 0.5, 1.0, 2.0, 4.0 Å
    - Sequence separation |i-j| >= 2 (excludes covalently bonded residues)
    - Per-residue averaging (equal weight per residue, not per pair)
    - No global superposition (purely local metric)
    """

    def __init__(self, inclusion_radius: float = 15.0,
                 thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                 sequence_separation: int = 2):
        """
        Initialize LDDT calculator.

        Args:
            inclusion_radius: Radius for considering nearby residues (default: 15.0 Å)
            thresholds: Distance difference thresholds for scoring (default: 0.5, 1.0, 2.0, 4.0 Å)
            sequence_separation: Minimum |i-j| to include pair (default: 2, excludes i±1)
        """
        self.inclusion_radius = inclusion_radius
        self.thresholds = thresholds
        self.sequence_separation = sequence_separation
        logger.debug(f"Initialized LDDT calculator: radius={inclusion_radius}Å, "
                    f"thresholds={thresholds}, seq_sep>={sequence_separation}")

    def calculate_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distance matrix for coordinates.

        Args:
            coords: Array of shape (N, 3) with coordinates

        Returns:
            Distance matrix of shape (N, N)
        """
        # Efficient vectorized distance calculation
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances

    def calculate_lddt_score(self, ref_coords: np.ndarray, pred_coords: np.ndarray) -> Dict[str, float]:
        """
        Calculate LDDT score using canonical per-residue averaging.

        Canonical Protocol (CASP/AlphaFold):
            1. For each residue i:
               - Find neighbors j where ref_dist(i,j) < inclusion_radius AND |i-j| >= seq_separation
               - For each threshold t in {0.5, 1.0, 2.0, 4.0 Å}:
                   * Count neighbors where |ref_dist(i,j) - pred_dist(i,j)| < t
                   * Fraction_t = count / total_neighbors
               - LDDT_i = mean(Fraction across 4 thresholds)
            2. LDDT = mean(LDDT_i across residues with neighbors)

        This gives equal weight to each residue (not pair-weighted).

        Args:
            ref_coords: Reference Cα coordinates, shape (N, 3)
            pred_coords: Predicted Cα coordinates, shape (N, 3)

        Returns:
            Dict with LDDT score and per-threshold scores
        """
        if ref_coords.shape != pred_coords.shape:
            raise ValueError(f"Coordinate shapes don't match: {ref_coords.shape} vs {pred_coords.shape}")

        n_residues = ref_coords.shape[0]

        if n_residues < self.sequence_separation + 1:
            logger.warning(f"Structure has fewer than {self.sequence_separation + 1} residues, LDDT not meaningful")
            return {'lddt': 0.0}

        # Calculate distance matrices
        ref_distances = self.calculate_distance_matrix(ref_coords)
        pred_distances = self.calculate_distance_matrix(pred_coords)

        # Calculate distance differences
        distance_diff = np.abs(ref_distances - pred_distances)

        # Create sequence separation mask: |i - j| >= sequence_separation
        indices = np.arange(n_residues)
        sep_matrix = np.abs(indices[:, None] - indices[None, :])
        sequence_mask = sep_matrix >= self.sequence_separation

        # Create inclusion radius mask: ref_dist < inclusion_radius
        # Use upper triangle only (i < j) to avoid double counting
        radius_mask = (ref_distances < self.inclusion_radius) & (ref_distances > 0)
        upper_triangle = np.triu(np.ones((n_residues, n_residues), dtype=bool), k=1)

        # Combined mask: sequence separation AND within radius AND upper triangle
        valid_pairs = sequence_mask & radius_mask & upper_triangle

        # Per-residue LDDT calculation
        per_residue_lddt = []
        per_threshold_global = {t: [] for t in self.thresholds}

        for i in range(n_residues):
            # Find neighbors of residue i (both as i in (i,j) and as j in (k,i))
            neighbors_as_first = valid_pairs[i, :]  # i is first index
            neighbors_as_second = valid_pairs[:, i]  # i is second index
            neighbors = neighbors_as_first | neighbors_as_second

            n_neighbors = np.sum(neighbors)

            if n_neighbors == 0:
                # Residue has no neighbors (edge case, skip)
                continue

            # Get distance differences for this residue's neighbors
            # Need to handle both (i,j) and (j,i) pairs
            neighbor_diffs_first = distance_diff[i, neighbors_as_first]
            neighbor_diffs_second = distance_diff[neighbors_as_second, i]
            all_neighbor_diffs = np.concatenate([neighbor_diffs_first, neighbor_diffs_second])

            # Calculate fraction preserved at each threshold for this residue
            threshold_fractions = []
            for threshold in self.thresholds:
                preserved = np.sum(all_neighbor_diffs < threshold)
                fraction = preserved / len(all_neighbor_diffs)
                threshold_fractions.append(fraction)
                per_threshold_global[threshold].append(fraction)

            # LDDT for this residue: mean across thresholds
            residue_lddt = np.mean(threshold_fractions)
            per_residue_lddt.append(residue_lddt)

        if not per_residue_lddt:
            logger.warning("No residues with valid neighbors")
            return {'lddt': 0.0}

        # Final LDDT: equal-weight average across residues
        lddt_score = np.mean(per_residue_lddt) * 100  # Convert to 0-100 scale

        # Global per-threshold scores (for reference)
        per_threshold_scores = {}
        for threshold in self.thresholds:
            if per_threshold_global[threshold]:
                avg_fraction = np.mean(per_threshold_global[threshold]) * 100
                per_threshold_scores[f'lddt_threshold_{threshold}'] = avg_fraction

        result = {
            'lddt': lddt_score,
            **per_threshold_scores
        }

        logger.debug(f"LDDT score: {lddt_score:.2f} (from {len(per_residue_lddt)} residues with neighbors)")
        return result

    def calculate_from_pdb_files(self, ref_pdb: str, pred_pdb: str,
                                  ref_chain_id: Optional[str] = None,
                                  pred_chain_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Calculate LDDT score from two PDB files using proper residue mapping.

        Uses residue_mapper with separate chain specification for ref and pred.
        Critical for ESMFold: predictions use chain 'A', references use original chains.

        Args:
            ref_pdb: Path to reference PDB file
            pred_pdb: Path to predicted PDB file
            ref_chain_id: Chain from reference (None = first chain)
            pred_chain_id: Chain from predicted (None = first chain, 'A' for ESMFold)

        Returns:
            Dict with LDDT scores or None if calculation fails
        """
        try:
            # Use proper residue mapping with asymmetric chain specification
            ref_coords, pred_coords, residue_ids = map_residues(
                ref_pdb, pred_pdb,
                ref_chain_id=ref_chain_id,
                pred_chain_id=pred_chain_id
            )

            if ref_coords is None or pred_coords is None:
                logger.error("Failed to map residues between structures")
                return None

            if len(ref_coords) < self.sequence_separation + 1:
                logger.error(f"Too few mapped residues: {len(ref_coords)}")
                return None

            logger.debug(f"Mapped {len(residue_ids)} residues for LDDT calculation")

            # Calculate LDDT on properly mapped residues
            result = self.calculate_lddt_score(ref_coords, pred_coords)

            return result

        except Exception as e:
            logger.error(f"Failed to calculate LDDT: {e}")
            import traceback
            traceback.print_exc()
            return None


def calculate_lddt(ref_pdb: str, pred_pdb: str,
                   ref_chain_id: Optional[str] = None,
                   pred_chain_id: Optional[str] = None,
                   inclusion_radius: float = 15.0,
                   thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                   sequence_separation: int = 2) -> Optional[Dict[str, float]]:
    """
    Convenience function to calculate canonical LDDT from PDB files.

    Implements CASP/AlphaFold-standard LDDT with:
    - Proper residue mapping (not truncation)
    - Per-residue averaging (not pair-weighted)
    - Sequence separation to exclude bonded neighbors
    - Asymmetric chain specification (for ESMFold: ref uses original chain, pred uses 'A')

    Args:
        ref_pdb: Path to reference structure
        pred_pdb: Path to predicted structure
        ref_chain_id: Chain from reference (None = first chain)
        pred_chain_id: Chain from predicted (None = first, 'A' for ESMFold)
        inclusion_radius: Radius for considering nearby residues (default: 15.0 Å)
        thresholds: Distance difference thresholds (default: 0.5, 1.0, 2.0, 4.0 Å)
        sequence_separation: Minimum |i-j| to include (default: 2, excludes i±1)

    Returns:
        Dict with LDDT score (0-100) or None if calculation fails
    """
    calculator = LDDTCalculator(
        inclusion_radius=inclusion_radius,
        thresholds=thresholds,
        sequence_separation=sequence_separation
    )
    return calculator.calculate_from_pdb_files(ref_pdb, pred_pdb,
                                              ref_chain_id=ref_chain_id,
                                              pred_chain_id=pred_chain_id)


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate LDDT score between two structures")
    parser.add_argument("reference", help="Reference PDB file")
    parser.add_argument("predicted", help="Predicted PDB file")
    parser.add_argument("--radius", type=float, default=15.0, help="Inclusion radius (default: 15.0 Å)")

    args = parser.parse_args()

    # Calculate LDDT
    result = calculate_lddt(args.reference, args.predicted, inclusion_radius=args.radius)

    if result:
        print(f"\nLDDT Score: {result['lddt']:.2f}")
        print("\nPer-threshold scores:")
        for key, value in result.items():
            if key.startswith('lddt_threshold_'):
                threshold = key.split('_')[-1]
                print(f"  Threshold {threshold}Å: {value:.2f}")
    else:
        print("Failed to calculate LDDT")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

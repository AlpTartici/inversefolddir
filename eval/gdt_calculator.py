#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
GDT (Global Distance Test) Calculator

Implements canonical GDT_TS and GDT_HA following the LGA algorithm used in CASP.

GDT Metrics (Canonical Implementation):
    - GDT_TS: Average coverage at 1, 2, 4, 8 Å cutoffs
    - GDT_HA: Average coverage at 0.5, 1, 2, 4 Å cutoffs (high accuracy)

LGA Algorithm:
    For each distance cutoff:
    1. Seed multiple alignments using short fragments (length 3-5)
    2. For each seed:
       a. Superpose structures using Kabsch algorithm on seed
       b. Find inliers (residues within cutoff distance)
       c. Re-superpose on inliers only
       d. Iterate until inlier set converges
    3. Keep the largest inlier set across all seeds
    4. Coverage = |largest_inlier_set| / |mapped_residues|
    5. GDT = mean(coverages) × 100

This RANSAC-like approach is robust to flexible regions and matches CASP tools.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

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


class GDTCalculator:
    """
    Calculator for canonical GDT (Global Distance Test) scores.

    Implements LGA-style algorithm with:
    - Fragment seeding for robust initialization
    - Iterative inlier refinement per cutoff
    - Proper residue mapping
    - Per-chain evaluation
    """

    def __init__(self, fragment_length: int = 3, fragment_step: int = 1,
                 max_iterations: int = 10, convergence_threshold: float = 0.01):
        """
        Initialize GDT calculator.

        Args:
            fragment_length: Length of fragments for seeding (default: 3 residues)
            fragment_step: Step size for sliding fragments (default: 1)
            max_iterations: Maximum refinement iterations per seed (default: 10)
            convergence_threshold: Fraction change to consider converged (default: 0.01)
        """
        self.fragment_length = fragment_length
        self.fragment_step = fragment_step
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        logger.debug(f"Initialized GDT calculator: frag_len={fragment_length}, "
                    f"step={fragment_step}, max_iter={max_iterations}")

    def kabsch_superposition(self, ref_coords: np.ndarray, pred_coords: np.ndarray,
                            indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Perform optimal superposition using Kabsch algorithm.

        If indices provided, superposes only those residues but transforms all pred_coords.

        Args:
            ref_coords: Reference coordinates, shape (N, 3)
            pred_coords: Predicted coordinates to align, shape (N, 3)
            indices: Optional boolean array or indices to superpose on (default: all)

        Returns:
            Tuple of (transformed_pred_coords, rmsd_on_subset)
        """
        if indices is not None:
            # Superpose on subset, transform all
            ref_subset = ref_coords[indices]
            pred_subset = pred_coords[indices]
        else:
            ref_subset = ref_coords
            pred_subset = pred_coords

        # Center both coordinate sets
        ref_center = np.mean(ref_subset, axis=0)
        pred_center = np.mean(pred_subset, axis=0)

        ref_centered = ref_subset - ref_center
        pred_centered = pred_subset - pred_center

        # Compute covariance matrix
        H = pred_centered.T @ ref_centered

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)

        # Compute optimal rotation matrix (handle reflection)
        # For H = X^T @ Y, optimal rotation is R = V @ diag @ U^T
        V = Vt.T
        d = np.sign(np.linalg.det(V @ U.T))
        diag = np.diag([1.0, 1.0, d])
        rotation = V @ diag @ U.T

        # Apply transformation to ALL predicted coordinates
        pred_all_centered = pred_coords - pred_center
        transformed_pred = (rotation @ pred_all_centered.T).T + ref_center

        # Calculate RMSD on the subset used for superposition
        diff = ref_subset - transformed_pred[indices] if indices is not None else ref_subset - transformed_pred
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        return transformed_pred, rmsd

    def find_inliers(self, ref_coords: np.ndarray, transformed_pred: np.ndarray,
                    cutoff: float) -> np.ndarray:
        """
        Find residues within cutoff distance after superposition.

        Args:
            ref_coords: Reference coordinates, shape (N, 3)
            transformed_pred: Transformed predicted coordinates, shape (N, 3)
            cutoff: Distance threshold in Angstroms

        Returns:
            Boolean array of inliers (True = within cutoff)
        """
        distances = np.sqrt(np.sum((ref_coords - transformed_pred) ** 2, axis=1))
        # Use <= not < (canonical GDT definition)
        return distances <= cutoff

    def refine_inliers(self, ref_coords: np.ndarray, pred_coords: np.ndarray,
                      initial_inliers: np.ndarray, cutoff: float) -> np.ndarray:
        """
        Iteratively refine inlier set: superpose on inliers, find new inliers, repeat.

        Args:
            ref_coords: Reference coordinates, shape (N, 3)
            pred_coords: Predicted coordinates, shape (N, 3)
            initial_inliers: Initial inlier boolean mask
            cutoff: Distance cutoff in Angstroms

        Returns:
            Final converged inlier boolean mask
        """
        current_inliers = initial_inliers.copy()

        for iteration in range(self.max_iterations):
            n_current = np.sum(current_inliers)

            if n_current < 3:
                # Need at least 3 points for meaningful superposition
                break

            # Superpose on current inliers
            transformed_pred, _ = self.kabsch_superposition(
                ref_coords, pred_coords, indices=current_inliers
            )

            # Find new inliers
            new_inliers = self.find_inliers(ref_coords, transformed_pred, cutoff)
            n_new = np.sum(new_inliers)

            # Check convergence
            if n_new == n_current:
                # Converged (no change in inlier count)
                current_inliers = new_inliers
                break
            elif abs(n_new - n_current) / max(n_current, 1) < self.convergence_threshold:
                # Converged (change below threshold)
                current_inliers = new_inliers
                break

            current_inliers = new_inliers

        return current_inliers

    def gdt_for_cutoff(self, ref_coords: np.ndarray, pred_coords: np.ndarray,
                      cutoff: float) -> Tuple[int, np.ndarray]:
        """
        Calculate maximum coverage for a single distance cutoff using fragment seeding.

        Args:
            ref_coords: Reference coordinates, shape (N, 3)
            pred_coords: Predicted coordinates, shape (N, 3)
            cutoff: Distance cutoff in Angstroms

        Returns:
            Tuple of (max_inlier_count, best_inlier_mask)
        """
        n_residues = len(ref_coords)

        if n_residues < self.fragment_length:
            # Structure too small for fragment seeding, use full structure
            transformed_pred, _ = self.kabsch_superposition(ref_coords, pred_coords)
            inliers = self.find_inliers(ref_coords, transformed_pred, cutoff)
            return np.sum(inliers), inliers

        max_inliers = 0
        best_inlier_mask = np.zeros(n_residues, dtype=bool)

        # Seed alignments with sliding fragments
        for start in range(0, n_residues - self.fragment_length + 1, self.fragment_step):
            end = start + self.fragment_length
            fragment_indices = np.arange(start, end)

            # Superpose on this fragment
            transformed_pred, _ = self.kabsch_superposition(
                ref_coords, pred_coords, indices=fragment_indices
            )

            # Find initial inliers
            initial_inliers = self.find_inliers(ref_coords, transformed_pred, cutoff)

            # Refine iteratively
            final_inliers = self.refine_inliers(
                ref_coords, pred_coords, initial_inliers, cutoff
            )

            n_inliers = np.sum(final_inliers)

            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_inlier_mask = final_inliers

        return max_inliers, best_inlier_mask

    def calculate_gdt_scores(self, ref_coords: np.ndarray, pred_coords: np.ndarray,
                            cutoffs_ts: List[float] = [1.0, 2.0, 4.0, 8.0],
                            cutoffs_ha: List[float] = [0.5, 1.0, 2.0, 4.0]) -> Dict[str, float]:
        """
        Calculate GDT scores using canonical LGA-style algorithm.

        For each cutoff, performs fragment seeding and iterative refinement
        to find the maximum inlier set. This is robust to flexible regions.

        Args:
            ref_coords: Reference Cα coordinates, shape (N, 3)
            pred_coords: Predicted Cα coordinates, shape (N, 3)
            cutoffs_ts: Distance cutoffs for GDT_TS (default: [1, 2, 4, 8] Å)
            cutoffs_ha: Distance cutoffs for GDT_HA (default: [0.5, 1, 2, 4] Å)

        Returns:
            Dict with GDT scores at each cutoff plus GDT_TS and GDT_HA
        """
        if ref_coords.shape != pred_coords.shape:
            raise ValueError(f"Coordinate shapes don't match: {ref_coords.shape} vs {pred_coords.shape}")

        n_residues = ref_coords.shape[0]

        if n_residues < 3:
            logger.warning("Structure has fewer than 3 residues, GDT not meaningful")
            return {
                'gdt_ts': 0.0,
                'gdt_ha': 0.0,
                'gdt_1': 0.0,
                'gdt_2': 0.0,
                'gdt_4': 0.0,
                'gdt_8': 0.0,
                'gdt_05': 0.0
            }

        # Calculate coverage for each cutoff
        all_cutoffs = sorted(set(cutoffs_ts + cutoffs_ha))
        cutoff_coverages = {}

        logger.debug(f"Calculating GDT for {n_residues} residues at {len(all_cutoffs)} cutoffs")

        for cutoff in all_cutoffs:
            max_inliers, _ = self.gdt_for_cutoff(ref_coords, pred_coords, cutoff)
            coverage = (max_inliers / n_residues) * 100  # Convert to percentage

            # Format cutoff for key (0.5 -> '05', 1.0 -> '1', etc.)
            cutoff_str = str(cutoff).replace('.', '')
            if cutoff_str.endswith('0'):
                cutoff_str = cutoff_str[:-1]

            cutoff_coverages[f'gdt_{cutoff_str}'] = coverage
            logger.debug(f"  Cutoff {cutoff}Å: {max_inliers}/{n_residues} = {coverage:.2f}%")

        # Calculate GDT_TS (average of 1, 2, 4, 8 Å)
        ts_coverages = []
        for cutoff in cutoffs_ts:
            cutoff_str = str(cutoff).replace('.', '')
            if cutoff_str.endswith('0'):
                cutoff_str = cutoff_str[:-1]
            ts_coverages.append(cutoff_coverages[f'gdt_{cutoff_str}'])
        gdt_ts = np.mean(ts_coverages)

        # Calculate GDT_HA (average of 0.5, 1, 2, 4 Å)
        ha_coverages = []
        for cutoff in cutoffs_ha:
            cutoff_str = str(cutoff).replace('.', '')
            if cutoff_str.endswith('0'):
                cutoff_str = cutoff_str[:-1]
            ha_coverages.append(cutoff_coverages[f'gdt_{cutoff_str}'])
        gdt_ha = np.mean(ha_coverages)

        result = {
            'gdt_ts': gdt_ts,
            'gdt_ha': gdt_ha,
            **cutoff_coverages
        }

        logger.debug(f"GDT_TS: {gdt_ts:.2f}, GDT_HA: {gdt_ha:.2f}")
        return result

    def calculate_from_pdb_files(self, ref_pdb: str, pred_pdb: str,
                                  ref_chain_id: Optional[str] = None,
                                  pred_chain_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Calculate GDT scores from two PDB files using proper residue mapping.

        Uses residue_mapper with separate chain specification for ref and pred.
        Critical for ESMFold: predictions use chain 'A', references use original chains.

        Args:
            ref_pdb: Path to reference PDB file
            pred_pdb: Path to predicted PDB file
            ref_chain_id: Chain from reference (None = first chain)
            pred_chain_id: Chain from predicted (None = first, 'A' for ESMFold)

        Returns:
            Dict with GDT scores or None if calculation fails
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

            if len(ref_coords) < 3:
                logger.error(f"Too few mapped residues: {len(ref_coords)}")
                return None

            logger.debug(f"Mapped {len(residue_ids)} residues for GDT calculation")

            # Calculate GDT scores
            result = self.calculate_gdt_scores(ref_coords, pred_coords)

            return result

        except Exception as e:
            logger.error(f"Failed to calculate GDT: {e}")
            import traceback
            traceback.print_exc()
            return None


def calculate_gdt(ref_pdb: str, pred_pdb: str,
                  ref_chain_id: Optional[str] = None,
                  pred_chain_id: Optional[str] = None,
                  fragment_length: int = 3) -> Optional[Dict[str, float]]:
    """
    Convenience function to calculate canonical GDT from PDB files.

    Implements LGA-style GDT with:
    - Fragment seeding for robust initialization
    - Iterative inlier refinement per cutoff
    - Proper residue mapping (not truncation)
    - Asymmetric chain specification (for ESMFold: ref uses original chain, pred uses 'A')

    Args:
        ref_pdb: Path to reference structure
        pred_pdb: Path to predicted structure
        ref_chain_id: Chain from reference (None = first chain)
        pred_chain_id: Chain from predicted (None = first, 'A' for ESMFold)
        fragment_length: Fragment size for seeding (default: 3)

    Returns:
        Dict with GDT_TS, GDT_HA, and individual cutoff scores (0-100) or None if fails
    """
    calculator = GDTCalculator(fragment_length=fragment_length)
    return calculator.calculate_from_pdb_files(ref_pdb, pred_pdb,
                                              ref_chain_id=ref_chain_id,
                                              pred_chain_id=pred_chain_id)


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate GDT scores between two structures")
    parser.add_argument("reference", help="Reference PDB file")
    parser.add_argument("predicted", help="Predicted PDB file")
    parser.add_argument("--ref-chain", default=None, help="Chain ID for reference structure")
    parser.add_argument("--pred-chain", default=None, help="Chain ID for predicted structure")
    parser.add_argument("--frag-length", type=int, default=3, help="Fragment length for seeding")

    args = parser.parse_args()

    # Calculate GDT
    result = calculate_gdt(args.reference, args.predicted,
                          ref_chain_id=args.ref_chain,
                          pred_chain_id=args.pred_chain,
                          fragment_length=args.frag_length)

    if result:
        print(f"\nGDT_TS: {result['gdt_ts']:.2f}")
        print(f"GDT_HA: {result['gdt_ha']:.2f}")
        print("\nIndividual cutoffs:")
        for key, value in sorted(result.items()):
            if key.startswith('gdt_') and key not in ['gdt_ts', 'gdt_ha']:
                cutoff = key.replace('gdt_', '')
                if cutoff == '05':
                    cutoff = '0.5'
                print(f"  {cutoff}Å: {value:.2f}%")
    else:
        print("Failed to calculate GDT")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

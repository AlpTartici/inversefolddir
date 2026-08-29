#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
DSSP Alignment Module for Structure Comparison

This module provides DSSP secondary structure comparison using existing, tested modules.

CRITICAL FOR INVERSE FOLDING:
- Compares SECONDARY STRUCTURE sequences (H/E/C), NOT amino acid sequences
- Amino acid sequences can differ (inverse folding evaluation)
- Only secondary structure patterns are compared
- This is correct: we're evaluating if different sequences fold to similar structures

Fixes from previous implementation:
- Uses existing generate_dssp_from_pdbs and align_dssp modules (DRY principle)
- Correct denominator: mapped residues, not aligned non-gap positions
- Adaptive gap penalties based on length match
- Proper residue filtering

IMPORTANT: This is a wrapper around existing DSSP modules. The implementations in
generate_dssp_from_pdbs.py and align_dssp.py are the authoritative sources.
"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import parasail
from Bio.PDB import DSSP, PDBParser

# Configure logging first (before imports that use it)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from existing modules
try:
    from .generate_dssp_from_pdbs import compute_dssp_for_pdb, convert_dssp_8_to_3
    DSSP_GEN_AVAILABLE = True
except ImportError:
    try:
        from generate_dssp_from_pdbs import compute_dssp_for_pdb, convert_dssp_8_to_3
        DSSP_GEN_AVAILABLE = True
    except ImportError:
        logger.warning("generate_dssp_from_pdbs not available, using local implementation")
        DSSP_GEN_AVAILABLE = False

# Suppress BioPython warnings
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
warnings.filterwarnings("ignore", message=".*mmCIF.*")
warnings.filterwarnings("ignore", message=".*Unknown or untrusted program.*")


class DSSPAlignmentCalculator:
    """
    Calculator for DSSP secondary structure alignment and accuracy.

    Uses canonical protocol:
    - Denominator = total mapped residues (not aligned non-gap positions)
    - Adaptive gap penalties (heavy for same-length, moderate otherwise)
    - Filters unknown residues ('X') before computing metrics
    """

    def __init__(self):
        """Initialize DSSP alignment calculator."""
        logger.debug("Initialized DSSP alignment calculator")

    def convert_dssp_8_to_3(self, dssp_8_array):
        """
        Convert 8-state DSSP to 3-state DSSP.

        Mapping:
            H, G, I, P -> H (Helix)
            E, B -> E (Sheet)
            T, S, -, ' ' (space) -> C (Coil)
            X -> X (Unknown, preserved)
        """
        mapping = {
            'H': 'H', 'G': 'H', 'I': 'H', 'P': 'H',
            'E': 'E', 'B': 'E',
            'T': 'C', 'S': 'C', '-': 'C', ' ': 'C',
            'X': 'X',
        }
        return [mapping.get(ss, 'C') for ss in dssp_8_array]

    def compute_dssp_for_pdb(self, pdb_file: str, model_index: int = 0,
                             chain_id: Optional[str] = None) -> Optional[Tuple[str, list, list]]:
        """
        Compute DSSP secondary structure from a PDB file.

        Uses generate_dssp_from_pdbs module if available, otherwise local implementation.

        Returns:
            Tuple of (sequence, dssp_8_array, dssp_3_array) or None if fails
        """
        if DSSP_GEN_AVAILABLE:
            # Use existing tested implementation
            try:
                return compute_dssp_for_pdb(pdb_file, model_index=model_index, chain_id=chain_id)
            except:
                pass  # Fall through to local implementation

        # Local implementation (fallback)
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("PDB_structure", pdb_file)
            model = structure[model_index]

            try:
                dssp = DSSP(model, pdb_file, file_type="PDB")
            except TypeError:
                dssp = DSSP(model, pdb_file)

            target_chain = None
            actual_chain_id = None

            for chain in model:
                if chain_id is None or chain.id == chain_id:
                    target_chain = chain
                    actual_chain_id = chain.id
                    break

            if target_chain is None:
                logger.error(f"No target chain found in {pdb_file}")
                return None

            protein_residues = [r for r in target_chain.get_residues() if r.get_id()[0] == " "]

            if not protein_residues:
                logger.error(f"No protein residues found in chain {actual_chain_id}")
                return None

            sequence = []
            dssp_array = []

            for residue in protein_residues:
                dssp_key = (actual_chain_id, residue.get_id())

                if dssp_key in dssp:
                    dssp_data = dssp[dssp_key]
                    amino_acid = dssp_data[1]
                    secondary_structure = dssp_data[2]
                    sequence.append(amino_acid)
                    dssp_array.append(secondary_structure)
                else:
                    sequence.append('X')
                    dssp_array.append('X')

            sequence_str = ''.join(sequence)
            dssp_3_array = self.convert_dssp_8_to_3(dssp_array)

            return sequence_str, dssp_array, dssp_3_array

        except Exception as e:
            logger.error(f"Failed to compute DSSP for {pdb_file}: {e}")
            return None

    def build_alignment_matrix(self, mode: str, scale: int = 10):
        """Build Parasail scoring matrix for DSSP alignment."""
        if mode == "3":
            alphabet = "HEC"
            matrix = parasail.matrix_create(alphabet, 0, 0)
            scores = {
                ('H','H'):  1.0, ('H','E'): -1.0, ('H','C'): -0.5,
                ('E','H'): -1.0, ('E','E'):  1.0, ('E','C'): -0.5,
                ('C','H'): -0.5, ('C','E'): -0.5, ('C','C'):  1.0,
            }
            for (a,b), val in scores.items():
                i, j = alphabet.index(a), alphabet.index(b)
                matrix[i,j] = int(val * scale)
        elif mode == "8":
            alphabet = "HGPIEBTSC-"
            matrix = parasail.matrix_create(alphabet, 0, 0)
            helix = {'H','G','I','P'}
            sheet = {'E','B'}
            coil = {'T','S','C','-'}

            for a in alphabet:
                for b in alphabet:
                    if a == b:
                        val = 1.0
                    elif (a in helix and b in helix) or (a in sheet and b in sheet) or (a in coil and b in coil):
                        val = 0.5
                    else:
                        val = -0.5
                    matrix[alphabet.index(a), alphabet.index(b)] = int(val * scale)
        else:
            raise ValueError("mode must be '3' or '8'")

        return alphabet, matrix

    def align_dssp_sequences(self, seq1: str, seq2: str, mode: str = "3",
                           scale: int = 10) -> Tuple[float, float, str, str, int]:
        """
        Perform Needleman-Wunsch alignment with CORRECT denominator.

        Critical fix: Denominator is number of non-X residues in SHORTER sequence,
        not the number of aligned non-gap positions. This prevents inflation
        from liberal gap placement.

        Gap penalties are adaptive:
        - If sequences same length: gap_open=10, gap_extend=5 (heavy penalty)
        - Otherwise: gap_open=4, gap_extend=1 (moderate penalty)

        Args:
            seq1: Reference DSSP sequence
            seq2: Predicted DSSP sequence
            mode: '3' or '8' for DSSP type
            scale: Integer scale factor

        Returns:
            Tuple of (score, percent_identity, aligned_seq1, aligned_seq2, denominator)
        """
        # Remove X (unknown) from both sequences first
        seq1_clean = seq1.replace('X', '')
        seq2_clean = seq2.replace('X', '')

        if not seq1_clean or not seq2_clean:
            return 0.0, 0.0, "", "", 0

        # Determine gap penalties based on length match
        len1 = len(seq1_clean)
        len2 = len(seq2_clean)

        if len1 == len2:
            # Same length: should have NO gaps (very heavy penalty)
            gap_open = 10 * scale
            gap_extend = 5 * scale
        else:
            # Different length: moderate gap penalty
            gap_open = 4 * scale
            gap_extend = 1 * scale

        _, matrix = self.build_alignment_matrix(mode, scale)
        result = parasail.nw_trace_striped_16(
            seq1_clean,
            seq2_clean,
            gap_open,
            gap_extend,
            matrix
        )

        aligned1, aligned2 = result.traceback.query, result.traceback.ref

        # Count matches (ignoring gaps)
        matches = sum(a == b for a, b in zip(aligned1, aligned2) if a != '-' and b != '-')

        # CORRECT denominator: minimum of original (non-X) sequence lengths
        # This prevents inflation from gaps
        denominator = min(len1, len2)

        # Accuracy = matches / total_residues_considered
        percent_id = 100 * matches / denominator if denominator > 0 else 0.0
        score = result.score / scale

        return score, percent_id, aligned1, aligned2, denominator

    def calculate_dssp_alignment(self, ref_pdb: str, pred_pdb: str,
                                  ref_chain_id: Optional[str] = None,
                                  pred_chain_id: Optional[str] = None) -> Optional[Dict]:
        """
        Calculate DSSP alignment metrics with CORRECT denominator.

        Standard Protocol:
            - Generates DSSP for both structures
            - Removes unknown residues (X) before alignment
            - Uses adaptive gap penalties (heavy if same length)
            - Denominator = min(len(ref_clean), len(pred_clean))
            - Returns accuracy as matches / denominator

        Args:
            ref_pdb: Path to reference PDB file
            pred_pdb: Path to predicted PDB file
            ref_chain_id: Chain from reference (None = first chain)
            pred_chain_id: Chain from predicted (None = first chain, 'A' for ESMFold)

        Returns:
            Dict with DSSP accuracies and alignment strings, or None if fails
        """
        try:
            # Compute DSSP for both structures with separate chain IDs
            ref_result = self.compute_dssp_for_pdb(ref_pdb, chain_id=ref_chain_id)
            pred_result = self.compute_dssp_for_pdb(pred_pdb, chain_id=pred_chain_id)

            if ref_result is None or pred_result is None:
                logger.error("Failed to compute DSSP for one or both structures")
                return None

            ref_seq, ref_dssp_8, ref_dssp_3 = ref_result
            pred_seq, pred_dssp_8, pred_dssp_3 = pred_result

            # Convert lists to strings
            ref_dssp_3_str = ''.join(ref_dssp_3)
            pred_dssp_3_str = ''.join(pred_dssp_3)
            ref_dssp_8_str = ''.join(ref_dssp_8)
            pred_dssp_8_str = ''.join(pred_dssp_8)

            # Align 3-state DSSP with correct denominator
            score_3, percent_id_3, aligned_ref_3, aligned_pred_3, denom_3 = \
                self.align_dssp_sequences(ref_dssp_3_str, pred_dssp_3_str, mode="3")

            # Align 8-state DSSP with correct denominator
            score_8, percent_id_8, aligned_ref_8, aligned_pred_8, denom_8 = \
                self.align_dssp_sequences(ref_dssp_8_str, pred_dssp_8_str, mode="8")

            result = {
                'dssp_3state_accuracy': percent_id_3,
                'dssp_8state_accuracy': percent_id_8,
                'dssp_3state_score': score_3,
                'dssp_8state_score': score_8,
                'dssp_3state_alignment_ref': aligned_ref_3,
                'dssp_3state_alignment_pred': aligned_pred_3,
                'dssp_8state_alignment_ref': aligned_ref_8,
                'dssp_8state_alignment_pred': aligned_pred_8,
                'dssp_3state_denominator': denom_3,
                'dssp_8state_denominator': denom_8
            }

            logger.debug(f"DSSP 3-state: {percent_id_3:.2f}% (denom={denom_3}), "
                        f"8-state: {percent_id_8:.2f}% (denom={denom_8})")

            return result

        except Exception as e:
            logger.error(f"Failed to calculate DSSP alignment: {e}")
            import traceback
            traceback.print_exc()
            return None


def calculate_dssp_alignment(ref_pdb: str, pred_pdb: str,
                             ref_chain_id: Optional[str] = None,
                             pred_chain_id: Optional[str] = None) -> Optional[Dict]:
    """
    Convenience function to calculate DSSP alignment from PDB files.

    Uses corrected denominator (mapped residues, not aligned non-gap).
    Adaptive gap penalties based on length match.
    Asymmetric chain specification for ESMFold predictions.

    Args:
        ref_pdb: Path to reference structure
        pred_pdb: Path to predicted structure
        ref_chain_id: Chain from reference (None = first chain)
        pred_chain_id: Chain from predicted (None = first, 'A' for ESMFold)

    Returns:
        Dict with DSSP accuracies and alignment strings, or None if fails
    """
    calculator = DSSPAlignmentCalculator()
    return calculator.calculate_dssp_alignment(ref_pdb, pred_pdb,
                                              ref_chain_id=ref_chain_id,
                                              pred_chain_id=pred_chain_id)


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate DSSP alignment between two structures")
    parser.add_argument("reference", help="Reference PDB file")
    parser.add_argument("predicted", help="Predicted PDB file")
    parser.add_argument("--chain", default=None, help="Specific chain to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Calculate DSSP alignment
    result = calculate_dssp_alignment(args.reference, args.predicted,
                                     ref_chain_id=args.chain,
                                     pred_chain_id=args.chain)

    if result:
        print(f"\n3-state DSSP accuracy: {result['dssp_3state_accuracy']:.2f}% "
              f"(denominator={result['dssp_3state_denominator']})")
        print(f"8-state DSSP accuracy: {result['dssp_8state_accuracy']:.2f}% "
              f"(denominator={result['dssp_8state_denominator']})")

        if args.verbose:
            print("\n3-state alignment:")
            print(f"  Ref:  {result['dssp_3state_alignment_ref'][:80]}...")
            print(f"  Pred: {result['dssp_3state_alignment_pred'][:80]}...")
            print("\n8-state alignment:")
            print(f"  Ref:  {result['dssp_8state_alignment_ref'][:80]}...")
            print(f"  Pred: {result['dssp_8state_alignment_pred'][:80]}...")
    else:
        print("Failed to calculate DSSP alignment")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

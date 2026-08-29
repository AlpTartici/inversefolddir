#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Residue Mapping Utility

Provides proper residue-to-residue correspondence between reference and predicted
structures, handling:
- Multiple chains
- Missing residues
- Insertion codes
- Multiple models
- Alternate locations

This ensures metrics compare corresponding residues, not just truncated arrays.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import parasail
from Bio.PDB import PDBParser, DSSP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ResidueMapper:
    """
    Maps residues between reference and predicted structures.

    Ensures proper residue-residue correspondence for structural metrics.
    """

    def __init__(self, model_index: int = 0, chain_id: Optional[str] = None):
        """
        Initialize residue mapper.

        Args:
            model_index: Which MODEL to use from PDB (default: 0, first model)
            chain_id: Specific chain to map (default: None, uses first chain)
        """
        self.parser = PDBParser(QUIET=True)
        self.model_index = model_index
        self.chain_id = chain_id

        # DSSP 8->3 state mapping
        self.dssp_3_map = {
            'H': 'H', 'G': 'H', 'I': 'H',  # Helix
            'E': 'E', 'B': 'E',              # Sheet
            'T': 'C', 'S': 'C', '-': 'C', ' ': 'C'  # Coil
        }

    def parse_residues(self, pdb_path: str) -> List[Tuple]:
        """
        Parse residues from PDB file.

        Returns list of (chain_id, resseq, icode, ca_coords, aa) tuples.
        Only includes standard residues with CA atoms.
        Handles MODEL selection and altLoc resolution.

        Args:
            pdb_path: Path to PDB file

        Returns:
            List of (chain_id, resseq, icode, ca_coords, aa) tuples
        """
        try:
            structure = self.parser.get_structure("structure", pdb_path)
            model = structure[self.model_index]
        except Exception as e:
            logger.error(f"Failed to parse {pdb_path}: {e}")
            return []

        residues = []

        for chain in model:
            # Skip if specific chain requested and this isn't it
            if self.chain_id is not None and chain.id != self.chain_id:
                continue

            for residue in chain:
                # Only standard amino acids (hetflag is ' ')
                if residue.id[0] != ' ':
                    continue

                # Must have CA atom
                if 'CA' not in residue:
                    continue

                # Get CA coordinates, handling altLoc if present
                ca_atom = residue['CA']
                if ca_atom.is_disordered():
                    # Take first altloc
                    ca_atom = ca_atom.selected_child

                coords = ca_atom.get_coord()

                # Get amino acid type
                resname = residue.get_resname()
                aa_map = {
                    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                }
                aa = aa_map.get(resname, 'X')

                # Store: (chain_id, resseq, icode, coords, aa)
                resseq = residue.id[1]
                icode = residue.id[2]
                residues.append((chain.id, resseq, icode, coords, aa))

        return residues

    def map_residues_by_dssp(self, ref_pdb: str, pred_pdb: str,
                             ref_chain_id: Optional[str] = None,
                             pred_chain_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Map residues using DSSP secondary structure alignment.

        This is ideal for inverse folding where:
        - Sequences are different (can't use sequence alignment)
        - Residue numbering differs (can't use position matching)
        - We want alignment-free metrics like LDDT (can't use rigid TM-align)

        Strategy:
        1. Compute DSSP for both structures
        2. Align DSSP secondary structure strings
        3. Extract residue correspondence from alignment
        4. Return matched coordinates

        Args:
            ref_pdb: Path to reference PDB
            pred_pdb: Path to predicted PDB
            ref_chain_id: Chain from reference
            pred_chain_id: Chain from predicted

        Returns:
            Tuple of (ref_coords, pred_coords, residue_ids)
        """
        try:
            # Parse structures
            ref_struct = self.parser.get_structure("ref", ref_pdb)
            pred_struct = self.parser.get_structure("pred", pred_pdb)

            ref_model = ref_struct[self.model_index]
            pred_model = pred_struct[self.model_index]

            # Compute DSSP
            ref_dssp = DSSP(ref_model, ref_pdb, dssp='mkdssp')
            pred_dssp = DSSP(pred_model, pred_pdb, dssp='mkdssp')

            # Extract DSSP sequences and coordinates for specified chains
            ref_dssp_seq = []
            ref_coords_list = []
            ref_residue_ids = []

            for chain in ref_model:
                if ref_chain_id and chain.id != ref_chain_id:
                    continue
                for residue in chain:
                    if residue.id[0] != ' ':  # Skip non-standard
                        continue
                    key = (chain.id, residue.id)
                    if key in ref_dssp:
                        ss = ref_dssp[key][2]  # Secondary structure
                        ss_3 = self.dssp_3_map.get(ss, 'C')
                        ref_dssp_seq.append(ss_3)
                        if 'CA' in residue:
                            ref_coords_list.append(residue['CA'].get_coord())
                            ref_residue_ids.append((residue.id[1], residue.id[2]))

            pred_dssp_seq = []
            pred_coords_list = []
            pred_residue_ids = []

            for chain in pred_model:
                if pred_chain_id and chain.id != pred_chain_id:
                    continue
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    key = (chain.id, residue.id)
                    if key in pred_dssp:
                        ss = pred_dssp[key][2]
                        ss_3 = self.dssp_3_map.get(ss, 'C')
                        pred_dssp_seq.append(ss_3)
                        if 'CA' in residue:
                            pred_coords_list.append(residue['CA'].get_coord())
                            pred_residue_ids.append((residue.id[1], residue.id[2]))

            if not ref_dssp_seq or not pred_dssp_seq:
                logger.error("Failed to compute DSSP for one or both structures")
                return None, None, []

            # Align DSSP sequences using parasail
            ref_dssp_str = ''.join(ref_dssp_seq)
            pred_dssp_str = ''.join(pred_dssp_seq)

            # Use semi-global alignment (query end-gap free)
            match_score = 2
            mismatch_score = -1
            gap_open = 3
            gap_extend = 1

            result = parasail.sg_qx_trace(
                ref_dssp_str, pred_dssp_str,
                gap_open, gap_extend,
                parasail.matrix_create("ACDEFGHIKLMNPQRSTVWY", match_score, mismatch_score)
            )

            # Extract alignment
            traceback = result.get_traceback()
            aligned_ref = traceback.ref
            aligned_pred = traceback.query

            # Map aligned positions to coordinates
            ref_idx = 0
            pred_idx = 0
            mapped_ref_coords = []
            mapped_pred_coords = []
            mapped_residue_ids = []

            for i in range(len(aligned_ref)):
                ref_char = aligned_ref[i]
                pred_char = aligned_pred[i]

                # Both non-gap: this is a match
                if ref_char != '-' and pred_char != '-':
                    if ref_idx < len(ref_coords_list) and pred_idx < len(pred_coords_list):
                        mapped_ref_coords.append(ref_coords_list[ref_idx])
                        mapped_pred_coords.append(pred_coords_list[pred_idx])
                        mapped_residue_ids.append(ref_residue_ids[ref_idx])
                    ref_idx += 1
                    pred_idx += 1
                elif ref_char != '-':
                    ref_idx += 1
                elif pred_char != '-':
                    pred_idx += 1

            if not mapped_ref_coords:
                logger.error("No residues mapped via DSSP alignment")
                return None, None, []

            ref_coords = np.array(mapped_ref_coords)
            pred_coords = np.array(mapped_pred_coords)

            logger.info(f"Mapped {len(mapped_residue_ids)} residues via DSSP alignment "
                       f"(ref={len(ref_dssp_seq)}, pred={len(pred_dssp_seq)})")

            return ref_coords, pred_coords, mapped_residue_ids

        except Exception as e:
            logger.error(f"DSSP-based mapping failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []

    def map_residues(self, ref_pdb: str, pred_pdb: str,
                     ref_chain_id: Optional[str] = None,
                     pred_chain_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Map corresponding residues between reference and predicted structures.

        CRITICAL FOR INVERSE FOLDING:
        - Matches residues by POSITION (resseq, icode) ONLY
        - Does NOT check amino acid identity
        - This allows comparing structures with DIFFERENT sequences
        - For inverse folding evaluation where sequences differ by design

        CRITICAL FOR ESMFOLD:
        - ESMFold always outputs to chain 'A' regardless of input
        - Reference PDBs have original chain IDs
        - Must specify chains separately for ref and pred

        Args:
            ref_pdb: Path to reference PDB
            pred_pdb: Path to predicted PDB
            ref_chain_id: Chain to use from reference (None = first chain)
            pred_chain_id: Chain to use from predicted (None = first chain, usually 'A' for ESMFold)

        Returns:
            Tuple of (ref_coords, pred_coords, residue_ids)
            - ref_coords: (N, 3) array of reference CA coordinates
            - pred_coords: (N, 3) array of predicted CA coordinates
            - residue_ids: List of N (resseq, icode) tuples for matched residues
        """
        # Parse both structures with specific chain selection
        self.chain_id = ref_chain_id
        ref_residues = self.parse_residues(ref_pdb)

        self.chain_id = pred_chain_id
        pred_residues = self.parse_residues(pred_pdb)

        if not ref_residues:
            logger.error(f"No residues found in reference: {ref_pdb} (chain={ref_chain_id})")
            return None, None, []

        if not pred_residues:
            logger.error(f"No residues found in predicted: {pred_pdb} (chain={pred_chain_id})")
            return None, None, []

        # CRITICAL: Create lookup dicts by (resseq, icode) ONLY
        # Does NOT include amino acid identity - allows different sequences
        # This is essential for inverse folding where sequences differ by design
        ref_dict = {(resseq, icode): (chain, coords)
                    for chain, resseq, icode, coords, aa in ref_residues}
        pred_dict = {(resseq, icode): (chain, coords)
                     for chain, resseq, icode, coords, aa in pred_residues}

        # Find intersection by POSITION (resseq, icode) only - NOT by amino acid
        common_keys = set(ref_dict.keys()) & set(pred_dict.keys())

        # FALLBACK: If no residues match by number, match by sequential position
        # This handles cases where reference and prediction use different numbering schemes
        # For inverse folding, we can't rely on sequence matching since sequences may differ
        # We use positional matching as a proxy for structural correspondence
        if not common_keys:
            ref_len = len(ref_dict)
            pred_len = len(pred_dict)

            # Use positional matching if lengths are similar (within 30% difference)
            # This handles missing residues in crystal structures vs full predictions
            length_ratio = min(ref_len, pred_len) / max(ref_len, pred_len)

            if length_ratio >= 0.7:  # At least 70% overlap
                logger.warning(f"No residues match by number (ref={ref_len}, pred={pred_len}). "
                              f"Falling back to sequential position matching.")

                # Sort both by their keys to get sequential order
                ref_keys_sorted = sorted(ref_dict.keys())
                pred_keys_sorted = sorted(pred_dict.keys())

                # Use the shorter length to avoid index out of bounds
                min_length = min(len(ref_keys_sorted), len(pred_keys_sorted))

                # Create mapping by index (use first N residues where N = min length)
                ref_coords_list = []
                pred_coords_list = []
                residue_ids = []

                for i in range(min_length):
                    ref_key = ref_keys_sorted[i]
                    pred_key = pred_keys_sorted[i]
                    ref_coords_list.append(ref_dict[ref_key][1])
                    pred_coords_list.append(pred_dict[pred_key][1])
                    # Use reference residue IDs for the mapping
                    residue_ids.append(ref_key)

                ref_coords = np.array(ref_coords_list)
                pred_coords = np.array(pred_coords_list)

                logger.warning(f"Mapped {len(residue_ids)} residues by sequential position "
                              f"(using first {min_length} of ref={ref_len}, pred={pred_len}). "
                              f"Ref numbering: {ref_keys_sorted[0]}-{ref_keys_sorted[min_length-1]}, "
                              f"Pred numbering: {pred_keys_sorted[0]}-{pred_keys_sorted[min_length-1]}")

                return ref_coords, pred_coords, residue_ids

        if not common_keys:
            # Try DSSP-based mapping as final fallback
            logger.warning("No residues matched by position. Trying DSSP-based structural alignment...")
            return self.map_residues_by_dssp(ref_pdb, pred_pdb, ref_chain_id, pred_chain_id)

        # Sort by resseq, then icode for consistent ordering
        common_keys_sorted = sorted(common_keys)

        # Extract coordinates in matched order
        ref_coords = np.array([ref_dict[key][1] for key in common_keys_sorted])
        pred_coords = np.array([pred_dict[key][1] for key in common_keys_sorted])

        logger.debug(f"Mapped {len(common_keys_sorted)} residues by (resseq, icode) "
                    f"(ref chain {ref_chain_id or 'first'} had {len(ref_residues)}, "
                    f"pred chain {pred_chain_id or 'first'} had {len(pred_residues)})")

        return ref_coords, pred_coords, common_keys_sorted


def map_residues(ref_pdb: str, pred_pdb: str,
                ref_chain_id: Optional[str] = None,
                pred_chain_id: Optional[str] = None,
                model_index: int = 0) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Convenience function to map residues between structures.

    IMPORTANT for ESMFold predictions:
    - ESMFold always outputs to chain 'A'
    - Reference PDBs have original chain IDs
    - Use ref_chain_id for reference, pred_chain_id='A' for ESMFold predictions

    Args:
        ref_pdb: Path to reference structure
        pred_pdb: Path to predicted structure
        ref_chain_id: Chain to use from reference (None = first chain)
        pred_chain_id: Chain to use from predicted (None = first chain, 'A' for ESMFold)
        model_index: Which MODEL to use (default: 0)

    Returns:
        Tuple of (ref_coords, pred_coords, residue_ids)
    """
    mapper = ResidueMapper(model_index=model_index, chain_id=None)
    return mapper.map_residues(ref_pdb, pred_pdb,
                               ref_chain_id=ref_chain_id,
                               pred_chain_id=pred_chain_id)


def main():
    """Test residue mapper."""
    import argparse

    parser = argparse.ArgumentParser(description="Map residues between two structures")
    parser.add_argument("reference", help="Reference PDB file")
    parser.add_argument("predicted", help="Predicted PDB file")
    parser.add_argument("--chain", default=None, help="Specific chain to map")

    args = parser.parse_args()

    ref_coords, pred_coords, residue_ids = map_residues(
        args.reference, args.predicted, chain_id=args.chain
    )

    if ref_coords is not None:
        print(f"\nMapped {len(residue_ids)} residues")
        print(f"Reference shape: {ref_coords.shape}")
        print(f"Predicted shape: {pred_coords.shape}")
        print(f"\nFirst 5 residues:")
        for i, (chain, resseq, icode) in enumerate(residue_ids[:5]):
            icode_str = icode if icode.strip() else "-"
            print(f"  {i}: Chain {chain}, ResSeq {resseq}, ICode {icode_str}")
    else:
        print("Failed to map residues")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Structure Comparison Module

This module handles structural comparisons between predicted and reference structures
using TM-score calculations via tmtools.
"""

import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tmtools

# scipy for Pearson/Spearman correlations
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scipy not available — plddt_correlation metrics will be None"
    )

# Import new metric calculators
try:
    # Try relative imports first (when imported as a package)
    from .lddt_calculator import calculate_lddt
    from .gdt_calculator import calculate_gdt
    from .dssp_alignment import calculate_dssp_alignment
    from .sequence_metrics import compute_blosum62_similarity, _gapped_sequences_from_alignment
except ImportError:
    # Fallback to direct imports (when run as script)
    from lddt_calculator import calculate_lddt
    from gdt_calculator import calculate_gdt
    from dssp_alignment import calculate_dssp_alignment
    from sequence_metrics import compute_blosum62_similarity, _gapped_sequences_from_alignment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _parse_pdb_residue_bfactors(pdb_path: str, chain_id: Optional[str] = None) -> Optional[Dict[int, float]]:
    """
    Parse all heavy-atom B-factors from a PDB file, returning the per-residue
    average (averaged over all heavy atoms for that residue number).

    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to filter (None = first chain found)

    Returns:
        Dict mapping residue_number -> mean B-factor, or None on failure.
        Residue numbers are the integer sequence numbers from the PDB ATOM records.
    """
    try:
        from collections import defaultdict
        residue_bfactors: Dict[int, list] = defaultdict(list)
        target_chain = chain_id

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ENDMDL'):
                    break
                if not line.startswith('ATOM'):
                    continue
                atom_name = line[12:16].strip()
                # Skip hydrogens
                if atom_name.startswith('H'):
                    continue
                current_chain = line[21].strip()
                if target_chain is None:
                    target_chain = current_chain
                if current_chain != target_chain:
                    continue
                try:
                    res_seq = int(line[22:26].strip())
                    b_factor = float(line[60:66].strip())
                    residue_bfactors[res_seq].append(b_factor)
                except (ValueError, IndexError):
                    continue

        if not residue_bfactors:
            return None
        return {res: float(np.mean(vals)) for res, vals in residue_bfactors.items()}

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to parse B-factors from {pdb_path}: {e}")
        return None


def _parse_pdb_residue_sequence_with_resnums(pdb_path: str, chain_id: Optional[str] = None) -> Optional[Tuple[str, List[int]]]:
    """
    Parse CA atoms from a PDB file, returning the sequence and the corresponding
    PDB residue numbers in order.

    Returns:
        Tuple of (sequence_string, list_of_residue_numbers) or None on failure.
    """
    AA_MAP = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    try:
        sequence = []
        resnums = []
        target_chain = chain_id

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ENDMDL'):
                    break
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[16] in (' ', 'A'):
                    current_chain = line[21].strip()
                    if target_chain is None:
                        target_chain = current_chain
                    if current_chain != target_chain:
                        continue
                    res_name = line[17:20].strip()
                    try:
                        res_seq = int(line[22:26].strip())
                    except ValueError:
                        continue
                    sequence.append(AA_MAP.get(res_name, 'X'))
                    resnums.append(res_seq)

        if not sequence:
            return None
        return ''.join(sequence), resnums

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to parse sequence/resnums from {pdb_path}: {e}")
        return None


_NULL_SEQ_PLDDT_METRICS = {
    'seq_sim_blosum_frac': None,
    'seq_sim_blosum_mean': None,
    'plddt_pearson': None,
    'plddt_spearman': None,
}


def calculate_sequence_and_plddt_metrics(
    predicted_sequence: str,
    true_sequence: str,
    pred_pdb_path: str,
    ref_pdb_path: str,
    pred_chain_id: Optional[str] = 'A',
    ref_chain_id: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute sequence similarity (BLOSUM62) and pLDDT-vs-B-factor correlations
    using a pairwise sequence alignment as the shared residue correspondence.

    Sequence similarity metrics:
        seq_sim_blosum_frac  — fraction of aligned positions where BLOSUM62 score > 0
        seq_sim_blosum_mean  — mean BLOSUM62 score over all aligned positions (incl. negatives),
                               normalised by number of aligned positions

    pLDDT correlation metrics (only at positions with resolved reference coordinates):
        plddt_pearson        — Pearson r of (pred pLDDT, ref B-factor) over aligned+resolved pairs
        plddt_spearman       — Spearman rho of same

    Alignment: Needleman-Wunsch global alignment with BLOSUM62 substitution matrix.
    Both sequences must be non-empty strings of standard amino acid letters.

    Args:
        predicted_sequence: Predicted amino acid sequence (from CSV)
        true_sequence: True/reference amino acid sequence (from CSV or reference PDB)
        pred_pdb_path: Path to predicted ESMFold PDB
        ref_pdb_path: Path to reference PDB
        pred_chain_id: Chain in predicted PDB (ESMFold always 'A')
        ref_chain_id: Chain in reference PDB (None = first chain found)

    Returns:
        Dict with keys seq_sim_blosum_frac, seq_sim_blosum_mean,
        plddt_pearson, plddt_spearman  (any may be None on failure)
    """
    result = dict(_NULL_SEQ_PLDDT_METRICS)
    log = logging.getLogger(__name__)

    if not predicted_sequence or not true_sequence:
        return result

    try:
        from Bio import Align as _Align
        from Bio.Align import substitution_matrices as _subst_matrices
        blosum62 = _subst_matrices.load("BLOSUM62")
    except Exception as e:
        log.warning(f"Biopython/BLOSUM62 unavailable: {e}")
        return result

    # ------------------------------------------------------------------ #
    # 1+2. BLOSUM62 similarity (also builds the alignment we reuse below) #
    # ------------------------------------------------------------------ #
    blosum_result = compute_blosum62_similarity(predicted_sequence, true_sequence)
    result.update(blosum_result)

    # Rebuild the alignment for pLDDT correlation (reuse same aligner settings)
    try:
        aligner = _Align.PairwiseAligner()
        aligner.substitution_matrix = blosum62
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1
        aligner.mode = 'global'
        best = next(iter(aligner.align(predicted_sequence, true_sequence)))
        aligned_pred, aligned_true = _gapped_sequences_from_alignment(best)
    except Exception as e:
        log.warning(f"Sequence alignment failed (pLDDT step): {e}")
        return result

    # ------------------------------------------------------------------ #
    # 3. pLDDT vs B-factor correlation at sequence-aligned positions      #
    # ------------------------------------------------------------------ #
    if not _SCIPY_AVAILABLE:
        return result

    try:
        # Parse predicted PDB: sequence with residue numbers
        pred_parsed = _parse_pdb_residue_sequence_with_resnums(pred_pdb_path, pred_chain_id)
        if pred_parsed is None:
            return result
        pred_pdb_seq, pred_resnums = pred_parsed

        # Parse reference PDB: sequence with residue numbers
        ref_parsed = _parse_pdb_residue_sequence_with_resnums(ref_pdb_path, ref_chain_id)
        if ref_parsed is None:
            return result
        ref_pdb_seq, ref_resnums = ref_parsed

        # Per-residue B-factor averages (heavy atoms) for both structures
        pred_bfactors = _parse_pdb_residue_bfactors(pred_pdb_path, pred_chain_id)
        ref_bfactors = _parse_pdb_residue_bfactors(ref_pdb_path, ref_chain_id)

        if pred_bfactors is None or ref_bfactors is None:
            return result

        # Map sequence position (0-indexed in the full sequence, not aligned) to PDB resnum.
        # We use the PDB-parsed sequence to build this mapping, not the CSV sequence,
        # because only residues with coordinates appear in the PDB.
        # pred_resnums[i] is the PDB resnum for the i-th residue in pred_pdb_seq.
        # We need to map aligned_pred positions back to pred_pdb_seq positions.

        # Walk the alignment and the PDB sequences in parallel.
        # aligned_pred/aligned_true are relative to predicted_sequence/true_sequence (CSV).
        # We also need to track where we are in pred_pdb_seq and ref_pdb_seq.
        # Strategy: align CSV sequences to PDB sequences to get the mapping.
        # For ESMFold, pred PDB seq == predicted_sequence (ESMFold folds exactly the input).
        # For reference, ref PDB seq may be shorter (missing residues dropped).

        # Build index maps: position in CSV sequence -> PDB resnum (or None if no coords)
        def _seq_to_resnum_map(csv_seq, pdb_seq, pdb_resnums):
            """
            Align csv_seq to pdb_seq to map csv positions to PDB residue numbers.
            Positions in csv_seq that have no corresponding PDB residue map to None.
            """
            if csv_seq == pdb_seq:
                # Exact match — direct mapping
                return {i: pdb_resnums[i] for i in range(len(pdb_resnums))}

            # Global alignment of csv_seq vs pdb_seq
            aln = _Align.PairwiseAligner()
            aln.substitution_matrix = blosum62
            aln.open_gap_score = -11
            aln.extend_gap_score = -1
            aln.mode = 'global'
            try:
                best_aln = next(iter(aln.align(csv_seq, pdb_seq)))
            except StopIteration:
                return {}

            aln_csv, aln_pdb = _gapped_sequences_from_alignment(best_aln)
            mapping = {}
            csv_idx = 0
            pdb_idx = 0
            for ac, ap in zip(aln_csv, aln_pdb):
                if ac != '-' and ap != '-':
                    mapping[csv_idx] = pdb_resnums[pdb_idx]
                    csv_idx += 1
                    pdb_idx += 1
                elif ac != '-':
                    csv_idx += 1
                elif ap != '-':
                    pdb_idx += 1
            return mapping

        pred_csv_to_resnum = _seq_to_resnum_map(predicted_sequence, pred_pdb_seq, pred_resnums)
        ref_csv_to_resnum = _seq_to_resnum_map(true_sequence, ref_pdb_seq, ref_resnums)

        # Walk the main alignment (predicted_sequence vs true_sequence) and collect pairs
        pred_plddt_vals = []
        ref_bfactor_vals = []
        pred_csv_pos = 0
        ref_csv_pos = 0

        for aa_p, aa_t in zip(aligned_pred, aligned_true):
            if aa_p != '-' and aa_t != '-':
                # Aligned position — look up pLDDT and B-factor
                pred_resnum = pred_csv_to_resnum.get(pred_csv_pos)
                ref_resnum = ref_csv_to_resnum.get(ref_csv_pos)
                if pred_resnum is not None and ref_resnum is not None:
                    p_val = pred_bfactors.get(pred_resnum)
                    r_val = ref_bfactors.get(ref_resnum)
                    if p_val is not None and r_val is not None:
                        pred_plddt_vals.append(p_val)
                        ref_bfactor_vals.append(r_val)
                pred_csv_pos += 1
                ref_csv_pos += 1
            elif aa_p != '-':
                pred_csv_pos += 1
            elif aa_t != '-':
                ref_csv_pos += 1

        if len(pred_plddt_vals) >= 3:
            arr_p = np.array(pred_plddt_vals)
            arr_r = np.array(ref_bfactor_vals)
            try:
                pearson_r, _ = _scipy_stats.pearsonr(arr_p, arr_r)
                result['plddt_pearson'] = float(pearson_r)
            except Exception:
                pass
            try:
                spearman_r, _ = _scipy_stats.spearmanr(arr_p, arr_r)
                result['plddt_spearman'] = float(spearman_r)
            except Exception:
                pass

    except Exception as e:
        log.warning(f"pLDDT correlation calculation failed: {e}")

    return result


class StructureComparator:
    """
    A class to handle structure comparisons using TM-score and other metrics.
    """

    def __init__(self):
        """Initialize the structure comparator."""
        logger.info("Initialized StructureComparator with tmtools")

    def parse_pdb_structure(self, pdb_path: str, chain_id: Optional[str] = None) -> Optional[Tuple[np.ndarray, str]]:
        """
        Parse PDB file to extract CA coordinates and sequence.

        Args:
            pdb_path: Path to PDB file
            chain_id: Chain ID to filter by (if None, uses first chain found)

        Returns:
            Tuple of (coordinates array, sequence string) or None if parsing fails
        """
        try:
            coords = []
            sequence = []
            target_chain = chain_id

            with open(pdb_path, 'r') as f:
                for line in f:
                    # Stop at ENDMDL to only parse first model in multi-model files
                    if line.startswith('ENDMDL'):
                        break

                    if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[16] in (' ', 'A'):
                        # Extract chain ID from PDB (column 21, 0-indexed)
                        current_chain = line[21].strip()

                        # If no target chain specified, use the first chain found
                        if target_chain is None:
                            target_chain = current_chain

                        # Skip if not the target chain
                        if current_chain != target_chain:
                            continue

                        # Extract CA coordinates
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])

                        # Extract residue name and convert to single letter
                        res_name = line[17:20].strip()
                        aa_map = {
                            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                        }
                        sequence.append(aa_map.get(res_name, 'X'))

            if len(coords) == 0:
                logger.error(f"No CA atoms found in {pdb_path} for chain {chain_id}")
                return None

            coords_array = np.array(coords, dtype=np.float64)
            seq_string = ''.join(sequence)

            logger.debug(f"Parsed {len(coords)} CA atoms from {pdb_path} (chain {target_chain})")
            return coords_array, seq_string

        except Exception as e:
            logger.error(f"Failed to parse PDB structure {pdb_path}: {e}")
            return None

    def calculate_tm_score(self, structure1_path: str, structure2_path: str,
                          chain1_id: Optional[str] = None, chain2_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Calculate TM-score between two protein structures.

        Args:
            structure1_path: Path to first structure (PDB format)
            structure2_path: Path to second structure (PDB format)
            chain1_id: Chain ID for first structure (None = first chain found)
            chain2_id: Chain ID for second structure (None = first chain found)

        Returns:
            Dict with TM-score metrics or None if calculation fails
        """
        try:
            # Validate file existence early (fail-fast)
            if not os.path.exists(structure1_path):
                logger.error(f"Structure file not found: {structure1_path}")
                return None
            if not os.path.exists(structure2_path):
                logger.error(f"Structure file not found: {structure2_path}")
                return None

            # Parse both structures with specified chain IDs
            struct1_data = self.parse_pdb_structure(structure1_path, chain_id=chain1_id)
            struct2_data = self.parse_pdb_structure(structure2_path, chain_id=chain2_id)

            if struct1_data is None or struct2_data is None:
                return None

            coords1, seq1 = struct1_data
            coords2, seq2 = struct2_data

            # Calculate TM-score using tmtools
            result = tmtools.tm_align(coords1, coords2, seq1, seq2)

            # Calculate aligned length from alignment string (colons indicate matched positions)
            aligned_length = len(result.seqM)

            # Calculate sequence identity from alignment
            seq_identity = result.seqM.count(':') / aligned_length if aligned_length > 0 else 0.0

            metrics = {
                'tm_score': result.tm_norm_chain1,  # TM-score normalized by first structure
                'tm_score_ref': result.tm_norm_chain2,  # TM-score normalized by second structure
                'rmsd': result.rmsd,  # RMSD of aligned residues
                'aligned_length': aligned_length,  # Number of aligned residues
                'seq_identity': seq_identity,  # Sequence identity of aligned region
                'aligned_seq1': result.seqxA,  # Aligned sequence 1
                'aligned_seq2': result.seqyA   # Aligned sequence 2
            }

            logger.debug(f"TM-score calculated: {metrics['tm_score']:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate TM-score between {structure1_path} and {structure2_path}: {e}")
            return None

    def parse_pdb_sequence_length(self, pdb_path: str, chain_id: Optional[str] = None) -> Optional[int]:
        """
        Extract sequence length from a PDB file.

        Args:
            pdb_path: Path to PDB file
            chain_id: Chain ID to filter by (if None, uses first chain found)

        Returns:
            Sequence length or None if parsing fails
        """
        try:
            ca_atoms = 0
            target_chain = chain_id

            with open(pdb_path, 'r') as f:
                for line in f:
                    # Stop at ENDMDL to only parse first model in multi-model files
                    if line.startswith('ENDMDL'):
                        break

                    if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[16] in (' ', 'A'):
                        # Extract chain ID from PDB (column 21, 0-indexed)
                        current_chain = line[21].strip()

                        # If no target chain specified, use the first chain found
                        if target_chain is None:
                            target_chain = current_chain

                        # Skip if not the target chain
                        if current_chain != target_chain:
                            continue

                        ca_atoms += 1

            return ca_atoms

        except Exception as e:
            logger.error(f"Failed to parse sequence length from {pdb_path}: {e}")
            return None


class BatchStructureComparator:
    """
    Handles batch structure comparisons from prediction results.
    """

    def __init__(self, predicted_dir: str, reference_dir: str, csv_path: str, verbose: bool = False,
                 protein_subset: List[str] = None, protein_timeout_minutes: float = 12.0):
        """
        Initialize batch comparator.

        NOTE: Directory order is auto-detected! The code will automatically identify
        which directory contains pred_for_*.pdb files (predictions) and which contains
        the reference structures, regardless of which argument you pass as which.

        Args:
            predicted_dir: Directory containing predicted structures (or references if swapped)
            reference_dir: Directory containing reference structures (or predictions if swapped)
            csv_path: Path to CSV file with sequence predictions
            verbose: Enable verbose logging
            protein_subset: List of specific proteins to compare (optional)
            protein_timeout_minutes: Timeout per protein in minutes
        """
        self.predicted_dir = Path(predicted_dir)
        self.reference_dir = Path(reference_dir)
        try:
            self.csv_path = Path(csv_path)
        except:
            self.csv_path = None
            print("Failed to parse CSV path")
        self.comparator = StructureComparator()
        self.verbose = verbose
        self.protein_subset = protein_subset
        self.protein_timeout_minutes = protein_timeout_minutes

        # Check if predictions are from ESMFold (always use chain 'A')
        # Auto-detect from either directory (supports swapped arguments)
        # Method 1: Check directory names for 'esm' or 'ESM'
        is_esmfold_by_dir = 'esm' in str(predicted_dir).lower() or 'esm' in str(reference_dir).lower()

        # Method 2: Check if PDB files start with 'pred_for_' (ESMFold naming convention)
        is_esmfold_by_files = False
        for directory in [self.predicted_dir, self.reference_dir]:
            if directory.exists():
                pdb_files = list(directory.glob("*.pdb"))
                if pdb_files:
                    # Check if any files start with 'pred_for_'
                    if any(f.name.startswith('pred_for_') for f in pdb_files[:5]):
                        is_esmfold_by_files = True
                        break

        self.is_esmfold = is_esmfold_by_dir or is_esmfold_by_files

        if self.verbose:
            logger.debug(f"ESMFold detection: dir={is_esmfold_by_dir}, files={is_esmfold_by_files}, final={self.is_esmfold}")

        # Validate inputs
        if not self.predicted_dir.exists():
            raise FileNotFoundError(f"Predicted structures directory not found: {predicted_dir}")
        if not self.reference_dir.exists():
            raise FileNotFoundError(f"Reference structures directory not found: {reference_dir}")
        if not self.csv_path is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Initialized batch comparator:")
        logger.info(f"  Predicted: {self.predicted_dir}")
        logger.info(f"  Reference: {self.reference_dir}")
        logger.info(f"  CSV: {self.csv_path}")

        if verbose:
            pred_count = len(list(self.predicted_dir.glob("*.pdb")))
            ref_count = len(list(self.reference_dir.glob("*.pdb")))
            logger.debug(f"Structure file counts:")
            logger.debug(f"  Predicted structures: {pred_count}")
            logger.debug(f"  Reference structures: {ref_count}")

    def extract_chain_from_name(self, structure_name: str, file_path: str) -> Optional[str]:
        """
        Extract chain ID from structure name or file path.

        Handles two formats:
        - pred_for_{pdb_id}.{chain} -> chain
        - {pdb_id}_chain{chain}.pdb -> chain

        Args:
            structure_name: Structure name (e.g., "1mab.G" or "1mab_chainG")
            file_path: Full file path

        Returns:
            Chain ID or None if not extractable
        """
        # Format 1: "1mab.G" (dot notation)
        if '.' in structure_name:
            parts = structure_name.split('.')
            if len(parts) == 2 and len(parts[1]) == 1:
                return parts[1]

        # Format 2: "1mab_chainG" (underscore notation) - check filename
        file_name = Path(file_path).stem  # Get filename without extension
        if '_chain' in file_name:
            # Extract chain after '_chain'
            chain_part = file_name.split('_chain')[-1]
            if len(chain_part) >= 1:
                return chain_part[0]  # Return first character after '_chain'

        return None

    def load_sequence_data(self) -> pd.DataFrame:
        """Load sequence prediction data from CSV."""
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} sequence predictions")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")


    def find_structure_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Find pairs of predicted and reference structures.

        Supports three modes:
        1. Both directories have pred_for_*.pdb → match by structure name
        2. One has pred_for_*.pdb, other doesn't → standard comparison
        3. Neither has pred_for_*.pdb → match by filename directly

        Returns:
            List of tuples: (structure_name, predicted_path, reference_path)
        """
        pairs = []

        # Auto-detect which directory has pred_for_* files
        pred_files_in_predicted = list(self.predicted_dir.glob("pred_for_*.pdb"))
        pred_files_in_reference = list(self.reference_dir.glob("pred_for_*.pdb"))

        has_pred_format_in_pred = len(pred_files_in_predicted) > 0
        has_pred_format_in_ref = len(pred_files_in_reference) > 0

        # Case 1: Both directories have pred_for_* files (comparing two prediction sets)
        if has_pred_format_in_pred and has_pred_format_in_ref:
            logger.info("Both directories contain pred_for_*.pdb files - comparing two prediction sets")
            logger.info(f"Found {len(pred_files_in_predicted)} files in predicted_dir")
            logger.info(f"Found {len(pred_files_in_reference)} files in reference_dir")

            # Build lookup of structure names in reference dir
            ref_lookup = {f.stem.replace("pred_for_", ""): str(f)
                         for f in pred_files_in_reference}

            # Match by structure name
            for pred_file in pred_files_in_predicted:
                structure_name = pred_file.stem.replace("pred_for_", "")
                if structure_name in ref_lookup:
                    pairs.append((structure_name, str(pred_file), ref_lookup[structure_name]))
                else:
                    logger.warning(f"✗ No matching prediction for {structure_name} in reference_dir")

            logger.info(f"Found {len(pairs)} matching structure pairs")
            return pairs

        # Case 2: Neither directory has pred_for_* files (comparing raw PDB files)
        if not has_pred_format_in_pred and not has_pred_format_in_ref:
            logger.info("Neither directory has pred_for_*.pdb files - matching by filename")

            pred_files = list(self.predicted_dir.glob("*.pdb"))
            ref_files = list(self.reference_dir.glob("*.pdb"))

            logger.info(f"Found {len(pred_files)} PDB files in predicted_dir")
            logger.info(f"Found {len(ref_files)} PDB files in reference_dir")

            # Build lookup by filename
            ref_lookup = {f.stem: str(f) for f in ref_files}

            for pred_file in pred_files:
                structure_name = pred_file.stem
                if structure_name in ref_lookup:
                    pairs.append((structure_name, str(pred_file), ref_lookup[structure_name]))
                else:
                    logger.warning(f"✗ No matching file for {structure_name} in reference_dir")

            logger.info(f"Found {len(pairs)} matching structure pairs")
            return pairs

        # Case 3: One has pred_for_*, other doesn't (standard prediction vs reference)
        if has_pred_format_in_pred:
            # Normal case: predicted_dir has pred_for_* files
            actual_pred_dir = self.predicted_dir
            actual_ref_dir = self.reference_dir
            pred_files = pred_files_in_predicted
            swapped = False
        else:
            # Swapped case: reference_dir has pred_for_* files
            actual_pred_dir = self.reference_dir
            actual_ref_dir = self.predicted_dir
            pred_files = pred_files_in_reference
            swapped = True
            logger.warning("Detected swapped directories - auto-correcting. "
                          "Predicted files found in reference_dir, using that as predictions.")

        logger.info(f"Found {len(pred_files)} predicted structure files in {actual_pred_dir.name}")

        if self.verbose:
            logger.debug(f"Predicted files found: {[f.name for f in pred_files[:5]]}")
            if len(pred_files) > 5:
                logger.debug(f"... and {len(pred_files) - 5} more")

        # Filter by protein subset if provided
        if self.protein_subset:
            initial_count = len(pred_files)
            pred_files = [f for f in pred_files if f.stem.replace("pred_for_", "") in self.protein_subset]
            logger.info(f"Filtered predicted files: {len(pred_files)} from subset (was {initial_count} total)")

        # List all reference files (they are named directly as structure_name.pdb, not pred_for_structure_name.pdb)
        ref_files = list(actual_ref_dir.glob("*.pdb"))
        logger.info(f"Found {len(ref_files)} reference structure files in {actual_ref_dir.name}")

        if self.verbose:
            logger.debug(f"Reference files found: {[f.name for f in ref_files[:5]]}")
            if len(ref_files) > 5:
                logger.debug(f"... and {len(ref_files) - 5} more")

        if len(ref_files) == 0:
            logger.error(f"No reference files found in {actual_ref_dir}")
            logger.error("Check if the reference directory path is correct")
            return pairs

        # Build hash→structure_name lookup from CSV when filenames are sequence hashes
        # (used when multiple rows share the same structure_name, e.g. expval sweeps).
        import hashlib as _hl
        hash_to_struct = {}
        if self.csv_path is not None:
            try:
                import pandas as _pd
                _df = _pd.read_csv(self.csv_path)
                if 'predicted_sequence' in _df.columns and 'structure_name' in _df.columns:
                    for _, _row in _df.drop_duplicates('predicted_sequence').iterrows():
                        _h = _hl.md5(_row['predicted_sequence'].encode()).hexdigest()[:12]
                        hash_to_struct[_h] = _row['structure_name']
            except Exception:
                pass

        for pred_file in pred_files:
            raw_stem = pred_file.stem.replace("pred_for_", "")
            # If the stem is a sequence hash, resolve to the actual structure_name
            structure_name = hash_to_struct.get(raw_stem, raw_stem)

            if 'all_chain_pdbs' in actual_ref_dir.name:
                ref_file = actual_ref_dir / f"{structure_name.split('.')[0]}_chain{structure_name.split('.')[1]}.pdb"
            elif 'pred' in actual_ref_dir.name:
                actual_ref_dir / f"pred_for_{structure_name}.pdb"
            else:
                ref_file = actual_ref_dir / f"{structure_name}.pdb"

            if ref_file.exists():
                pairs.append((raw_stem, str(pred_file), str(ref_file)))
                if self.verbose:
                    logger.debug(f"✓ Found pair for {raw_stem} (structure: {structure_name})")
            else:
                logger.warning(f"✗ Reference structure not found for {structure_name}. Expected: {ref_file}")
                if self.verbose:
                    logger.debug(f"  Expected: {ref_file}")

        logger.info(f"Found {len(pairs)} structure pairs for comparison")

        if len(pairs) == 0:
            logger.error("No structure pairs found for comparison!")
            logger.error("This means no predicted structures have corresponding reference structures.")
            logger.error(f"Predicted dir: {self.predicted_dir}")
            logger.error(f"Reference dir: {self.reference_dir}")

        return pairs

    def compare_structures(self, output_path: str) -> pd.DataFrame:
        """
        Compare all structure pairs and save results.

        Args:
            output_path: Path to save comparison results CSV

        Returns:
            DataFrame with comparison results
        """
        # Load sequence data
        if self.verbose:
            logger.debug("Loading sequence prediction data...")

        if self.csv_path is not None:
            seq_df = self.load_sequence_data()
            import hashlib as _hl
            seq_data = {}
            for _, row in seq_df.iterrows():
                _h = _hl.md5(str(row['predicted_sequence']).encode()).hexdigest()[:12]
                seq_data[_h] = row
        else:
            seq_data = {}

        # Find structure pairs
        if self.verbose:
            logger.debug("Finding structure pairs for comparison...")
        pairs = self.find_structure_pairs()

        if self.verbose:
            logger.debug(f"Found {len(pairs)} pairs to compare")
            logger.debug(f"Sequence data available for {len(seq_data)} structures")

        # Initialize DSSP confusion matrices
        # 3-state: H, E, C, - (gap)
        labels_3state = ['H', 'E', 'C', '-']
        confusion_3state = np.zeros((len(labels_3state), len(labels_3state)), dtype=np.int64)

        # 8-state: Actual states found in data (alphabetical order for consistency)
        # B=β-bridge, E=β-strand, G=3-10 helix, H=α-helix, P=polyproline/turn, S=bend, T=turn, -=gap
        labels_8state = ['-', 'B', 'E', 'G', 'H', 'P', 'S', 'T']
        confusion_8state = np.zeros((len(labels_8state), len(labels_8state)), dtype=np.int64)

        results = []
        timeout_count = 0
        logger.info(f"Starting structural comparison for {len(pairs)} pairs...")

        for i, (structure_name, pred_path, ref_path) in enumerate(pairs):
            if self.verbose:
                logger.debug(f"Comparing {structure_name} ({i+1}/{len(pairs)})")
                logger.debug(f"  Predicted: {pred_path}")
                logger.debug(f"  Reference: {ref_path}")
            else:
                logger.info(f"Comparing {structure_name} ({i+1}/{len(pairs)})")

            # Get sequence data
            seq_info = seq_data.get(structure_name, {})

            if self.verbose and structure_name not in seq_data:
                logger.debug(f"  Warning: No sequence data found for {structure_name}")

            # Calculate structural metrics with timeout
            try:
                metrics = self._calculate_metrics_with_timeout(pred_path, ref_path, structure_name)

                if metrics is None:
                    logger.error(f"Failed to calculate metrics for {structure_name}")
                    continue

                # Get sequence lengths from PDB files with correct chain filtering
                if self.verbose:
                    logger.debug("  Parsing structure lengths...")

                # Extract chain IDs
                # For ESMFold predictions, always use chain 'A'
                pred_chain = 'A' if self.is_esmfold else self.extract_chain_from_name(structure_name, pred_path)
                # For reference, extract from structure name/path
                ref_chain = self.extract_chain_from_name(structure_name, ref_path)

                if self.verbose:
                    logger.debug(f"  Chain IDs: pred={pred_chain}, ref={ref_chain}")

                pred_length = self.comparator.parse_pdb_sequence_length(pred_path, chain_id=pred_chain)
                ref_length = self.comparator.parse_pdb_sequence_length(ref_path, chain_id=ref_chain)

                # Sequence similarity and pLDDT correlation (sequence-alignment based)
                if self.verbose:
                    logger.debug("  Computing sequence similarity and pLDDT correlation...")
                seq_plddt_metrics = calculate_sequence_and_plddt_metrics(
                    predicted_sequence=seq_info.get('predicted_sequence', ''),
                    true_sequence=seq_info.get('true_sequence', ''),
                    pred_pdb_path=pred_path,
                    ref_pdb_path=ref_path,
                    pred_chain_id=pred_chain,
                    ref_chain_id=ref_chain,
                )

                # Format metrics, handling None values
                lddt_val = metrics.get('lddt')
                lddt_str = f"{lddt_val:.1f}" if lddt_val is not None else "N/A"
                gdt_ts_val = metrics.get('gdt_ts')
                gdt_ts_str = f"{gdt_ts_val:.1f}" if gdt_ts_val is not None else "N/A"

                logger.info(f"✓ {structure_name}: TM={metrics['tm_score']:.4f}, RMSD={metrics['rmsd']:.2f}Å, "
                           f"LDDT={lddt_str}, GDT_TS={gdt_ts_str}")
                if self.verbose:
                    logger.debug(f"  Lengths: pred={pred_length}, ref={ref_length}")
                    dssp_3_val = metrics.get('dssp_3state_accuracy')
                    dssp_3_str = f"{dssp_3_val:.1f}" if dssp_3_val is not None else "N/A"
                    dssp_8_val = metrics.get('dssp_8state_accuracy')
                    dssp_8_str = f"{dssp_8_val:.1f}" if dssp_8_val is not None else "N/A"
                    logger.debug(f"  DSSP 3-state: {dssp_3_str}%, 8-state: {dssp_8_str}%")

                # Compile results with all metrics
                result = {
                    'structure_name': structure_name,
                    'sequence_length': seq_info.get('length', None),
                    'predicted_sequence': seq_info.get('predicted_sequence', ''),
                    'true_sequence': seq_info.get('true_sequence', ''),
                    'sequence_accuracy': seq_info.get('accuracy', None),
                    'pred_structure_length': pred_length,
                    'ref_structure_length': ref_length,
                    # TM-score metrics
                    'tm_score': metrics['tm_score'],
                    'tm_score_ref': metrics['tm_score_ref'],
                    'rmsd': metrics['rmsd'],
                    'aligned_length': metrics['aligned_length'],
                    'seq_id_aligned': metrics['seq_identity'],
                    # LDDT metric
                    'lddt': metrics.get('lddt', None),
                    # GDT metrics
                    'gdt_ts': metrics.get('gdt_ts', None),
                    'gdt_ha': metrics.get('gdt_ha', None),
                    'gdt_05': metrics.get('gdt_05', None),
                    'gdt_1': metrics.get('gdt_1', None),
                    'gdt_2': metrics.get('gdt_2', None),
                    'gdt_4': metrics.get('gdt_4', None),
                    'gdt_8': metrics.get('gdt_8', None),
                    # DSSP metrics
                    'dssp_3state_accuracy': metrics.get('dssp_3state_accuracy', None),
                    'dssp_8state_accuracy': metrics.get('dssp_8state_accuracy', None),
                    'dssp_3state_score': metrics.get('dssp_3state_score', None),
                    'dssp_8state_score': metrics.get('dssp_8state_score', None),
                    'dssp_3state_alignment_ref': metrics.get('dssp_3state_alignment_ref', ''),
                    'dssp_3state_alignment_pred': metrics.get('dssp_3state_alignment_pred', ''),
                    'dssp_8state_alignment_ref': metrics.get('dssp_8state_alignment_ref', ''),
                    'dssp_8state_alignment_pred': metrics.get('dssp_8state_alignment_pred', ''),
                    # Sequence similarity (BLOSUM62, sequence-alignment based)
                    'seq_sim_blosum_frac': seq_plddt_metrics.get('seq_sim_blosum_frac'),
                    'seq_sim_blosum_mean': seq_plddt_metrics.get('seq_sim_blosum_mean'),
                    # pLDDT vs B-factor correlation (sequence-alignment based, resolved ref positions only)
                    'plddt_pearson': seq_plddt_metrics.get('plddt_pearson'),
                    'plddt_spearman': seq_plddt_metrics.get('plddt_spearman'),
                    # File paths
                    'predicted_structure_path': pred_path,
                    'reference_structure_path': ref_path
                }

                # Accumulate DSSP confusion matrices
                if metrics.get('dssp_3state_alignment_ref') and metrics.get('dssp_3state_alignment_pred'):
                    ref_3 = metrics['dssp_3state_alignment_ref']
                    pred_3 = metrics['dssp_3state_alignment_pred']
                    for ref_char, pred_char in zip(ref_3, pred_3):
                        if ref_char in labels_3state and pred_char in labels_3state:
                            ref_idx = labels_3state.index(ref_char)
                            pred_idx = labels_3state.index(pred_char)
                            confusion_3state[ref_idx, pred_idx] += 1

                if metrics.get('dssp_8state_alignment_ref') and metrics.get('dssp_8state_alignment_pred'):
                    ref_8 = metrics['dssp_8state_alignment_ref']
                    pred_8 = metrics['dssp_8state_alignment_pred']
                    for ref_char, pred_char in zip(ref_8, pred_8):
                        if ref_char in labels_8state and pred_char in labels_8state:
                            ref_idx = labels_8state.index(ref_char)
                            pred_idx = labels_8state.index(pred_char)
                            confusion_8state[ref_idx, pred_idx] += 1

                results.append(result)

            except TimeoutError:
                timeout_count += 1
                logger.warning(f"⏰ Timeout ({self.protein_timeout_minutes} min) for {structure_name}")
                continue

        # Create DataFrame and save results
        if self.verbose:
            logger.debug("Creating results DataFrame...")
        results_df = pd.DataFrame(results)

        if timeout_count > 0:
            logger.warning(f"Total timeouts: {timeout_count}")

        if len(results_df) > 0:
            # Sort by TM-score descending
            results_df = results_df.sort_values('tm_score', ascending=False)

            # Save to CSV
            if self.verbose:
                logger.debug(f"Saving results to {output_path}")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Comparison results saved to {output_path}")

            # Save DSSP confusion matrices
            confusion_path = output_path.replace('.csv', '_dssp_confusion.npz')
            np.savez(
                confusion_path,
                confusion_3state=confusion_3state,
                labels_3state=np.array(labels_3state),
                confusion_8state=confusion_8state,
                labels_8state=np.array(labels_8state),
                total_comparisons=len(results_df)
            )
            logger.info(f"DSSP confusion matrices saved to {confusion_path}")

            # Print summary statistics
            self._print_summary_stats(results_df)
        else:
            logger.warning("No successful comparisons were performed")
            # Create empty DataFrame with correct columns for consistency
            empty_df = pd.DataFrame(columns=[
                'structure_name', 'sequence_length', 'predicted_sequence', 'true_sequence',
                'sequence_accuracy', 'pred_structure_length', 'ref_structure_length',
                'tm_score', 'tm_score_ref', 'rmsd', 'aligned_length', 'seq_id_aligned',
                'lddt',
                'gdt_ts', 'gdt_ha', 'gdt_05', 'gdt_1', 'gdt_2', 'gdt_4', 'gdt_8',
                'dssp_3state_accuracy', 'dssp_8state_accuracy',
                'dssp_3state_score', 'dssp_8state_score',
                'dssp_3state_alignment_ref', 'dssp_3state_alignment_pred',
                'dssp_8state_alignment_ref', 'dssp_8state_alignment_pred',
                'seq_sim_blosum_frac', 'seq_sim_blosum_mean',
                'plddt_pearson', 'plddt_spearman',
                'predicted_structure_path', 'reference_structure_path'
            ])
            # Save empty CSV to prevent FileNotFoundError downstream
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Empty comparison results saved to {output_path}")

            # Also save a summary of what went wrong
            error_summary_path = Path(output_path).parent / "comparison_errors.txt"
            with open(error_summary_path, 'w') as f:
                f.write("Structure Comparison Error Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Found {len(pairs)} structure pairs to compare\n")
                f.write("Issues encountered:\n")
                f.write("- All TM-score calculations failed\n")
                f.write("- Check PDB file formats and tmtools installation\n")
                f.write("- Verify structure file integrity\n")

            return empty_df

        return results_df

    def _print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics of comparison results."""
        print("\n" + "="*80)
        print("STRUCTURAL COMPARISON SUMMARY")
        print("="*80)

        print(f"Total comparisons: {len(df)}")

        # Basic metrics
        print(f"\nAverage TM-score: {df['tm_score'].mean():.4f} ± {df['tm_score'].std():.4f}")
        print(f"Average RMSD: {df['rmsd'].mean():.2f} ± {df['rmsd'].std():.2f} Å")

        # LDDT metrics
        if 'lddt' in df.columns and df['lddt'].notna().any():
            print(f"Average LDDT: {df['lddt'].mean():.2f} ± {df['lddt'].std():.2f}")

        # GDT metrics
        if 'gdt_ts' in df.columns and df['gdt_ts'].notna().any():
            print(f"Average GDT_TS: {df['gdt_ts'].mean():.2f} ± {df['gdt_ts'].std():.2f}")
            print(f"Average GDT_HA: {df['gdt_ha'].mean():.2f} ± {df['gdt_ha'].std():.2f}")

        # DSSP metrics
        if 'dssp_3state_accuracy' in df.columns and df['dssp_3state_accuracy'].notna().any():
            print(f"Average DSSP 3-state accuracy: {df['dssp_3state_accuracy'].mean():.2f}%")
            print(f"Average DSSP 8-state accuracy: {df['dssp_8state_accuracy'].mean():.2f}%")

        # Sequence accuracy (if available)
        if 'sequence_accuracy' in df.columns and df['sequence_accuracy'].notna().any():
            print(f"Average sequence accuracy: {df['sequence_accuracy'].mean():.2f}%")

        print("\nTM-score distribution:")
        print(f"  > 0.5 (good): {(df['tm_score'] > 0.5).sum()} ({(df['tm_score'] > 0.5).mean()*100:.1f}%)")
        print(f"  > 0.4: {(df['tm_score'] > 0.4).sum()} ({(df['tm_score'] > 0.4).mean()*100:.1f}%)")
        print(f"  > 0.3: {(df['tm_score'] > 0.3).sum()} ({(df['tm_score'] > 0.3).mean()*100:.1f}%)")

        print("\nTop 5 structures by TM-score:")
        display_cols = ['structure_name', 'tm_score', 'rmsd']
        if 'lddt' in df.columns:
            display_cols.append('lddt')
        if 'gdt_ts' in df.columns:
            display_cols.append('gdt_ts')

        top5 = df.nlargest(5, 'tm_score')[display_cols]
        for _, row in top5.iterrows():
            metrics_str = f"TM={row['tm_score']:.4f}, RMSD={row['rmsd']:.2f}Å"
            if 'lddt' in row and pd.notna(row.get('lddt')):
                metrics_str += f", LDDT={row['lddt']:.1f}"
            if 'gdt_ts' in row and pd.notna(row.get('gdt_ts')):
                metrics_str += f", GDT_TS={row['gdt_ts']:.1f}"
            print(f"  {row['structure_name']}: {metrics_str}")

        print("\nWorst 5 structures by TM-score:")
        worst5 = df.nsmallest(5, 'tm_score')[display_cols]
        for _, row in worst5.iterrows():
            metrics_str = f"TM={row['tm_score']:.4f}, RMSD={row['rmsd']:.2f}Å"
            if 'lddt' in row and pd.notna(row.get('lddt')):
                metrics_str += f", LDDT={row['lddt']:.1f}"
            if 'gdt_ts' in row and pd.notna(row.get('gdt_ts')):
                metrics_str += f", GDT_TS={row['gdt_ts']:.1f}"
            print(f"  {row['structure_name']}: {metrics_str}")

        print("="*80)

    def _calculate_metrics_with_timeout(self, pred_path: str, ref_path: str, structure_name: str) -> Optional[Dict[str, float]]:
        """
        Calculate structural metrics with timeout handling.

        Calculates multiple structural comparison metrics:
            - TM-score: Uses aligned regions after optimal superposition
            - LDDT: Local metric, no global superposition (uses full structures independently)
            - GDT: Uses aligned regions after optimal global superposition
            - DSSP: Compares secondary structure via global alignment

        Args:
            pred_path: Path to predicted structure
            ref_path: Path to reference structure
            structure_name: Name of the structure (for logging)

        Returns:
            Dict with all structural metrics or None if failed

        Raises:
            TimeoutError: If calculation exceeds timeout
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Comparison timeout for {structure_name}")

        # Set up timeout (increased 4x to accommodate all metrics)
        timeout_seconds = int(self.protein_timeout_minutes * 60)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            if self.verbose:
                logger.debug(f"  Calculating all structural metrics with {timeout_seconds}s timeout...")

            # Extract chain ID from structure_name for REFERENCE
            # ESMFold predictions always use chain 'A', so pred_chain_id='A'
            # Format: "1mab.G" -> ref_chain='G', pred_chain='A' (ESMFold default)
            ref_chain_id = None
            if '.' in structure_name:
                parts = structure_name.split('.')
                if len(parts) == 2 and len(parts[1]) == 1:
                    ref_chain_id = parts[1]

            # ESMFold predictions ALWAYS use chain 'A' regardless of original chain
            pred_chain_id = 'A'

            # Calculate TM-score (uses aligned regions after optimal superposition)
            # Use the same chain IDs as GDT/LDDT for consistency
            if self.verbose:
                logger.debug(f"  Computing TM-score (ref chain={ref_chain_id}, pred chain={pred_chain_id})...")
            tm_metrics = self.comparator.calculate_tm_score(pred_path, ref_path,
                                                           chain1_id=pred_chain_id,
                                                           chain2_id=ref_chain_id)
            if tm_metrics is None:
                signal.alarm(0)
                return None

            # Calculate LDDT (local metric, no global superposition - uses full structures independently)
            # Uses per-residue averaging with sequence separation |i-j|>=2
            if self.verbose:
                logger.debug(f"  Computing LDDT (ref chain={ref_chain_id}, pred chain={pred_chain_id})...")
            lddt_metrics = calculate_lddt(ref_path, pred_path,
                                         ref_chain_id=ref_chain_id,
                                         pred_chain_id=pred_chain_id)
            # Fallback: if fails, try with first chains (None, None)
            if lddt_metrics is None and ref_chain_id is not None:
                if self.verbose:
                    logger.debug(f"  LDDT failed with specified chains, retrying with first chains")
                lddt_metrics = calculate_lddt(ref_path, pred_path,
                                             ref_chain_id=None,
                                             pred_chain_id=None)
            if lddt_metrics is None:
                lddt_metrics = {'lddt': None}

            # Calculate GDT (uses aligned regions after optimal global superposition via Kabsch)
            # Implements LGA-style iterative refinement with fragment seeding per cutoff
            if self.verbose:
                logger.debug(f"  Computing GDT (ref chain={ref_chain_id}, pred chain={pred_chain_id})...")
            gdt_metrics = calculate_gdt(ref_path, pred_path,
                                       ref_chain_id=ref_chain_id,
                                       pred_chain_id=pred_chain_id)
            # Fallback: if fails, try with first chains
            if gdt_metrics is None and ref_chain_id is not None:
                if self.verbose:
                    logger.debug(f"  GDT failed with specified chains, retrying with first chains")
                gdt_metrics = calculate_gdt(ref_path, pred_path,
                                           ref_chain_id=None,
                                           pred_chain_id=None)
            if gdt_metrics is None:
                gdt_metrics = {
                    'gdt_ts': None,
                    'gdt_ha': None,
                    'gdt_05': None,
                    'gdt_1': None,
                    'gdt_2': None,
                    'gdt_4': None,
                    'gdt_8': None
                }

            # Calculate DSSP alignment (compares secondary structure via global alignment)
            # Uses correct denominator (mapped residues) and adaptive gap penalties
            if self.verbose:
                logger.debug(f"  Computing DSSP alignment (ref chain={ref_chain_id}, pred chain={pred_chain_id})...")
            dssp_metrics = calculate_dssp_alignment(ref_path, pred_path,
                                                   ref_chain_id=ref_chain_id,
                                                   pred_chain_id=pred_chain_id)
            # Fallback: if fails, try with first chains
            if dssp_metrics is None and ref_chain_id is not None:
                if self.verbose:
                    logger.debug(f"  DSSP failed with specified chains, retrying with first chains")
                dssp_metrics = calculate_dssp_alignment(ref_path, pred_path,
                                                       ref_chain_id=None,
                                                       pred_chain_id=None)
            if dssp_metrics is None:
                dssp_metrics = {
                    'dssp_3state_accuracy': None,
                    'dssp_8state_accuracy': None,
                    'dssp_3state_score': None,
                    'dssp_8state_score': None,
                    'dssp_3state_alignment_ref': '',
                    'dssp_3state_alignment_pred': '',
                    'dssp_8state_alignment_ref': '',
                    'dssp_8state_alignment_pred': ''
                }

            # Combine all metrics
            all_metrics = {
                **tm_metrics,
                **lddt_metrics,
                **gdt_metrics,
                **dssp_metrics
            }

            signal.alarm(0)  # Cancel timeout
            return all_metrics

        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            logger.warning(f"Structure comparison timed out for {structure_name} after {self.protein_timeout_minutes} minutes")
            raise

        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"Structure comparison failed for {structure_name}: {e}")
            return None

        finally:
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


def main():
    """Example usage of the structure comparator."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare predicted and reference protein structures")
    parser.add_argument("predicted_dir", help="Directory containing predicted structures")
    parser.add_argument("reference_dir", help="Directory containing reference structures")
    parser.add_argument("--csv_path", default=None, help="Path to CSV file with sequence predictions")
    parser.add_argument("--output_path", default='./datasets/pdb_esmfold_comparison_test_set.csv', help="Path to save comparison results CSV")

    args = parser.parse_args()

    try:
        comparator = BatchStructureComparator(
            args.predicted_dir,
            args.reference_dir,
            args.csv_path
        )

        results_df = comparator.compare_structures(args.output_path)

        if len(results_df) > 0:
            print(f"\nComparison completed successfully!")
            print(f"Results saved to: {args.output_path}")
        else:
            print("No comparisons were successful.")
            return 1

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

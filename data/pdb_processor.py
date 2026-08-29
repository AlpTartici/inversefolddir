# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
pdb_processor.py

PDB file processing utilities for extracting B-factors, sequences, and coordinates from PDB files.
Handles PDB parsing, B-factor extraction, sequence extraction, coordinate extraction, and normalization.
"""
import os
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class PDBProcessor:
    """
    Processor for PDB files to extract B-factors, sequences, and coordinates.

    Handles:
    - PDB file parsing
    - B-factor extraction per residue
    - Sequence extraction from PDB files
    - Coordinate extraction for backbone atoms (N, CA, C, O)
    - B-factor normalization to [0, 1] range
    - Error handling for missing or invalid data
    """

    def __init__(self,
                 pdb_directory: str,
                 b_factor_percentiles: Tuple[float, float] = (5.0, 95.0),
                 default_b_factor: float = 50.0,
                 verbose: bool = False):
        """
        Initialize PDB processor.

        Args:
            pdb_directory: Directory containing PDB files
            b_factor_percentiles: Percentiles for B-factor normalization (min, max)
            default_b_factor: Default B-factor for missing values
            verbose: Whether to print detailed processing information
        """
        self.pdb_directory = Path(pdb_directory)
        self.b_factor_percentiles = b_factor_percentiles
        self.default_b_factor = default_b_factor
        self.verbose = verbose

        # Amino acid three-to-one letter code mapping
        self.aa_3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        # No longer computing global statistics since we normalize within each protein

    # Global statistics computation removed - now using within-protein normalization
    # def _compute_global_b_factor_stats(self):
    #     """Compute global B-factor statistics for normalization."""
    #     # This method is no longer used since we normalize within each protein

    def _extract_b_factors_from_file(self, pdb_file: Path) -> Optional[List[float]]:
        """
        Extract B-factors from a PDB file.

        Args:
            pdb_file: Path to PDB file

        Returns:
            List of B-factors per residue or None if extraction fails
        """
        try:
            with open(pdb_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            if self.verbose:
                print(f"Error reading {pdb_file}: {e}")
            return None

        # Extract B-factors from ATOM records
        residue_b_factors = {}  # residue_id -> list of b_factors

        for line in lines:
            if line.startswith('ATOM'):
                try:
                    # Parse PDB ATOM record
                    atom_name = line[12:16].strip()
                    res_seq = int(line[22:26].strip())
                    chain_id = line[21]
                    b_factor = float(line[60:66].strip())

                    # Create unique residue identifier
                    res_id = f"{chain_id}_{res_seq}"

                    # Only consider backbone atoms for representative B-factor
                    if atom_name in ['CA', 'C', 'N', 'O']:
                        if res_id not in residue_b_factors:
                            residue_b_factors[res_id] = []
                        residue_b_factors[res_id].append(b_factor)

                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"Error parsing line in {pdb_file}: {line.strip()}")
                    continue
        #print(residue_b_factors)
        # Average B-factors per residue
        if residue_b_factors:
            avg_b_factors = []
            for res_id in residue_b_factors.keys():
                avg_b_factor = np.mean(residue_b_factors[res_id])
                avg_b_factors.append(float(avg_b_factor))  # Convert numpy scalar to Python float
            #print(avg_b_factors)
            return avg_b_factors

        return None

    def _extract_sequence_and_coords_from_file(self, pdb_file: Path) -> Tuple[Optional[str], Optional[Dict[str, np.ndarray]]]:
        """
        Extract sequence and coordinates from a PDB file.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (sequence_string, coordinates_dict) or (None, None) if extraction fails
            coordinates_dict has keys ['N', 'CA', 'C', 'O'] with values as Lx3 numpy arrays
        """
        try:
            with open(pdb_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            if self.verbose:
                print(f"Error reading {pdb_file}: {e}")
            return None, None

        # Extract coordinates and sequence from ATOM records
        residue_data = {}  # residue_id -> {'coords': {atom: [x,y,z]}, 'aa': residue_name}

        for line in lines:
            if line.startswith('ATOM'):
                try:
                    # Parse PDB ATOM record
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21]
                    res_seq = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    # Create unique residue identifier
                    res_id = f"{chain_id}_{res_seq}"

                    # Only consider backbone atoms
                    if atom_name in ['N', 'CA', 'C', 'O']:
                        if res_id not in residue_data:
                            residue_data[res_id] = {'coords': {}, 'aa': res_name}
                        residue_data[res_id]['coords'][atom_name] = [x, y, z]

                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"Error parsing line in {pdb_file}: {line.strip()}")
                    continue

        if not residue_data:
            return None, None

        # Sort residues by their sequence number for consistent ordering
        sorted_residues = sorted(residue_data.items(),
                               key=lambda x: int(x[0].split('_')[1]))

        # Extract sequence
        sequence = ""
        coords_dict = {'N': [], 'CA': [], 'C': [], 'O': []}

        for res_id, data in sorted_residues:
            # Add amino acid to sequence
            aa_3 = data['aa']
            aa_1 = self.aa_3to1.get(aa_3, 'X')  # 'X' for unknown amino acids
            sequence += aa_1

            # Add coordinates (use NaN if atom is missing)
            for atom in ['N', 'CA', 'C', 'O']:
                if atom in data['coords']:
                    coords_dict[atom].append(data['coords'][atom])
                else:
                    # Missing atom - use NaN coordinates
                    coords_dict[atom].append([np.nan, np.nan, np.nan])
                    if self.verbose:
                        print(f"Missing {atom} atom for residue {res_id} in {pdb_file}")

        # Convert to numpy arrays
        for atom in coords_dict:
            coords_dict[atom] = np.array(coords_dict[atom])

        return sequence, coords_dict

    def get_sequence_for_structure(self, structure_id: str) -> Optional[str]:
        """
        Get sequence for a specific structure.

        Args:
            structure_id: Structure identifier (e.g., "3nks_chainA")

        Returns:
            Sequence string or None if extraction fails
        """
        pdb_file = self.pdb_directory / f"{structure_id}.pdb"

        if not pdb_file.exists():
            if self.verbose:
                print(f"PDB file not found: {pdb_file}")
            return None

        try:
            sequence, _ = self._extract_sequence_and_coords_from_file(pdb_file)
            return sequence
        except Exception as e:
            if self.verbose:
                print(f"Error extracting sequence from {structure_id}: {e}")
                traceback.print_exc()
            return None

    def get_coordinates_for_structure(self, structure_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get coordinates for a specific structure.

        Args:
            structure_id: Structure identifier (e.g., "3nks_chainA")

        Returns:
            Dictionary with keys ['N', 'CA', 'C', 'O'] and values as Lx3 numpy arrays,
            or None if extraction fails
        """
        pdb_file = self.pdb_directory / f"{structure_id}.pdb"

        if not pdb_file.exists():
            if self.verbose:
                print(f"PDB file not found: {pdb_file}")
            return None

        try:
            _, coords = self._extract_sequence_and_coords_from_file(pdb_file)
            return coords
        except Exception as e:
            if self.verbose:
                print(f"Error extracting coordinates from {structure_id}: {e}")
                traceback.print_exc()
            return None

    def get_structure_data(self, structure_id: str) -> Dict[str, Union[str, Dict[str, np.ndarray], np.ndarray, int]]:
        """
        Get complete structure data including sequence, coordinates, and B-factors.

        Args:
            structure_id: Structure identifier (e.g., "3nks_chainA")

        Returns:
            Dictionary with keys:
            - 'sequence': string sequence
            - 'coords': dict with keys ['N', 'CA', 'C', 'O'] and Lx3 numpy arrays
            - 'b_factors': raw B-factors array
            - 'b_factors_normalized': normalized B-factors in [0, 1] range
            - 'length': sequence length
            Returns empty dict if extraction fails
        """
        pdb_file = self.pdb_directory / f"{structure_id}.pdb"

        if not pdb_file.exists():
            if self.verbose:
                print(f"PDB file not found: {pdb_file}")
            return {}

        try:
            # Extract sequence and coordinates
            sequence, coords = self._extract_sequence_and_coords_from_file(pdb_file)

            if sequence is None or coords is None:
                if self.verbose:
                    print(f"Failed to extract sequence/coords from {structure_id}")
                return {}

            seq_length = len(sequence)

            # Extract B-factors
            b_factors = self.get_b_factors_for_structure(structure_id, seq_length)

            return {
                'seq': sequence,
                'coords': coords,
                'b_factors': b_factors,
                'length': seq_length
            }

        except Exception as e:
            if self.verbose:
                print(f"Error processing complete structure data for {structure_id}: {e}")
                traceback.print_exc()
            return {}

    def get_b_factors_for_structure(self, structure_id: str, sequence_length: int) -> np.ndarray:
        """
        Get B-factors for a specific structure.

        Args:
            structure_id: Structure identifier (e.g., "3nks_chainA")
            sequence_length: Expected sequence length

        Returns:
            Numpy array of B-factors, padded/truncated to sequence_length
        """
        pdb_file = self.pdb_directory / f"{structure_id}.pdb"

        if not pdb_file.exists():
            if self.verbose:
                print(f"PDB file not found: {pdb_file}")
            # Return default B-factors
            return np.full(sequence_length, self.default_b_factor)

        try:
            b_factors = self._extract_b_factors_from_file(pdb_file)

            if b_factors is None or len(b_factors) == 0:
                if self.verbose:
                    print(f"No B-factors extracted from {pdb_file}")
                return np.full(sequence_length, self.default_b_factor)

            b_factors = np.array(b_factors)

            # Handle length mismatch
            if len(b_factors) != sequence_length:
                if self.verbose:
                    print(f"B-factor length mismatch for {structure_id}: "
                          f"got {len(b_factors)}, expected {sequence_length}")

                # Pad or truncate
                if len(b_factors) < sequence_length:
                    # Pad with last value or default
                    pad_value = b_factors[-1] if len(b_factors) > 0 else self.default_b_factor
                    b_factors = np.pad(b_factors, (0, sequence_length - len(b_factors)),
                                     constant_values=pad_value)
                else:
                    # Truncate
                    b_factors = b_factors[:sequence_length]

            return b_factors

        except Exception as e:
            if self.verbose:
                print(f"Error processing {structure_id}: {e}")
                traceback.print_exc()
            return np.full(sequence_length, self.default_b_factor)

    def normalize_b_factors(self, b_factors: np.ndarray) -> np.ndarray:
        """
        Normalize B-factors within each protein using mean-center unit-norm + sigmoid.
        NMR structures (all-zero B-factors) are set to neutral uncertainty (0.5).

        Args:
            b_factors: Array of B-factors for a single protein

        Returns:
            Normalized B-factors in [0, 1] range
        """
        # Check for all-zero B-factors (NMR structures)
        if np.all(b_factors == 0):
            # NMR structure: return neutral uncertainty for all positions
            return np.full_like(b_factors, 0.5, dtype=np.float32)

        # Check for constant B-factors (all values the same)
        if np.std(b_factors) == 0:
            # All values identical: return neutral uncertainty
            return np.full_like(b_factors, 0.5, dtype=np.float32)

        # Within-protein normalization using mean-center unit-norm + sigmoid
        mean = np.mean(b_factors)
        std = np.std(b_factors)

        # Standardize (mean=0, std=1)
        normalized = (b_factors - mean) / std

        # Apply sigmoid to get [0, 1] range
        sigmoid = 1 / (1 + np.exp(-normalized))

        # For B-factors: higher B-factor = more uncertainty = lower confidence
        # So we invert the sigmoid
        result = 1 - sigmoid

        return result.astype(np.float32)

    def process_structure(self, structure_id: str, sequence_length: int = None,
                         return_all: bool = False) -> Union[np.ndarray, Dict[str, Union[str, Dict[str, np.ndarray], np.ndarray, int]]]:
        """
        Complete processing pipeline for a structure.

        Args:
            structure_id: Structure identifier
            sequence_length: Expected sequence length (optional, will be inferred if not provided)
            return_all: If True, return dict with sequence, coords, and B-factors.
                       If False, return only normalized B-factors (backward compatibility)

        Returns:
            If return_all=False: Normalized B-factors in [0, 1] range
            If return_all=True: Dict with sequence, coords, b_factors, etc.
        """
        if return_all:
            return self.get_structure_data(structure_id)

        # Backward compatibility: just return B-factors
        if sequence_length is None:
            # Try to infer length from sequence
            sequence = self.get_sequence_for_structure(structure_id)
            if sequence is not None:
                sequence_length = len(sequence)
            else:
                raise ValueError(f"Could not infer sequence length for {structure_id} and none provided")

        # Extract B-factors
        b_factors = self.get_b_factors_for_structure(structure_id, sequence_length)

        # Normalize
        normalized = self.normalize_b_factors(b_factors)

        return normalized

    def get_statistics(self) -> Dict:
        """Get processor statistics."""
        return {
            'pdb_directory': str(self.pdb_directory),
            'default_b_factor': self.default_b_factor,
            'percentiles': self.b_factor_percentiles,
            'normalization_method': 'within_protein_mcun_sigmoid',
            'capabilities': ['b_factors', 'sequence_extraction', 'coordinate_extraction'],
            'coordinate_atoms': ['N', 'CA', 'C', 'O'],
            'amino_acid_mapping': 'three_to_one_letter'
        }


def extract_structure_id_from_cath(cath_entry: Dict) -> str:
    """
    Extract structure ID from CATH dataset entry.

    Args:
        cath_entry: CATH dataset entry dictionary

    Returns:
        Structure ID in format expected by PDB files
    """
    # CATH entries typically have 'name' field like "3nksA00"
    # Convert to PDB format like "3nks_chainA"
    name = cath_entry.get('name', '')

    if len(name) >= 5:
        pdb_id = name[:4].lower()
        chain_id = name[4].upper()
        return f"{pdb_id}_chain{chain_id}"

    return name


def integrate_b_factors_into_graph(graph_data: torch.Tensor,
                                 b_factors: np.ndarray,
                                 uncertainty_channel: int = -3) -> torch.Tensor:
    """
    Integrate B-factors into graph node features.

    Args:
        graph_data: Graph node features tensor [N, F]
        b_factors: Normalized B-factors [N]
        uncertainty_channel: Channel index for uncertainty values

    Returns:
        Updated graph data with B-factors
    """
    # Ensure b_factors is torch tensor
    if isinstance(b_factors, np.ndarray):
        b_factors = torch.from_numpy(b_factors).float()

    # Ensure correct length
    seq_len = graph_data.shape[0]
    if len(b_factors) != seq_len:
        print(f"Warning: B-factor length mismatch: {len(b_factors)} vs {seq_len}")
        # Pad or truncate
        if len(b_factors) < seq_len:
            b_factors = torch.cat([b_factors, b_factors[-1:].repeat(seq_len - len(b_factors))])
        else:
            b_factors = b_factors[:seq_len]

    # Update uncertainty channel
    graph_data[:, uncertainty_channel] = b_factors

    return graph_data


# Global PDB processor instance
_global_pdb_processor = None

def get_pdb_processor(pdb_directory: str = None, **kwargs) -> PDBProcessor:
    """Get global PDB processor instance."""
    global _global_pdb_processor

    if _global_pdb_processor is None or pdb_directory is not None:
        if pdb_directory is None:
            pdb_directory = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'datasets', 'all_chain_pdbs'
            )
        _global_pdb_processor = PDBProcessor(pdb_directory, **kwargs)

    return _global_pdb_processor

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
cath_dataset.py

This module provides dataset classes for loading protein structure data from
the CATH database and AlphaFold2 predictions. It handles both traditional
dPD            mask = torch.ones(L, dtype=torch.bool)

            # Validation: Confirm alignment is correct
            expected_total_nodes = L + (1 if data.use_virtual_node else 0)
            if data.num_nodes == expected_total_nodes:
                print(f"ALIGNMENT SUCCESS: Graph has {data.num_nodes} nodes "
                      f"({L} real + {1 if data.use_virtual_node else 0} virtual) "
                      f"matching sequence length {L} for {entry.get('name', 'unknown')}")
            else:
                print(f"ALIGNMENT ERROR: Graph has {data.num_nodes} nodes but sequence has {L} residues "
                      f"(virtual_node={data.use_virtual_node}) for {entry.get('name', 'unknown')}")

            #print(f"for {entry['name']}, data.shape: {data.x_s.shape}, y.shape: {y.shape}, mask.shape: {mask.shape}", flush=True)
            return data, y, maskith B-factors and AlphaFold2 predictions with pLDDT scores.

The main dataset class automatically detects the data source and processes
files accordingly, making it suitable for training on mixed datasets.
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .cif_parser import parse_cif_backbone_auto
from .graph_builder import GraphBuilder

# A mapping from the 3-letter amino acid code to an integer index.
# This is standard practice for representing categorical data in deep learning.
# Extended to include 'X' as the 21st amino acid for unknown residues.
AA_TO_IDX = {
 'ALA':0,'CYS':1,'ASP':2,'GLU':3,'PHE':4,'GLY':5,'HIS':6,'ILE':7,
 'LYS':8,'LEU':9,'MET':10,'ASN':11,'PRO':12,'GLN':13,'ARG':14,'SER':15,
 'THR':16,'VAL':17,'TRP':18,'TYR':19,'XXX':20  # Unknown residue
}

class CathDataset(Dataset):
    """
    Dataset for loading protein structures from CATH database and AlphaFold2.

    This dataset can handle both traditional PDB structures (with B-factors)
    and AlphaFold2 predictions (with pLDDT scores). It automatically detects
    the data source and processes files accordingly.

    The dataset loads pre-processed data from JSON and pickle files, which
    contain file paths and metadata for efficient training.
    """

    def __init__(self,
                 split_json: str,
                 map_pkl: str,
                 split: str = 'train',
                 max_len: Optional[int] = None,
                 graph_builder_kwargs: Optional[Dict] = None,

                 # Time sampling parameters (from training args) - NO DEFAULTS to enforce fail-fast
                 time_sampling_strategy: str = None,
                 t_min: float = None,
                 t_max: float = None,
                 alpha_range: Optional[float] = None,

                 # Optional protein list filter for efficient subsampling
                 protein_list_filter: Optional[List[str]] = None):
        """
        Initialize the CATH dataset.

        Args:
            split_json: Path to JSON file containing split information
            map_pkl: Path to pickle file containing file path mappings
            split: Dataset split ('train', 'val', 'test')
            max_len: Maximum sequence length to include (None for no limit)
            graph_builder_kwargs: Arguments for GraphBuilder initialization
            time_sampling_strategy: Time sampling strategy ('uniform', 'exponential', 'curriculum')
            t_min: Minimum time for sampling
            t_max: Maximum time for sampling
            alpha_range: Range scaling factor for exponential time sampling
            protein_list_filter: Optional list of protein names to include (for efficient subsampling)
        """
        self.split = split
        self.max_len = max_len
        self.protein_list_filter = set(protein_list_filter) if protein_list_filter else None

        # Print filter info if provided
        if self.protein_list_filter:
            print(f"Using protein list filter: {len(self.protein_list_filter)} proteins specified")

        # Validate time sampling parameters - no defaults allowed
        if time_sampling_strategy not in ['uniform', 'exponential', 'curriculum']:
            raise ValueError(f"time_sampling_strategy must be one of ['uniform', 'exponential', 'curriculum'], got: {time_sampling_strategy}")
        if t_min is None:
            raise ValueError("t_min must be explicitly provided - no default allowed")
        if t_max is None:
            raise ValueError("t_max must be explicitly provided - no default allowed")
        if alpha_range is None:
            raise ValueError("alpha_range must be explicitly provided - no default allowed")

        # Store time sampling parameters
        self.time_sampling_strategy = time_sampling_strategy
        self.t_min = t_min
        self.t_max = t_max
        self.alpha_range = alpha_range

        # Initialize graph builder
        if graph_builder_kwargs is None:
            graph_builder_kwargs = {}
        self.graph_builder = GraphBuilder(**graph_builder_kwargs)

        # Load split information
        with open(split_json, 'r') as f:
            split_data = json.load(f)

        # Load file path mappings
        with open(map_pkl, 'rb') as f:
            try:
                self.map_data = pickle.load(f)
            except Exception as e:
                print(f"Map pkl path was {map_pkl}", flush=True)
                raise Exception(e)



        # Store data entries directly instead of file paths
        self.entries = []
        filtered_count = 0
        for cath_id in split_data[split]:
            if cath_id in self.map_data:
                # Apply protein list filter if provided
                if self.protein_list_filter and cath_id not in self.protein_list_filter:
                    filtered_count += 1
                    continue

                entry = self.map_data[cath_id]
                if isinstance(entry, dict) and 'seq' in entry and 'coords' in entry:
                    if max_len is None or len(entry['seq']) <= max_len:
                        # CRITICAL FIX: Preserve the CATH ID as the structure name
                        entry = entry.copy()  # Create a copy to avoid modifying the original
                        entry['name'] = cath_id  # Add the CATH ID as the name field
                        self.entries.append(entry)

        if self.protein_list_filter:
            print(f"Filtered out {filtered_count} proteins not in protein list")
            print(f"Loaded {len(self.entries)} entries for {split} split (filtered from protein list)")
        else:
            print(f"Loaded {len(self.entries)} entries for {split} split")

        # Create name-to-index mapping for efficient protein lookup
        self._name_to_index = {}
        for idx, entry in enumerate(self.entries):
            if 'name' in entry:
                self._name_to_index[entry['name']] = idx

        # Length filtering is already done above when loading entries
        # No additional filtering needed since we already checked max_len

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.entries)

    def get_protein_index(self, protein_name: str) -> Optional[int]:
        """
        Get the dataset index for a specific protein name.

        Args:
            protein_name: Name of the protein to find

        Returns:
            Dataset index if found, None otherwise
        """
        return self._name_to_index.get(protein_name, None)

    def get_protein_names(self) -> List[str]:
        """
        Get all protein names in the dataset.

        Returns:
            List of protein names
        """
        return list(self._name_to_index.keys())

    def __getitem__(self, idx):
        """
        Get a single protein structure as a graph.

        Args:
            idx: Index of the protein to load

        Returns:
            data: PyTorch Geometric Data object
            y: one-hot encoded sequence tensor [L, 20]
            mask: boolean tensor [L] (all True for CATH)
            time_value: float sampled time value
            dssp_targets: DSSP class indices tensor [L] or None if no DSSP data
        """

        entry = self.entries[idx]
        try:
            # Use shared processing function to ensure consistency with unified dataset
            # Sample time for this individual protein using same logic as training
            # This time will be used for coordinate noise, sequence processing, and time embeddings
            import torch

            from .shared_processing import process_protein_entry

            # Time sampling strategy: use the same logic as training
            if self.time_sampling_strategy == 'uniform':
                # Uniform sampling between t_min and t_max
                time_value = torch.rand(1).item() * (self.t_max - self.t_min) + self.t_min
            elif self.time_sampling_strategy == 'exponential':
                # Exponential sampling aligned with Dirichlet flow matching (absolute scale)
                exp_sample = torch.distributions.Exponential(rate=1.0).sample((1,)).item()
                time_value = 0.0 + exp_sample * self.alpha_range  # Scale to [0.0, 10.0]
                time_value = max(0.0, min(10.0, time_value))  # Clamp to [0.0, 10.0]
            elif self.time_sampling_strategy == 'curriculum':
                # Curriculum learning: start with low time values, gradually increase
                # For now, implement as uniform (curriculum requires training step information)
                time_value = torch.rand(1).item() * (self.t_max - self.t_min) + self.t_min
            else:
                raise ValueError(f"time_sampling_strategy '{self.time_sampling_strategy}' is not supported. Use 'uniform', 'exponential', or 'curriculum'.")

            # Use strict validation for CATH dataset (original behavior)
            data, y, mask, dssp_targets = process_protein_entry(
                self.graph_builder,
                entry,
                source='pdb',  # CATH data treated as PDB source for source indicators
                strict_validation=True,
                time_param=time_value
            )

            return data, y, mask, time_value, dssp_targets

        except Exception as e:
            print(f"Error processing entry {entry.get('name', 'unknown')}: {e}")
            # Create a proper one-hot vector for dummy data (e.g., glycine)
            dummy_y = torch.zeros(1, 21)  # Extended to 21 classes
            dummy_y[0, 5] = 1.0  # Set to glycine (index 5 in AA_TO_IDX)
            return self._create_dummy_data(), dummy_y, torch.ones(1, dtype=torch.bool), 0.0, None


    def _filter_by_length(self, file_paths: List[str], max_len: int) -> List[str]:
        """
        Filter files by sequence length.

        Args:
            file_paths: List of file paths to filter
            max_len: Maximum allowed sequence length

        Returns:
            Filtered list of file paths
        """
        filtered_paths = []

        for file_path in file_paths:
            try:
                # Parse file to get sequence length
                coords, scores, residue_types, source = parse_cif_backbone_auto(file_path)
                if len(residue_types) <= max_len:
                    filtered_paths.append(file_path)
            except Exception as e:
                print(f"Error checking length of {file_path}: {e}")
                continue

        return filtered_paths

    def _create_dummy_data(self):
        """
        Create a dummy data object for error handling.

        Returns:
            PyTorch Geometric Data object with minimal structure
        """
        import torch_geometric.data as pyg_data

        # Create minimal dummy data
        dummy_data = pyg_data.Data(
            x_s=torch.zeros(1, 3),      # 1 residue, 3 scalar features
            x_v=torch.zeros(1, 3, 4),   # 1 residue, 3x4 vector features
            edge_index=torch.zeros(2, 0, dtype=torch.long),  # No edges
            edge_attr=torch.zeros(0, 20),  # No edges, 20 edge features
            num_nodes=1,
            source='error',
            file_path='error'
        )

        return dummy_data


class AlphaFold2Dataset(Dataset):
    """
    Specialized dataset for AlphaFold2 prediction files.

    This dataset is specifically designed for AlphaFold2 CIF files that
    contain pLDDT scores instead of B-factors. It provides additional
    functionality for handling confidence scores and filtering by quality.
    """

    def __init__(self,
                 file_paths: List[str],
                 min_plddt: float = 50.0,
                 max_len: Optional[int] = None,
                 graph_builder_kwargs: Optional[Dict] = None,

                 # Time sampling parameters (from training args) - NO DEFAULTS to enforce fail-fast
                 time_sampling_strategy: str = None,
                 t_min: float = None,
                 t_max: float = None,
                 alpha_range: Optional[float] = None):
        """
        Initialize the AlphaFold2 dataset.

        Args:
            file_paths: List of paths to AlphaFold2 CIF files
            min_plddt: Minimum average pLDDT score to include
            max_len: Maximum sequence length to include
            graph_builder_kwargs: Arguments for GraphBuilder initialization
            time_sampling_strategy: Time sampling strategy ('uniform', 'exponential', 'curriculum')
            t_min: Minimum time for sampling
            t_max: Maximum time for sampling
            alpha_range: Range scaling factor for exponential time sampling
        """
        self.min_plddt = min_plddt

        # Validate time sampling parameters - no defaults allowed
        if time_sampling_strategy not in ['uniform', 'exponential', 'curriculum']:
            raise ValueError(f"time_sampling_strategy must be one of ['uniform', 'exponential', 'curriculum'], got: {time_sampling_strategy}")
        if t_min is None:
            raise ValueError("t_min must be explicitly provided - no default allowed")
        if t_max is None:
            raise ValueError("t_max must be explicitly provided - no default allowed")
        if alpha_range is None:
            raise ValueError("alpha_range must be explicitly provided - no default allowed")

        # Store time sampling parameters
        self.time_sampling_strategy = time_sampling_strategy
        self.t_min = t_min
        self.t_max = t_max
        self.alpha_range = alpha_range
        self.max_len = max_len

        # Initialize graph builder
        if graph_builder_kwargs is None:
            graph_builder_kwargs = {}
        self.graph_builder = GraphBuilder(**graph_builder_kwargs)

        # Filter files by quality and length
        self.file_paths = self._filter_files(file_paths)

        print(f"Loaded {len(self.file_paths)} AlphaFold2 files "
              f"(min_plddt={min_plddt}, max_len={max_len})")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get a single AlphaFold2 structure as a graph.

        Args:
            idx: Index of the protein to load

        Returns:
            PyTorch Geometric Data object containing the protein graph
        """
        file_path = self.file_paths[idx]

        try:
            # Sample time for this individual protein using same logic as training
            # This time will be used for coordinate noise, sequence processing, and time embeddings
            import torch

            # Time sampling strategy: use the same logic as training
            if self.time_sampling_strategy == 'uniform':
                # Uniform sampling between t_min and t_max
                time_value = torch.rand(1).item() * (self.t_max - self.t_min) + self.t_min
            elif self.time_sampling_strategy == 'exponential':
                # Exponential sampling aligned with Dirichlet flow matching (absolute scale)
                exp_sample = torch.distributions.Exponential(rate=1.0).sample((1,)).item()
                time_value = 0.0 + exp_sample * self.alpha_range  # Scale to [0.0, 10.0]
                time_value = max(0.0, min(10.0, time_value))  # Clamp to [0.0, 10.0]
            elif self.time_sampling_strategy == 'curriculum':
                # Curriculum learning: start with low time values, gradually increase
                # For now, implement as uniform (curriculum requires training step information)
                time_value = torch.rand(1).item() * (self.t_max - self.t_min) + self.t_min
            else:
                raise ValueError(f"time_sampling_strategy '{self.time_sampling_strategy}' is not supported. Use 'uniform', 'exponential', or 'curriculum'.")

            # Build graph from AlphaFold2 CIF file
            data = self.graph_builder.build(file_path, time_param=time_value)

            data.file_path = file_path
            data.avg_plddt = torch.mean(data.x_s[:, 2]).item()  # pLDDT is 3rd feature

            return data, time_value

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return self._create_dummy_data(), 0.0

    def _filter_files(self, file_paths: List[str]) -> List[str]:
        """
        Filter AlphaFold2 files by quality and length.

        Args:
            file_paths: List of file paths to filter

        Returns:
            Filtered list of file paths
        """
        filtered_paths = []

        for file_path in file_paths:
            try:
                # Parse file to check quality and length
                coords, plddt_scores, residue_types, source = parse_cif_backbone_auto(file_path)

                # Check if it's actually an AlphaFold2 file
                if source != 'alphafold2':
                    continue

                # Check length
                if self.max_len is not None and len(residue_types) > self.max_len:
                    continue

                # Check average pLDDT score
                avg_plddt = torch.mean(plddt_scores).item()
                if avg_plddt >= self.min_plddt:
                    filtered_paths.append(file_path)

            except Exception as e:
                print(f"Error checking {file_path}: {e}")
                continue

        return filtered_paths

    def _create_dummy_data(self):
        """
        Create a dummy data object for error handling.

        Returns:
            PyTorch Geometric Data object with minimal structure
        """
        import torch_geometric.data as pyg_data

        # Create minimal dummy data
        dummy_data = pyg_data.Data(
            x_s=torch.zeros(1, 3),      # 1 residue, 3 scalar features
            x_v=torch.zeros(1, 3, 4),   # 1 residue, 3x4 vector features
            edge_index=torch.zeros(2, 0, dtype=torch.long),  # No edges
            edge_attr=torch.zeros(0, 20),  # No edges, 20 edge features
            num_nodes=1,
            source='alphafold2',
            file_path='error',
            avg_plddt=0.0
        )

        return dummy_data


def create_mixed_dataset(cath_json: str,
                        cath_pkl: str,
                        alphafold2_paths: List[str],
                        split: str = 'train',
                        cath_max_len: Optional[int] = None,
                        alphafold2_max_len: Optional[int] = None,
                        alphafold2_min_plddt: float = 50.0,
                        graph_builder_kwargs: Optional[Dict] = None) -> Dataset:
    """
    Create a mixed dataset combining CATH and AlphaFold2 data.

    Args:
        cath_json: Path to CATH split JSON file
        cath_pkl: Path to CATH mapping pickle file
        alphafold2_paths: List of AlphaFold2 CIF file paths
        split: Dataset split ('train', 'val', 'test')
        cath_max_len: Maximum length for CATH structures
        alphafold2_max_len: Maximum length for AlphaFold2 structures
        alphafold2_min_plddt: Minimum pLDDT score for AlphaFold2 structures
        graph_builder_kwargs: Arguments for GraphBuilder

    Returns:
        Combined dataset with both CATH and AlphaFold2 data
    """
    # Create CATH dataset
    cath_dataset = CathDataset(
        split_json=cath_json,
        map_pkl=cath_pkl,
        split=split,
        max_len=cath_max_len,
        graph_builder_kwargs=graph_builder_kwargs
    )

    # Create AlphaFold2 dataset
    alphafold2_dataset = AlphaFold2Dataset(
        file_paths=alphafold2_paths,
        min_plddt=alphafold2_min_plddt,
        max_len=alphafold2_max_len,
        graph_builder_kwargs=graph_builder_kwargs
    )

    # Combine datasets using torch.utils.data.ConcatDataset
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([cath_dataset, alphafold2_dataset])

    print(f"Created mixed dataset with {len(cath_dataset)} CATH and "
          f"{len(alphafold2_dataset)} AlphaFold2 structures")

    return combined_dataset

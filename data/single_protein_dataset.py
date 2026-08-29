# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Single protein dataset for efficient loading of one specific protein.
This bypasses loading the entire dataset when only one protein is needed.
"""

import json
import pickle

import torch
from torch.utils.data import Dataset

from .graph_builder import GraphBuilder


class SingleProteinDataset(Dataset):
    """
    Dataset for sampling from a single protein by name.
    This bypasses loading the entire dataset and focuses only on the target protein.
    """

    def __init__(
        self,
        split_json,
        map_pkl,
        protein_name,
        split,
        graph_builder_kwargs=None,
        # Time sampling parameters (required for compatibility)
        time_sampling_strategy="uniform",
        t_min=0.0,
        t_max=8.0,
        alpha_range=0.0,
    ):
        """
        Initialize dataset with single protein.

        Args:
            split_json: Path to split JSON file
            map_pkl: Path to mapping pickle file
            protein_name: Name of the specific protein to load
            split: Dataset split to search in
            graph_builder_kwargs: Parameters for graph builder
            time_sampling_strategy: Time sampling strategy ('uniform', 'exponential', 'curriculum')
            t_min: Minimum time for sampling
            t_max: Maximum time for sampling
            alpha_range: Range scaling factor for exponential time sampling
        """
        self.protein_name = protein_name
        self.split = split

        # Store time sampling parameters
        self.time_sampling_strategy = time_sampling_strategy
        self.t_min = t_min
        self.t_max = t_max
        self.alpha_range = alpha_range

        # Initialize graph builder
        if graph_builder_kwargs is None:
            graph_builder_kwargs = {}
        self.graph_builder = GraphBuilder(**graph_builder_kwargs)

        # Find and load only the target protein
        self.protein_entry = self._find_and_load_protein(split_json, map_pkl)

        if self.protein_entry is None:
            raise ValueError(f"Protein '{protein_name}' not found in {split} split")

        print(f"Single protein dataset created for: {protein_name}")
        print(f"Protein has {len(self.protein_entry['seq'])} residues")

    def _find_and_load_protein(self, split_json, map_pkl):
        """Find and load the specific protein from the dataset files."""

        # Load split information
        with open(split_json, "r") as f:
            splits = json.load(f)

        if self.split not in splits:
            raise ValueError(f"Split '{self.split}' not found in {split_json}")

        protein_ids = splits[self.split]
        print(
            f"Searching for '{self.protein_name}' among {len(protein_ids)} proteins in {self.split} split..."
        )

        # Load mapping data
        with open(map_pkl, "rb") as f:
            mapping_data = pickle.load(f)

        # Search for the target protein
        for protein_id in protein_ids:
            if protein_id in mapping_data:
                entry = mapping_data[protein_id]
                entry_name = entry.get("name", protein_id)

                if entry_name == self.protein_name:
                    print(f"Found protein '{self.protein_name}' with ID: {protein_id}")
                    return entry

        # If not found, list available proteins for debugging
        print(f"Protein '{self.protein_name}' not found. Available proteins:")
        for i, protein_id in enumerate(protein_ids[:10]):  # Show first 10
            if protein_id in mapping_data:
                entry = mapping_data[protein_id]
                entry_name = entry.get("name", protein_id)
                print(f"  {entry_name}")
            if i >= 9:
                print(f"  ... and {len(protein_ids) - 10} more")
                break

        return None

    def __len__(self):
        """Return 1 since we have only one protein."""
        return 1

    def __getitem__(self, idx):
        """
        Get the single protein structure as a graph.

        Args:
            idx: Must be 0 (only one protein)

        Returns:
            data: PyTorch Geometric Data object
            y: one-hot encoded sequence tensor [L, 20]
            mask: boolean tensor [L]
            time_value: float sampled time value
            dssp_targets: None (single protein doesn't have DSSP data)
        """
        if idx != 0:
            raise IndexError(
                f"SingleProteinDataset only has one entry (index 0), got {idx}"
            )

        # Sample time for this protein using same logic as CathDataset
        import torch

        # Time sampling strategy: use the same logic as training
        if self.time_sampling_strategy == "uniform":
            # Uniform sampling between t_min and t_max
            time_value = torch.rand(1).item() * (self.t_max - self.t_min) + self.t_min
        elif self.time_sampling_strategy == "exponential":
            # Exponential sampling aligned with Dirichlet flow matching (absolute scale)
            exp_sample = torch.distributions.Exponential(rate=1.0).sample((1,)).item()
            time_value = 0.0 + exp_sample * self.alpha_range  # Scale to [0.0, 10.0]
            time_value = max(0.0, min(10.0, time_value))  # Clamp to [0.0, 10.0]
        elif self.time_sampling_strategy == "curriculum":
            # Curriculum learning: start with low time values, gradually increase
            # For now, implement as uniform (curriculum requires training step information)
            time_value = torch.rand(1).item() * (self.t_max - self.t_min) + self.t_min
        else:
            raise ValueError(
                f"time_sampling_strategy '{self.time_sampling_strategy}' is not supported. Use 'uniform', 'exponential', or 'curriculum'."
            )

        # Use shared processing function to ensure consistency
        try:
            from .shared_processing import process_protein_entry

            data, y, mask, dssp_targets = process_protein_entry(
                self.graph_builder,
                self.protein_entry,
                source="pdb",  # Single protein data treated as PDB source
                strict_validation=True,
                time_param=time_value,
            )
            return data, y, mask, time_value, dssp_targets
        except ImportError:
            # Fallback to original implementation if shared_processing not available
            # Build graph from the single protein entry
            data = self.graph_builder.build_from_dict(self.protein_entry, time_value)
            data.source = "pdb"  # Single protein data treated as PDB source
            data.name = self.protein_entry.get("name", "unknown")

            # Create y and mask (same logic as CathDataset)
            if data.use_virtual_node:
                L = data.num_nodes - 1
            else:
                L = data.num_nodes

            original_seq = self.protein_entry.get(
                "original_seq", self.protein_entry["seq"]
            )

            if len(original_seq) != L:
                raise ValueError(f"Sequence length mismatch for {self.protein_name}")

            # Convert sequence to one-hot encoding
            aa_to_idx = {
                "ALA": 0,
                "CYS": 1,
                "ASP": 2,
                "GLU": 3,
                "PHE": 4,
                "GLY": 5,
                "HIS": 6,
                "ILE": 7,
                "LYS": 8,
                "LEU": 9,
                "MET": 10,
                "ASN": 11,
                "PRO": 12,
                "GLN": 13,
                "ARG": 14,
                "SER": 15,
                "THR": 16,
                "VAL": 17,
                "TRP": 18,
                "TYR": 19,
                "XXX": 20,
            }

            # Also support 1-letter amino acid codes
            aa_1to3 = {
                "A": "ALA",
                "C": "CYS",
                "D": "ASP",
                "E": "GLU",
                "F": "PHE",
                "G": "GLY",
                "H": "HIS",
                "I": "ILE",
                "K": "LYS",
                "L": "LEU",
                "M": "MET",
                "N": "ASN",
                "P": "PRO",
                "Q": "GLN",
                "R": "ARG",
                "S": "SER",
                "T": "THR",
                "V": "VAL",
                "W": "TRP",
                "Y": "TYR",
                "X": "XXX",
            }

            y = torch.zeros(L, 21)  # 21 classes including XXX
            unknown_count = 0

            for i, aa in enumerate(original_seq):
                aa_3letter = None

                if isinstance(aa, str):
                    if len(aa) == 3:
                        # 3-letter code
                        aa_upper = aa.upper()
                        if aa_upper in aa_to_idx:
                            aa_3letter = aa_upper
                    elif len(aa) == 1:
                        # 1-letter code
                        aa_upper = aa.upper()
                        if aa_upper in aa_1to3:
                            aa_3letter = aa_1to3[aa_upper]

                if aa_3letter and aa_3letter in aa_to_idx:
                    y[i, aa_to_idx[aa_3letter]] = 1
                else:
                    y[i, aa_to_idx["XXX"]] = 1
                    unknown_count += 1

            if unknown_count > 0:
                print(f"Warning: Mapped {unknown_count}/{L} residues to unknown (XXX)")

            mask = torch.ones(L, dtype=torch.bool)

            return data, y, mask, time_value, None  # No DSSP data for single protein

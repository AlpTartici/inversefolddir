# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
shared_processing.py

Shared data processing functions for both CATH and AF2 datasets to ensure identical processing.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from .dssp_constants import DSSP_TO_IDX


def create_target_tensor_and_mask(data, entry: Dict[str, Any], strict_validation: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Create target tensor (y), mask, and DSSP targets from graph data and entry.
    This function ensures identical processing for both CATH and AF2 datasets.

    Args:
        data: Graph data object from GraphBuilder
        entry: Dictionary containing protein data
        strict_validation: If True, raise error for non-standard amino acids; if False, use unknown class

    Returns:
        Tuple of (y, mask, dssp_targets) where:
        - y: One-hot encoded amino acid tensor [L, 21]
        - mask: Boolean mask tensor [L]
        - dssp_targets: DSSP class indices tensor [L] or None if no DSSP data
    """
    from .cath_dataset import AA_TO_IDX

    # Calculate real node count
    if data.use_virtual_node:
        L = data.num_nodes - 1  # Real nodes = total - virtual
    else:
        L = data.num_nodes  # Real nodes = total

    # Use filtered sequence from graph builder as ground truth
    # The graph builder ensures perfect alignment between coordinates and sequence
    filtered_seq = getattr(data, 'filtered_seq', entry['seq'])
    ground_truth_seq = filtered_seq

    # Verify alignment
    if len(ground_truth_seq) != L:
        raise ValueError(f"CRITICAL ERROR: Ground truth sequence length ({len(ground_truth_seq)}) != "
                        f"expected real nodes ({L}) for {entry.get('name', 'unknown')}. "
                        f"Graph has {data.num_nodes} total nodes, use_virtual_node={data.use_virtual_node}")

    # Create one-hot tensor (21 classes: 20 standard AAs + 1 unknown)
    y = torch.zeros(L, 21, dtype=torch.float32)

    # Standard amino acid mapping (1-letter to 3-letter)
    standard_aa_map = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        'X': 'XXX'  # Unknown residue mapping
    }

    for i, aa in enumerate(ground_truth_seq):
        idx_aa = None

        # Handle both 1-letter and 3-letter codes
        if len(aa) == 1:
            # Convert 1-letter to 3-letter using standard mapping
            aa3 = standard_aa_map.get(aa, None)
            if aa3 is not None:
                idx_aa = AA_TO_IDX.get(aa3, None)
        else:
            idx_aa = AA_TO_IDX.get(aa, None)

        if idx_aa is not None:
            y[i, idx_aa] = 1.0
        else:
            if strict_validation:
                # CATH dataset behavior: raise error for non-standard amino acids
                raise ValueError(f"CRITICAL ERROR: Non-standard amino acid '{aa}' found in filtered sequence "
                               f"for {entry.get('name', 'unknown')} at position {i}. "
                               f"Graph builder should have removed this!")
            else:
                # AF2 dataset behavior: default to unknown residue
                y[i, AA_TO_IDX['XXX']] = 1.0

    # Process DSSP targets if available
    dssp_targets = None
    dssp_seq = entry.get('dssp')
    if dssp_seq is not None:
        # Convert DSSP characters to indices
        dssp_targets = torch.zeros(L, dtype=torch.long)

        # Handle potential length mismatch (graph builder filtering)
        if len(dssp_seq) != L:
            # DSSP sequence might not have been filtered by graph builder
            # Use the same filtering logic as the sequence
            filtered_dssp = getattr(data, 'filtered_dssp', dssp_seq)
            if len(filtered_dssp) == L:
                dssp_seq = filtered_dssp
            else:
                print(f"WARNING: DSSP length mismatch for {entry.get('name', 'unknown')}: "
                      f"DSSP={len(dssp_seq)}, sequence={L}. Skipping DSSP targets.")
                dssp_targets = None

        if dssp_targets is not None:
            for i, dssp_char in enumerate(dssp_seq):
                dssp_idx = DSSP_TO_IDX.get(dssp_char, DSSP_TO_IDX['X'])  # Default to 'X' for unknown
                dssp_targets[i] = dssp_idx

    # Create mask (all True for now, could be modified for missing residues)
    mask = torch.ones(L, dtype=torch.bool)

    return y, mask, dssp_targets

def process_protein_entry(graph_builder, entry: Dict[str, Any], source: str, strict_validation: bool = True, time_param: float = None) -> Tuple[Any, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Complete processing pipeline for a protein entry.
    This ensures identical processing for both CATH and AF2 datasets.

    Args:
        graph_builder: GraphBuilder instance
        entry: Dictionary containing protein data
        source: Source identifier ('cath', 'af2', etc.)
        strict_validation: If True, use strict validation for amino acids
        time_param: Time parameter for time-dependent coordinate noise (None = no time dependence)

    Returns:
        Tuple of (data, y, mask, dssp_targets)
    """
    # Build graph from entry (identical for all sources)
    data = graph_builder.build_from_dict(entry, time_param=time_param)

    # Set source information
    data.source = source
    data.name = entry.get('name', 'unknown')

    # STORE ORIGINAL ENTRY for validation graph rebuilding
    # This allows the validation fix to rebuild graphs from scratch with different time parameters
    data._original_entry = entry.copy()  # Store a copy of the original entry

    # Create target tensor, mask, and DSSP targets using shared logic
    y, mask, dssp_targets = create_target_tensor_and_mask(data, entry, strict_validation=strict_validation)

    return data, y, mask, dssp_targets

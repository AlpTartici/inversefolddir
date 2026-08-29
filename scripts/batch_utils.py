#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Utility functions for batched inpainting processing.

Provides functions for:
- Determining optimal batch size based on protein length
- Checking if positions have already been processed
- Filtering unprocessed positions
- Estimating sequence length from PDB files
"""

import os
import logging
from typing import Optional

import pandas as pd


def determine_batch_size(sequence_length: int, config_batch_size: int = 8) -> int:
    """
    Dynamically determine batch size based on sequence length.

    Args:
        sequence_length: Length of the protein sequence
        config_batch_size: Default batch size from config

    Returns:
        Batch size to use (6 for large proteins >600aa, config value otherwise)
    """
    if sequence_length > 600:
        return 6
    return config_batch_size


def check_position_exists(output_dir: str, pos_valid: str) -> bool:
    """
    Check if a position has already been processed.

    Args:
        output_dir: Base output directory
        pos_valid: Position string (e.g., "A392")

    Returns:
        True if the position result file exists, False otherwise
    """
    pos_dir = os.path.join(output_dir, f"pos_{pos_valid}")
    result_file = os.path.join(pos_dir, "inpainting_results.npz")
    return os.path.exists(result_file)


def filter_unprocessed_positions(
    positions_df: pd.DataFrame,
    output_dir: str,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Filter out positions that already have results.

    Args:
        positions_df: DataFrame with position information
        output_dir: Base output directory
        logger: Optional logger instance

    Returns:
        Filtered DataFrame with only unprocessed positions
    """
    mask = positions_df.apply(
        lambda row: not check_position_exists(output_dir, row['pos_valid']),
        axis=1
    )
    filtered_df = positions_df[mask].copy()

    num_skipped = len(positions_df) - len(filtered_df)
    if num_skipped > 0 and logger:
        logger.info(f"Skipping {num_skipped} already-processed positions")

    return filtered_df


def estimate_sequence_length(pdb_file: str, map_pkl_path: Optional[str] = None) -> int:
    """
    Estimate sequence length for batch size determination.

    Returns conservative default of 600 if cannot determine.
    This is safe since it will use smaller batch size (6) which works for all proteins.

    Args:
        pdb_file: Path to PDB file
        map_pkl_path: Optional path to map PKL file (not currently used)

    Returns:
        Estimated sequence length (default: 600)
    """
    # Option 1: Quick PDB file parsing (count CA atoms)
    try:
        if os.path.exists(pdb_file):
            ca_count = 0
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and ' CA ' in line:
                        ca_count += 1
            if ca_count > 0:
                return ca_count
    except Exception:
        pass

    # Option 2: Conservative default
    return 600  # Will use batch_size=6, safe for all proteins

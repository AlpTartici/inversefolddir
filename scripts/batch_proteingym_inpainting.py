#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Batch inpainting for ProteinGym mutation analysis with parallel processing.

This script processes ProteinGym CSV files to run inpainting on unique positions
and extract probability distributions for mutation effect prediction.
Supports parallel processing of multiple positions simultaneously.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add import from helpers for config parsing
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.join(Path(__file__).parent.parent, 'helpers'))
from build_inpainting_cmd import build_inpainting_args
from paths import default_chain_set_map

# Import batch utilities
from batch_utils import (
    determine_batch_size,
    check_position_exists,
    filter_unprocessed_positions,
    estimate_sequence_length
)

# AA mappings matching sample_utils.py IDX_TO_AA ordering
_IDX_TO_AA = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
    'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
    'THR', 'VAL', 'TRP', 'TYR', 'XXX'
]
_AA_TO_IDX = {aa: i for i, aa in enumerate(_IDX_TO_AA)}
_ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}


def setup_logging(verbose: bool = False, log_file: str = None) -> logging.Logger:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def extract_unique_positions(csv_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Extract unique positions from ProteinGym CSV file.

    Args:
        csv_path: Path to CSV file
        logger: Logger instance

    Returns:
        DataFrame with columns: pdb_file, pos_valid, pos (numeric)
    """
    logger.info(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['pos_valid', 'pdb_file', 'pos']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")


    # Extract unique combinations of pos_valid and pdb_file
    unique_positions = df[['pdb_file', 'pos_valid', 'pos']].drop_duplicates()

    logger.info(f"Found {len(unique_positions)} unique positions")
    logger.debug(f"Sample positions:\n{unique_positions.head()}")

    return unique_positions


def build_batch_inpainting_command(
    config: dict,
    pdb_file: str,
    position_list: str,
    output_dir: str,
    project_root: str,
) -> List[str]:
    """
    Build inpainting command for multiple mask configurations (batched).

    Args:
        config: Configuration dictionary
        pdb_file: Path to PDB file
        position_list: Semicolon-separated position configs (e.g., '392;143;56')
        output_dir: Base output directory
        project_root: Root directory of the project

    Returns:
        List of command arguments
    """
    # Build full command
    cmd = [
        sys.executable,
        os.path.join(project_root, 'training', 'inpainting.py'),
        '--pdb_input', pdb_file,
        '--batch-mask-configs', position_list,  # Use new parameter with semicolon separator
        '--output-dir', output_dir,
        '--split_json', os.path.join(project_root, 'datasets', 'cath-4.2', 'chain_set_splits.json'),
        '--map_pkl', default_chain_set_map(),
    ]

    # Add config args using imported function
    config_args = build_inpainting_args(config)
    cmd.extend(config_args)

    # Force disable trajectory for batching (memory savings)
    if '--detailed_json' in cmd:
        cmd.remove('--detailed_json')

    return cmd


def score_results(
    csv_path: str,
    output_dir: str,
    csv_name: str,
    logger: logging.Logger
) -> None:
    """
    Read the consolidated NPZ for this protein and add wt_prob, mut_prob, llr
    columns to the original CSV, writing <csv_name>_scored.csv into output_dir.
    """
    npz_path = os.path.join(output_dir, f"{csv_name}_inpainting_probs.npz")
    if not os.path.exists(npz_path):
        logger.warning(f"NPZ not found, skipping scoring: {npz_path}")
        return

    logger.info(f"Scoring {csv_name}...")

    # Load NPZ and build pos_valid -> prob vector map
    data = np.load(npz_path, allow_pickle=True)

    # Determine AA index mapping (use stored if available, else default)
    if 'aa_names' in data:
        stored = list(data['aa_names'])
        aa_to_idx = {aa: i for i, aa in enumerate(stored)}
    else:
        aa_to_idx = _AA_TO_IDX

    prob_map = {}
    for pos_valid in data['positions']:
        key = f'{pos_valid}_probs'
        if key in data:
            prob_map[pos_valid] = data[key]

    # Load original CSV
    df = pd.read_csv(csv_path)

    wt_probs, mut_probs, llrs = [], [], []
    missing_positions = set()

    for _, row in df.iterrows():
        mutant_str = str(row['mutant'])
        pos_valid = str(row['pos_valid'])

        # Parse mutant string e.g. "A25C" -> wt='A', mut='C'
        if len(mutant_str) < 3 or not mutant_str[0].isalpha() or not mutant_str[-1].isalpha() or not mutant_str[1:-1].isdigit():
            wt_probs.append(np.nan); mut_probs.append(np.nan); llrs.append(np.nan)
            continue

        wt_1, mut_1 = mutant_str[0], mutant_str[-1]

        if pos_valid not in prob_map:
            if pos_valid not in missing_positions:
                logger.warning(f"  Position '{pos_valid}' not in NPZ")
                missing_positions.add(pos_valid)
            wt_probs.append(np.nan); mut_probs.append(np.nan); llrs.append(np.nan)
            continue

        probs = prob_map[pos_valid]
        wt_3 = _ONE_TO_THREE.get(wt_1)
        mut_3 = _ONE_TO_THREE.get(mut_1)

        if wt_3 not in aa_to_idx or mut_3 not in aa_to_idx:
            wt_probs.append(np.nan); mut_probs.append(np.nan); llrs.append(np.nan)
            continue

        wt_p = float(probs[aa_to_idx[wt_3]])
        mut_p = float(probs[aa_to_idx[mut_3]])
        wt_probs.append(wt_p)
        mut_probs.append(mut_p)
        if wt_p > 0 and mut_p > 0:
            llrs.append(np.log(mut_p / wt_p))
        else:
            llrs.append(np.nan)

    df['wt_prob'] = wt_probs
    df['mut_prob'] = mut_probs
    df['llr'] = llrs

    n_scored = int(np.sum(~np.isnan(llrs)))
    logger.info(f"  Scored {n_scored}/{len(df)} rows")
    if missing_positions:
        logger.warning(f"  {len(missing_positions)} positions missing from NPZ: {sorted(missing_positions)}")

    scored_path = os.path.join(output_dir, f"{csv_name}_scored.csv")
    df.to_csv(scored_path, index=False)
    logger.info(f"  Scored CSV: {scored_path}")


def consolidate_results(
    output_dir: str,
    csv_name: str,
    positions_df: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """
    Consolidate individual position results into a single NPZ file.

    Args:
        output_dir: Base output directory
        csv_name: Name of the CSV file (without extension)
        positions_df: DataFrame with position information
        logger: Logger instance
    """
    logger.info("Consolidating results...")

    consolidated_data = {
        'positions': [],
        'aa_names': None,
        'pdb_file': None,
        'csv_source': f"{csv_name}.csv"
    }

    for idx, row in positions_df.iterrows():
        pos_valid = row['pos_valid']
        pdb_file = row['pdb_file']

        # Path to individual result file
        pos_dir = os.path.join(output_dir, f"pos_{pos_valid}")
        result_file = os.path.join(pos_dir, 'inpainting_results.npz')

        if not os.path.exists(result_file):
            logger.warning(f"Result file not found for position {pos_valid}: {result_file}")
            continue

        # Load NPZ file
        logger.debug(f"Loading results for position {pos_valid}")
        data = np.load(result_file, allow_pickle=True)

        # Extract probability for the masked position
        final_probs = data['final_probabilities']  # Shape: [N, 21]
        inpainting_mask = data['inpainting_mask']  # Shape: [N], bool

        # Find masked position index
        masked_indices = np.where(inpainting_mask)[0]

        if len(masked_indices) == 0:
            logger.warning(f"No masked positions found in results for {pos_valid}")
            continue

        if len(masked_indices) > 1:
            logger.warning(f"Multiple masked positions found for {pos_valid}, using first one")

        masked_idx = masked_indices[0]
        probs_for_position = final_probs[masked_idx, :]  # Shape: [21]

        # Store in consolidated data
        consolidated_data['positions'].append(pos_valid)
        consolidated_data[f'{pos_valid}_probs'] = probs_for_position

        # Store AA names (same for all positions)
        if consolidated_data['aa_names'] is None and 'aa_index_to_name' in data:
            consolidated_data['aa_names'] = data['aa_index_to_name']

        # Store PDB file (should be same for all)
        if consolidated_data['pdb_file'] is None:
            consolidated_data['pdb_file'] = pdb_file

    # Convert positions list to array
    consolidated_data['positions'] = np.array(consolidated_data['positions'])

    # Save consolidated NPZ
    output_file = os.path.join(output_dir, f"{csv_name}_inpainting_probs.npz")
    np.savez_compressed(output_file, **consolidated_data)

    logger.info(f"Consolidated results saved to: {output_file}")
    logger.info(f"Total positions saved: {len(consolidated_data['positions'])}")


def save_summary_csv(
    output_dir: str,
    csv_name: str,
    positions_df: pd.DataFrame,
    results: List[Dict],
    logger: logging.Logger
) -> None:
    """
    Save summary CSV with processing metadata.

    Args:
        output_dir: Output directory
        csv_name: Name of the CSV file
        positions_df: DataFrame with position information
        results: List of result dictionaries
        logger: Logger instance
    """
    summary_data = []

    for result in results:
        summary_data.append({
            'position': result['position'],
            'status': 'success' if result['success'] else 'failed',
            'elapsed_time': result['elapsed_time']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"{csv_name}_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    logger.info(f"Summary saved to: {summary_file}")


def initialize_model_and_dataset(config: dict, project_root: str, logger: logging.Logger):
    """
    Initialize model and dataset for direct-call mode.

    Args:
        config: Configuration dictionary
        project_root: Project root directory
        logger: Logger instance

    Returns:
        Tuple of (sampling_coordinator, dataset) where dataset is the coordinator's dataset
    """
    import sys
    import torch
    sys.path.insert(0, project_root)

    from training.sample_utils import SamplingCoordinator

    logger.info("Initializing model (one-time load)...")

    # Build args object for SamplingCoordinator
    class Args:
        def __init__(self):
            self.model = config.get('model_path')
            self.steps = config.get('steps', 20)
            self.flow_temp = config.get('flow_temp', 0.1)
            self.t_max = 8.0
            self.t_min = 0.0
            self.dirichlet_concentration = config.get('dirichlet_concentration', 200)
            # NOTE: this defaults to True, unlike the design entry points, where
            # the c_factor is off unless asked for. It is kept as-is because the
            # published zero-shot numbers were produced this way -- but it is not
            # a neutral choice. With the c_factor on, the terminal simplex state
            # is diffuse by construction (Dirichlet marginal ~0.31 at K=21,
            # T=8), and the saved probabilities this script scores come from that
            # state rather than from the model's own prediction. Set
            # "use_c_factor": false in the config to score the sharper
            # distribution instead; expect the numbers to move.
            self.use_c_factor = config.get('use_c_factor', True)
            self.dssp_initialization = config.get('dssp_initialization', False)
            self.dssp_guidance = config.get('dssp_guidance', False)
            self.filter_out_missing_flanks = config.get('filter_out_missing_flanks', False)
            self.ensemble_size = config.get('ensemble_size', 1)
            self.ensemble_consensus_strength = config.get('ensemble_consensus_strength', 0.7)
            self.ensemble_method = config.get('ensemble_method', 'arithmetic')
            self.structure_noise_mag_std = config.get('structure_noise_mag_std', 0.0)
            self.time_as_temperature = config.get('time_as_temperature', False)
            self.use_smoothed_targets = config.get('use_smoothed_targets', False)
            self.use_smoothed_labels = config.get('use_smoothed_labels', False)
            self.uncertainty_struct_noise_scaling = False
            self.verbose = False
            self.rbf_3d_min = 2.0
            self.rbf_3d_max = 350.0
            self.rbf_3d_spacing = 'exponential'
            # Add paths for dataset loading
            self.split_json = os.path.join(project_root, 'datasets/cath-4.2/chain_set_splits.json')
            self.map_pkl = default_chain_set_map()

    args = Args()

    # Initialize coordinator (this loads model and creates dataset)
    coordinator = SamplingCoordinator(
        model_path=config.get('model_path'),
        dataset_path="",  # Will be extracted from checkpoint
        split='test'
    )

    # Load model and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coordinator.load_model_and_dataset(device, args)

    logger.info(f"✓ Model loaded on device: {device}")
    logger.info(f"✓ Dataset loaded with {len(coordinator.dataset)} entries")

    # Return coordinator and its dataset
    return coordinator, coordinator.dataset


def process_csv_file_direct(
    csv_path: str,
    config: dict,
    output_base_dir: str,
    project_root: str,
    logger: logging.Logger,
    sampling_coordinator,
    dataset
) -> None:
    """
    Process a single CSV file using direct function calls (no subprocess).

    This is the optimized path that avoids model reloading overhead.

    Args:
        csv_path: Path to CSV file
        config: Configuration dictionary
        output_base_dir: Base output directory
        project_root: Root directory of the project
        logger: Logger instance
        sampling_coordinator: Pre-loaded SamplingCoordinator
        dataset: Pre-loaded dataset
    """
    import sys
    sys.path.insert(0, project_root)
    from training.inpainting import run_batch_inpainting_direct

    # Extract CSV name
    csv_name = Path(csv_path).stem

    # Create output directory
    output_dir = os.path.join(output_base_dir, csv_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Processing CSV (DIRECT MODE): {csv_name}")
    logger.info(f"Output directory: {output_dir}")

    # Extract unique positions
    positions_df = extract_unique_positions(csv_path, logger)
    original_positions_df = positions_df.copy()

    # Filter out already-processed positions
    positions_df = filter_unprocessed_positions(positions_df, output_dir, logger)

    if len(positions_df) == 0:
        logger.info("All positions already processed, running consolidation only")
        consolidate_results(output_dir, csv_name, original_positions_df, logger)
        score_results(csv_path, output_dir, csv_name, logger)
        save_summary_csv(output_dir, csv_name, original_positions_df, [], logger)
        return

    # Group by pdb_file
    grouped_by_pdb = positions_df.groupby('pdb_file')
    logger.info(f"Positions grouped into {len(grouped_by_pdb)} PDB file(s)")

    all_results = []

    # Process each PDB group
    for pdb_file, group_df in grouped_by_pdb:
        logger.info(f"Processing {len(group_df)} positions for PDB: {pdb_file}")

        # Load protein data once per PDB
        logger.info(f"  Loading PDB structure: {pdb_file}")
        protein_data = dataset.load_structure_from_pdb(
            pdb_file,
            chain=None,  # Auto-detect
            use_c_factor=config.get('use_c_factor', True)
        )

        # Determine batch size
        seq_length = estimate_sequence_length(pdb_file)
        config_batch_size = config.get('batch_size', config.get('batch_size ', 8))
        batch_size = determine_batch_size(seq_length, config_batch_size)

        logger.info(f"  Sequence length: {seq_length}, using batch size: {batch_size}")

        # Create batches
        num_batches = (len(group_df) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(group_df))
            batch_df = group_df.iloc[start_idx:end_idx].copy()

            # Build position list (semicolon-separated)
            pos_list = ';'.join(batch_df['pos'].astype(str).tolist())
            pos_valid_list = batch_df['pos_valid'].tolist()

            logger.info(f"  Batch {batch_idx+1}/{num_batches}: {len(batch_df)} positions {pos_valid_list}")

            # Run direct batch inpainting
            start_time = time.time()
            try:
                batch_results = run_batch_inpainting_direct(
                    sampling_coordinator=sampling_coordinator,
                    protein_data=protein_data,
                    batch_mask_configs=pos_list,
                    output_dir=output_dir,
                    config=config,
                    project_root=project_root,
                    verbose=False
                )
                elapsed = time.time() - start_time

                # Verify all positions saved
                for pos_valid in pos_valid_list:
                    pos_file = os.path.join(output_dir, f"pos_{pos_valid}", "inpainting_results.npz")
                    if not os.path.exists(pos_file):
                        raise RuntimeError(
                            f"Position {pos_valid} not saved after batch processing. "
                            f"Expected: {pos_file}"
                        )

                # Track results
                for result in batch_results:
                    all_results.append({
                        'success': result['success'],
                        'position': result['position'],
                        'elapsed_time': elapsed / len(batch_df)
                    })

                logger.info(f"  ✓ Batch completed in {elapsed:.2f}s")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"  ✗ Batch FAILED after {elapsed:.2f}s")
                logger.error(f"  Error: {str(e)}")

                # Fail-fast
                raise RuntimeError(
                    f"Batch inpainting failed for positions {pos_valid_list}:\\n"
                    f"Error: {str(e)}"
                )

    # Consolidate results
    consolidate_results(output_dir, csv_name, original_positions_df, logger)
    score_results(csv_path, output_dir, csv_name, logger)
    save_summary_csv(output_dir, csv_name, original_positions_df, all_results, logger)

    logger.info(f"CSV processing complete (DIRECT MODE): {csv_name}")


def process_csv_file(
    csv_path: str,
    config: dict,
    output_base_dir: str,
    project_root: str,
    logger: logging.Logger,
    num_workers: int = 1
) -> None:
    """
    Process a single CSV file with batched position processing (subprocess mode).

    Args:
        csv_path: Path to CSV file
        config: Configuration dictionary
        output_base_dir: Base output directory
        project_root: Root directory of the project
        logger: Logger instance
        num_workers: Number of parallel workers (DEPRECATED - now using batching instead)
    """
    # Extract CSV name (without extension)
    csv_name = Path(csv_path).stem

    # Create output directory
    output_dir = os.path.join(output_base_dir, csv_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Processing CSV: {csv_name}")
    logger.info(f"Output directory: {output_dir}")

    # Extract unique positions
    positions_df = extract_unique_positions(csv_path, logger)

    # Keep original for consolidation at end
    original_positions_df = positions_df.copy()

    # Filter out already-processed positions
    positions_df = filter_unprocessed_positions(positions_df, output_dir, logger)

    if len(positions_df) == 0:
        logger.info("All positions already processed, running consolidation only")
        consolidate_results(output_dir, csv_name, original_positions_df, logger)
        score_results(csv_path, output_dir, csv_name, logger)
        save_summary_csv(output_dir, csv_name, original_positions_df, [], logger)
        return

    # Group by pdb_file (required for batching - same structure)
    grouped_by_pdb = positions_df.groupby('pdb_file')
    logger.info(f"Positions grouped into {len(grouped_by_pdb)} PDB file(s)")

    all_results = []  # Collect results across all PDB groups

    # Process each PDB group with batching
    for pdb_file, group_df in grouped_by_pdb:
        logger.info(f"Processing {len(group_df)} positions for PDB: {pdb_file}")

        # Determine batch size based on sequence length
        seq_length = estimate_sequence_length(pdb_file)
        config_batch_size = config.get('batch_size', config.get('batch_size ', 8))  # Handle space in key
        batch_size = determine_batch_size(seq_length, config_batch_size)

        logger.info(f"  Sequence length: {seq_length}, using batch size: {batch_size}")

        # Create batches for this PDB
        num_batches = (len(group_df) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(group_df))
            batch_df = group_df.iloc[start_idx:end_idx].copy()

            # Build position list using semicolons to separate mask configurations
            # Each position is a separate mask configuration (single position masked)
            # Use pos_valid which includes amino acid prefix (e.g., "C116") for validation
            pos_valid_list = batch_df['pos_valid'].tolist()
            pos_list = ';'.join(pos_valid_list)

            logger.info(f"  Batch {batch_idx+1}/{num_batches}: {len(batch_df)} positions {pos_valid_list}")

            # Build command
            cmd = build_batch_inpainting_command(
                config, pdb_file, pos_list, output_dir, project_root
            )

            # Run batch
            start_time = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                elapsed = time.time() - start_time

                # Verify all positions saved
                for pos_valid in pos_valid_list:
                    pos_file = os.path.join(output_dir, f"pos_{pos_valid}", "inpainting_results.npz")
                    if not os.path.exists(pos_file):
                        raise RuntimeError(
                            f"Position {pos_valid} not saved after batch processing. "
                            f"Expected: {pos_file}"
                        )

                # Track results
                for pos_valid in pos_valid_list:
                    all_results.append({
                        'success': True,
                        'position': pos_valid,
                        'elapsed_time': elapsed / len(batch_df)  # Approximate per position
                    })

                logger.info(f"  ✓ Batch completed in {elapsed:.2f}s")

            except subprocess.CalledProcessError as e:
                elapsed = time.time() - start_time

                logger.error(f"  ✗ Batch FAILED after {elapsed:.2f}s")
                logger.error(f"  Exit code: {e.returncode}")

                # Save error logs
                error_dir = os.path.join(output_dir, 'batch_errors')
                os.makedirs(error_dir, exist_ok=True)

                if e.stderr:
                    error_log = os.path.join(error_dir, f'batch_{batch_idx+1}_stderr.log')
                    with open(error_log, 'w') as f:
                        f.write(e.stderr)
                    logger.error(f"  Stderr saved to: {error_log}")

                if e.stdout:
                    stdout_log = os.path.join(error_dir, f'batch_{batch_idx+1}_stdout.log')
                    with open(stdout_log, 'w') as f:
                        f.write(e.stdout)
                    logger.error(f"  Stdout saved to: {stdout_log}")

                # Fail-fast
                raise RuntimeError(
                    f"Batch inpainting failed for positions {pos_valid_list}:\n"
                    f"Exit code: {e.returncode}\n"
                    f"Check logs in: {error_dir}/"
                )

    # Consolidate results
    consolidate_results(output_dir, csv_name, original_positions_df, logger)
    score_results(csv_path, output_dir, csv_name, logger)

    # Save summary CSV
    save_summary_csv(output_dir, csv_name, original_positions_df, all_results, logger)

    logger.info(f"CSV processing complete: {csv_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch inpainting for ProteinGym mutation analysis'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output_proteingym_inpainting',
        help='Output base directory (default: ./output_proteingym_inpainting)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers for processing positions (default: 1)'
    )
    parser.add_argument(
        '--use-direct-calls',
        action='store_true',
        help='(Unmaintained -- refuses to run. Subprocess mode is the supported path.)'
    )

    args = parser.parse_args()

    # The direct-call path was an optimisation that avoided reloading the model
    # per position. It has since rotted: it calls CathDataset.load_structure_from_pdb,
    # which no longer exists, and hardcodes the map_pkl path instead of honouring
    # --map_pkl. Failing here beats failing with an AttributeError several
    # minutes into a run. The code is left in place for anyone who wants to
    # revive it; see process_csv_file_direct below.
    if args.use_direct_calls:
        parser.error(
            "--use-direct-calls is unmaintained and does not work: it calls "
            "CathDataset.load_structure_from_pdb, which no longer exists, and it "
            "ignores --map_pkl. Run without the flag; subprocess mode is the "
            "supported path."
        )

    # Determine project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent.absolute()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.output_dir, 'processing.log')
    logger = setup_logging(args.verbose, log_file)

    logger.info("=" * 80)
    logger.info("Batch ProteinGym Inpainting")
    logger.info("=" * 80)
    logger.info(f"CSV file: {args.csv}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info("=" * 80)

    # Validate inputs
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config: {json.dumps(config, indent=2)}")

    # Process CSV file
    start_time = time.time()

    try:
        if args.use_direct_calls:
            # Direct-call mode: Load model once, reuse across batches
            logger.info("=" * 80)
            logger.info("Using DIRECT CALL mode (optimized, no subprocess)")
            logger.info("=" * 80)

            # Initialize model and dataset
            sampling_coordinator, dataset = initialize_model_and_dataset(
                config, str(project_root), logger
            )

            # Process using direct calls
            process_csv_file_direct(
                args.csv, config, args.output_dir, str(project_root), logger,
                sampling_coordinator, dataset
            )
        else:
            # Subprocess mode (backward compatible)
            logger.info("=" * 80)
            logger.info("Using SUBPROCESS mode (backward compatible)")
            logger.info("=" * 80)

            process_csv_file(
                args.csv, config, args.output_dir, str(project_root), logger, args.num_workers
            )

        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("Processing complete!")
        logger.info(f"Total time: {elapsed_time:.2f}s")
        logger.info("=" * 80)

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error("Processing failed!")
        logger.error(f"Total time: {elapsed_time:.2f}s")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 80)
        # Print full traceback for debugging
        import traceback
        logger.error("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

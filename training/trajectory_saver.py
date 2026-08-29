
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Trajectory Analysis Saver for Protein Sequence Sampling

This module provides functionality to save detailed sampling trajectories
in NPZ format for easy analysis of accuracy evolution over time.

Data Structure:
- Time points: Array of time values during sampling
- Ground truth: One-hot encoded ground truth for each position [N, K]
- Predictions: Model predictions (softmax) at each time point [T, N, K]
- Current state: Current denoised state at each time point [T, N, K]
"""

import os
from datetime import datetime

import numpy as np

# Per-protein arrays written by save_compact_trajectory_npz. Everything is [T, N] except
# the per-position vectors, which are [N].
COMPACT_STEP_FIELDS = (
    'pred_argmax', 'pred_maxprob', 'pred_top2gap', 'pred_entropy',
    'state_argmax', 'state_maxprob', 'state_top2gap', 'state_entropy',
)
COMPACT_POSITION_FIELDS = ('true_indices', 'dssp_targets')


def save_compact_trajectory_npz(results, output_path, metadata=None):
    """
    Save compact denoising trajectories produced by sample_multiple_proteins_batched
    (i.e. results carrying a 'trajectory' entry from --save_trajectory_npz).

    Unlike save_trajectory_analysis_npz, this stores per-step summary statistics rather
    than full [T, N, K] probability tensors: roughly 45 KB per protein instead of tens of
    megabytes, which is what makes a whole-test-set sweep practical.

    Step-count convention (mirrors the sampling loop, which breaks before the final time
    point): with `steps` requested time points, the model is evaluated `steps - 1` times,
    so pred_* arrays have `steps - 1` rows while state_* arrays have `steps` rows — the
    extra row is the final x_T that the output sequence is decoded from.

    Args:
        results: List of result dicts from sample_multiple_proteins_batched. Entries
            without a 'trajectory' key (e.g. failed batches) are skipped.
        output_path: Path of the .npz file to write.
        metadata: Optional dict of run-level scalars (checkpoint path, flow_temp,
            use_c_factor, seed, steps, T, ...) stored alongside the arrays.

    Returns:
        str: Path to the written NPZ file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    npz_data = {}
    stored_names = []
    skipped = 0

    for i, result in enumerate(results):
        trajectory = result.get('trajectory') if isinstance(result, dict) else None
        if trajectory is None:
            skipped += 1
            continue

        name = result.get('structure_name') or f"structure_{result.get('structure_idx', i)}"
        if name in stored_names:
            # Duplicate names would silently overwrite each other inside the archive.
            name = f"{name}#{i}"

        for field in COMPACT_STEP_FIELDS + COMPACT_POSITION_FIELDS:
            value = trajectory.get(field)
            if value is not None:
                npz_data[f'{name}|{field}'] = np.asarray(value)

        npz_data[f'{name}|length'] = np.int32(trajectory['length'])
        npz_data[f'{name}|structure_idx'] = np.int32(result.get('structure_idx', i))
        npz_data[f'{name}|accuracy'] = np.float32(
            result['accuracy'] if result.get('accuracy') is not None else np.nan
        )

        flank = result.get('flank_filtering')
        if flank is not None:
            npz_data[f'{name}|start_offset'] = np.int32(flank.get('start_offset', 0))
            npz_data[f'{name}|original_length'] = np.int32(flank.get('original_length', -1))
        else:
            npz_data[f'{name}|start_offset'] = np.int32(0)
            npz_data[f'{name}|original_length'] = np.int32(-1)

        stored_names.append(name)

    npz_data['structure_names'] = np.array(stored_names, dtype=object)
    npz_data['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    for key, value in (metadata or {}).items():
        npz_data[f'meta|{key}'] = np.array(value)

    np.savez_compressed(output_path, **npz_data)
    print(f"Compact trajectories saved for {len(stored_names)} structures "
          f"({skipped} skipped): {output_path}")
    return output_path


def load_compact_trajectory_npz(npz_path):
    """
    Load an NPZ written by save_compact_trajectory_npz.

    Returns:
        tuple: (structures, metadata) where structures maps protein name -> dict of arrays
        and metadata maps metadata key -> value.
    """
    data = np.load(npz_path, allow_pickle=True)

    structures = {}
    metadata = {}
    for key in data.files:
        if key.startswith('meta|'):
            metadata[key.split('|', 1)[1]] = data[key].item() if data[key].ndim == 0 else data[key]
        elif '|' in key:
            name, field = key.rsplit('|', 1)
            entry = structures.setdefault(name, {})
            entry[field] = data[key].item() if data[key].ndim == 0 else data[key]

    metadata['timestamp'] = str(data['timestamp']) if 'timestamp' in data.files else None
    return structures, metadata


def save_trajectory_analysis_npz(results, structure_names, output_dir, output_prefix):
    """
    Save trajectory analysis data in NPZ format for easy loading and analysis.

    This function captures:
    - Ground truth one-hot vectors for each position
    - Model predictions (softmax outputs) at each time step
    - Current denoised state at each time step
    - Time points

    Args:
        results: List of result dictionaries containing trajectory data
        structure_names: List of structure names/PDB IDs
        output_dir: Output directory
        output_prefix: Prefix for output filename

    Returns:
        str: Path to the generated NPZ file
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp and filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    npz_filename = f"{timestamp}_{output_prefix}_trajectory_analysis.npz"
    npz_filepath = os.path.join(output_dir, npz_filename)

    # Data containers
    npz_data = {}

    # Process each structure
    for i, (result, structure_name) in enumerate(zip(results, structure_names)):
        if 'trajectory_data' not in result or 'error' in result:
            print(f"Warning: No trajectory data for structure {structure_name}")
            continue

        trajectory = result['trajectory_data']
        pdb_id = structure_name if structure_name != 'unknown' else f"structure_{i}"

        # Get dimensions
        time_points = np.array(trajectory['time_points'])  # [T]
        num_time_points = len(time_points)
        num_positions = len(trajectory['positions'])
        K = 21  # Number of amino acid classes

        # Initialize arrays for this structure
        ground_truth_onehot = np.zeros((num_positions, K), dtype=np.float32)
        model_predictions = np.zeros((num_time_points, num_positions, K), dtype=np.float32)
        current_states = np.zeros((num_time_points, num_positions, K), dtype=np.float32)

        # Extract ground truth if available
        true_indices = result.get('true_indices', None)
        if true_indices is not None:
            for pos, true_idx in enumerate(true_indices):
                if 0 <= true_idx < K and pos < num_positions:
                    ground_truth_onehot[pos, true_idx] = 1.0

        # Process each position
        for pos in range(num_positions):
            if pos not in trajectory['positions']:
                continue

            pos_data = trajectory['positions'][pos]
            detailed_breakdowns = pos_data.get('detailed_breakdown', [])

            # Extract model predictions and current states from detailed breakdown
            for t_idx, breakdown in enumerate(detailed_breakdowns):
                if t_idx >= num_time_points:
                    break

                # Extract model predictions (softmax outputs)
                # The breakdown contains amino acid indexed data
                aa_names = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

                for aa_idx, aa_name in enumerate(aa_names):
                    if aa_name in breakdown:
                        aa_data = breakdown[aa_name]
                        # Extract model prediction (predicted_prob)
                        if 'predicted_prob' in aa_data:
                            model_predictions[t_idx, pos, aa_idx] = aa_data['predicted_prob']
                        # Extract current state (current_prob)
                        if 'current_prob' in aa_data:
                            current_states[t_idx, pos, aa_idx] = aa_data['current_prob']

        # Store data for this structure
        npz_data[f'{pdb_id}_time_points'] = time_points
        npz_data[f'{pdb_id}_ground_truth'] = ground_truth_onehot
        npz_data[f'{pdb_id}_model_predictions'] = model_predictions
        npz_data[f'{pdb_id}_current_states'] = current_states
        npz_data[f'{pdb_id}_structure_idx'] = result.get('structure_idx', i)
        npz_data[f'{pdb_id}_length'] = num_positions

    # Add metadata
    npz_data['structure_names'] = np.array(structure_names, dtype='<U50')
    npz_data['num_structures'] = len(structure_names)
    npz_data['aa_names'] = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'])
    npz_data['timestamp'] = timestamp

    # Save NPZ file
    np.savez_compressed(npz_filepath, **npz_data)

    print(f"Trajectory analysis data saved to: {npz_filepath}")
    return npz_filepath


def load_trajectory_analysis_npz(npz_filepath):
    """
    Load trajectory analysis data from NPZ file.

    Args:
        npz_filepath: Path to the NPZ file

    Returns:
        dict: Dictionary containing trajectory data for analysis
    """
    data = np.load(npz_filepath, allow_pickle=True)

    # Extract metadata
    structure_names = data['structure_names']
    num_structures = int(data['num_structures'])
    aa_names = data['aa_names']

    # Organize data by structure
    structures = {}
    for structure_name in structure_names:
        pdb_id = str(structure_name)

        structures[pdb_id] = {
            'time_points': data[f'{pdb_id}_time_points'],
            'ground_truth': data[f'{pdb_id}_ground_truth'],
            'model_predictions': data[f'{pdb_id}_model_predictions'],
            'current_states': data[f'{pdb_id}_current_states'],
            'structure_idx': int(data[f'{pdb_id}_structure_idx']),
            'length': int(data[f'{pdb_id}_length'])
        }

    return {
        'structures': structures,
        'metadata': {
            'structure_names': structure_names,
            'num_structures': num_structures,
            'aa_names': aa_names,
            'timestamp': str(data['timestamp'])
        }
    }


def calculate_accuracy_over_time(trajectory_data):
    """
    Calculate accuracy at each time point for all structures.

    Args:
        trajectory_data: Data loaded from load_trajectory_analysis_npz

    Returns:
        dict: Accuracy analysis results
    """
    results = {}

    for pdb_id, struct_data in trajectory_data['structures'].items():
        time_points = struct_data['time_points']
        ground_truth = struct_data['ground_truth']  # [N, K]
        model_predictions = struct_data['model_predictions']  # [T, N, K]
        current_states = struct_data['current_states']  # [T, N, K]

        # Get ground truth indices
        true_indices = np.argmax(ground_truth, axis=1)  # [N]

        # Calculate accuracy at each time point
        num_time_points = len(time_points)

        # Model prediction accuracy (argmax of softmax outputs)
        model_pred_accuracy = []
        for t in range(num_time_points):
            predicted_indices = np.argmax(model_predictions[t], axis=1)  # [N]
            accuracy = np.mean(predicted_indices == true_indices)
            model_pred_accuracy.append(accuracy)

        # Current state accuracy (argmax of current denoised state)
        current_state_accuracy = []
        for t in range(num_time_points):
            predicted_indices = np.argmax(current_states[t], axis=1)  # [N]
            accuracy = np.mean(predicted_indices == true_indices)
            current_state_accuracy.append(accuracy)

        results[pdb_id] = {
            'time_points': time_points,
            'model_prediction_accuracy': np.array(model_pred_accuracy),
            'current_state_accuracy': np.array(current_state_accuracy),
            'length': struct_data['length']
        }

    return results


def example_usage():
    """
    Example of how to use the trajectory analysis functions.
    """
    # Example loading and analysis
    npz_filepath = "trajectory_analysis.npz"

    # Load data
    trajectory_data = load_trajectory_analysis_npz(npz_filepath)

    # Calculate accuracy over time
    accuracy_results = calculate_accuracy_over_time(trajectory_data)

    # Print results
    for pdb_id, results in accuracy_results.items():
        print(f"\nStructure: {pdb_id}")
        print(f"Length: {results['length']}")
        print(f"Time points: {len(results['time_points'])}")
        print(f"Initial model accuracy: {results['model_prediction_accuracy'][0]:.4f}")
        print(f"Final model accuracy: {results['model_prediction_accuracy'][-1]:.4f}")
        print(f"Initial state accuracy: {results['current_state_accuracy'][0]:.4f}")
        print(f"Final state accuracy: {results['current_state_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    example_usage()

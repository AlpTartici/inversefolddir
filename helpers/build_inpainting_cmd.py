#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Helper script to build inpainting command from JSON config file.

This script reads a sampling config JSON file and constructs the appropriate
command-line arguments for training/inpainting.py.
"""

import json
import sys
import argparse

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def build_inpainting_args(config):
    """
    Convert config dictionary to inpainting.py command-line arguments.

    Args:
        config: Dictionary loaded from JSON config file

    Returns:
        List of command-line argument strings
    """
    args = []

    # Map config keys to inpainting.py arguments
    # Note: Some keys in config may not be directly supported by inpainting.py

    if 'model_path' in config:
        args.extend(['--model', str(config['model_path'])])

    if 'steps' in config:
        args.extend(['--steps', str(config['steps'])])

    if 'flow_temp' in config:
        args.extend(['--flow_temp', str(config['flow_temp'])])

    if 't_max' in config:
        args.extend(['--t_max', str(config['t_max'])])

    if 't_min' in config:
        args.extend(['--t_min', str(config['t_min'])])

    if 'dirichlet_concentration' in config:
        args.extend(['--dirichlet_concentration', str(config['dirichlet_concentration'])])

    if 'ensemble_size' in config:
        args.extend(['--ensemble_size', str(config['ensemble_size'])])

    if 'ensemble_consensus_strength' in config:
        args.extend(['--ensemble_consensus_strength', str(config['ensemble_consensus_strength'])])

    if 'ensemble_method' in config:
        args.extend(['--ensemble_method', str(config['ensemble_method'])])

    if 'structure_noise_mag_std' in config:
        args.extend(['--structure_noise_mag_std', str(config['structure_noise_mag_std'])])

    # Boolean flags
    if config.get('use_c_factor', False):
        args.append('--use_c_factor')

    if config.get('time_as_temperature', False):
        args.append('--time_as_temperature')

    if config.get('uncertainty_struct_noise_scaling', False):
        args.append('--uncertainty_struct_noise_scaling')

    if config.get('use_smoothed_targets', False):
        args.append('--use_smoothed_targets')

    if config.get('use_smoothed_labels', False):
        args.append('--use_smoothed_labels')

    if config.get('dssp_initialization', False):
        args.append('--dssp_initialization')

    if config.get('dssp_guidance', False):
        args.append('--dssp_guidance')

    if config.get('filter_out_missing_flanks', False):
        args.append('--filter_out_missing_flanks')

    # Batch size - note that inpainting.py uses --batch_size
    if 'batch_size' in config or 'batch_size ' in config:  # Handle the space in key
        batch_size = config.get('batch_size', config.get('batch_size ', None))
        if batch_size is not None:
            args.extend(['--batch_size', str(batch_size)])

    return args

def main():
    parser = argparse.ArgumentParser(
        description='Build inpainting command from JSON config'
    )
    parser.add_argument('config_file', help='Path to JSON config file')
    parser.add_argument('--output-format', choices=['args', 'shell'], default='args',
                       help='Output format: args (one per line) or shell (space-separated)')

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config_file)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Build arguments
    inpainting_args = build_inpainting_args(config)

    # Output
    if args.output_format == 'args':
        for arg in inpainting_args:
            print(arg)
    else:  # shell
        print(' '.join(inpainting_args))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Consolidate batch PDB sampling results into a single master CSV.

This script scans the batch_sampling directory structure:
    output/batch_sampling/
        config_86/
            7edq_A.pdb/
                protein_sampling_sequences.csv
            7ee8_A.pdb/
                protein_sampling_sequences.csv
        config_87/
            ...

And consolidates all individual protein_sampling_sequences.csv files into a
single master CSV with additional columns tracking config_name and pdb_basename.

Output CSV format:
    structure_idx, config_name, pdb_basename, structure_name, length,
    predicted_sequence, true_sequence, accuracy

Usage:
    python helpers/consolidate_batch_sequences.py \
        output/batch_sampling \
        --output output/batch_sampling/master_sequences.csv \
        --configs 86 87 88
"""

import argparse
import csv
import os
from pathlib import Path
import pandas as pd


def find_sequence_csvs(base_dir, configs):
    """
    Find all protein_sampling_sequences.csv files in the directory structure.

    Args:
        base_dir: Base directory containing config_* subdirectories
        configs: List of config numbers to process

    Returns:
        List of tuples: (config_name, pdb_basename, csv_path)
    """
    csv_files = []

    for config_num in configs:
        config_name = f"config_{config_num}"
        config_dir = Path(base_dir) / config_name

        if not config_dir.exists():
            print(f"Warning: Config directory not found: {config_dir}")
            continue

        # Iterate through PDB directories
        for pdb_dir in sorted(config_dir.iterdir()):
            if not pdb_dir.is_dir():
                continue

            pdb_basename = pdb_dir.name
            seq_file = pdb_dir / "protein_sampling_sequences.csv"

            if seq_file.exists():
                csv_files.append((config_name, pdb_basename, seq_file))
            else:
                try:
                    files = os.listdir(pdb_dir)
                    seq_file = None
                    for file in files:
                        if "sequences.csv" in file:
                            seq_file =  f"{pdb_dir}/{file}"
                            break
                    csv_files.append((config_name, pdb_basename, seq_file))
                except:
                    print(f"Warning: No sequences CSV in {pdb_dir}")

    return csv_files


def consolidate_sequences(base_dir, configs, output_csv, verbose=True):
    """
    Consolidate all individual sequence CSVs into a master CSV.

    Args:
        base_dir: Base directory containing config_* subdirectories
        configs: List of config numbers to process
        output_csv: Output path for master CSV
        verbose: Print progress messages
    """
    if verbose:
        print("="*60)
        print("CONSOLIDATING BATCH SEQUENCES")
        print("="*60)
        print(f"Base directory: {base_dir}")
        print(f"Configs: {configs}")
        print(f"Output CSV: {output_csv}")
        print("")

    # Find all CSV files
    csv_files = find_sequence_csvs(base_dir, configs)

    if not csv_files:
        print("ERROR: No sequence CSV files found!")
        print(f"Expected pattern: {base_dir}/config_*/{{pdb_basename}}/protein_sampling_sequences.csv")
        return False

    if verbose:
        print(f"Found {len(csv_files)} sequence CSV files")
        print("")

    # Process each CSV file
    all_data = []
    failed_files = []

    for idx, (config_name, pdb_basename, csv_path) in enumerate(csv_files, 1):
        try:
            if verbose:
                print(f"[{idx}/{len(csv_files)}] Processing: {config_name}/{pdb_basename}")

            # Read CSV
            df = pd.read_csv(csv_path)

            # Validate required columns
            required_cols = ['structure_name', 'length', 'predicted_sequence', 'true_sequence', 'accuracy']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"  WARNING: Missing columns {missing_cols} in {csv_path}")
                failed_files.append(csv_path)
                continue

            # Add tracking columns
            df['config_name'] = config_name
            df['pdb_basename'] = pdb_basename

            all_data.append(df)

            if verbose:
                print(f"  Loaded {len(df)} row(s)")

        except Exception as e:
            print(f"  ERROR processing {csv_path}: {e}")
            failed_files.append(csv_path)
            continue

    if not all_data:
        print("ERROR: No valid CSV files could be processed!")
        return False

    # Concatenate all dataframes
    if verbose:
        print("")
        print("Merging all dataframes...")

    master_df = pd.concat(all_data, ignore_index=True)

    # Reorder columns to match desired schema
    # Try to preserve all columns from original CSVs
    base_cols = ['config_name', 'pdb_basename', 'structure_name', 'length',
                 'predicted_sequence', 'true_sequence', 'accuracy']

    # Add any additional columns that might exist (exclude structure_idx — we regenerate it)
    other_cols = [col for col in master_df.columns if col not in base_cols and col != 'structure_idx']
    ordered_cols = base_cols + other_cols

    master_df = master_df[ordered_cols]

    # Add a fresh sequential structure_idx column at the beginning
    master_df.insert(0, 'structure_idx', range(len(master_df)))

    # Save to CSV
    master_df.to_csv(output_csv, index=False)

    # Print summary
    if verbose:
        print("")
        print("="*60)
        print("CONSOLIDATION COMPLETE")
        print("="*60)
        print(f"Successfully processed: {len(all_data)} files")
        print(f"Failed: {len(failed_files)} files")
        print(f"Total sequences: {len(master_df)}")
        print(f"Output: {output_csv}")

        # Print accuracy statistics
        if 'accuracy' in master_df.columns:
            print("")
            print("Accuracy Statistics:")
            print(f"  Mean: {master_df['accuracy'].mean():.2f}%")
            print(f"  Median: {master_df['accuracy'].median():.2f}%")
            print(f"  Min: {master_df['accuracy'].min():.2f}%")
            print(f"  Max: {master_df['accuracy'].max():.2f}%")

        # Print per-config statistics
        if 'config_name' in master_df.columns:
            print("")
            print("Per-Config Statistics:")
            for config in sorted(master_df['config_name'].unique()):
                config_df = master_df[master_df['config_name'] == config]
                mean_acc = config_df['accuracy'].mean()
                print(f"  {config}: {len(config_df)} sequences, mean accuracy: {mean_acc:.2f}%")

        print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate batch PDB sampling results into master CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('directory',
                       help='Base directory containing config_* subdirectories')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CSV file path')
    parser.add_argument('--configs', nargs='+', type=int, required=True,
                       help='Config numbers to process (e.g., --configs 86 87 88)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.directory):
        print(f"ERROR: Directory not found: {args.directory}")
        return 1

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Consolidate sequences
    success = consolidate_sequences(
        args.directory,
        args.configs,
        args.output,
        verbose=not args.quiet
    )

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

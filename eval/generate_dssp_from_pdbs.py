#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script to generate DSSP secondary structure assignments from ESMfold PDB files.

This script processes a directory of PDB files, computes DSSP secondary structure
for each chain, and saves the results in a structured JSON format for easy
downstream analysis.

Performance:
    - Supports parallel processing using multiprocessing for efficient large-scale analysis
    - Automatically uses all available CPU cores by default
    - Can be configured to use a specific number of workers or run sequentially
    - Uses chunked processing and imap for memory efficiency
    - Progress bar shows real-time processing status

Usage:
    python generate_dssp_from_pdbs.py --input_dir <pdb_directory> --output_file <output.json>

Output Format:
    {
        "chain_name": {
            "sequence": "MKTAYIAK...",
            "dssp_8": ["H", "H", "E", "-", ...],
            "dssp_3": ["H", "H", "E", "C", ...]
        },
        ...
    }

DSSP Secondary Structure Codes (8-state):
    H = Alpha helix
    B = Beta bridge
    E = Extended strand (beta sheet)
    G = 3-helix (310 helix)
    I = 5-helix (pi helix)
    T = Hydrogen bonded turn
    S = Bend
    - = Loop or irregular

DSSP Secondary Structure Codes (3-state):
    H = Helix (includes H, G, I from 8-state)
    E = Sheet (includes E, B from 8-state)
    C = Coil (includes all others: T, S, -, and spaces)
"""

import argparse
import json
import os
import sys
import traceback
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

from Bio.PDB import DSSP, PDBParser
from tqdm import tqdm

# Suppress BioPython warnings about DSSP and mmCIF validation
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
warnings.filterwarnings("ignore", message=".*mmCIF.*")
warnings.filterwarnings("ignore", message=".*Unknown or untrusted program.*")


def extract_chain_name_from_filename(filename):
    """
    Extract chain name from ESMfold PDB filename.

    Expected format: "pred_for_1a2p.A.pdb" -> chain_name: "1a2p.A"

    Args:
        filename (str): The PDB filename

    Returns:
        str: The extracted chain name, or None if format doesn't match
    """
    # Remove .pdb extension
    name_without_ext = filename.replace('.pdb', '')

    # Remove "pred_for_" prefix if present
    if name_without_ext.startswith('pred_for_'):
        chain_name = name_without_ext.replace('pred_for_', '')
        return chain_name
    elif "chain" in name_without_ext:
        pdb_id, chain_letter = name_without_ext.split('_chain')
        chain_name = f"{pdb_id}.{chain_letter}"
        return chain_name

    # If no prefix, just return the name without extension
    return name_without_ext


def convert_dssp_8_to_3(dssp_8_array):
    """
    Convert 8-state DSSP to 3-state DSSP.

    Mapping:
        H, G, I -> H (Helix)
        E, B -> E (Sheet)
        T, S, -, ' ' (space) -> C (Coil)
        X -> X (Unknown, preserved)

    Args:
        dssp_8_array (list): List of 8-state DSSP assignments

    Returns:
        list: List of 3-state DSSP assignments
    """
    # Define the mapping from 8-state to 3-state
    mapping = {
        'H': 'H',  # Alpha helix -> Helix
        'G': 'H',  # 3-10 helix -> Helix
        'I': 'H',  # Pi helix -> Helix (standard DSSP notation)
        'P': 'H',  # Pi helix -> Helix (BioPython uses 'P' instead of 'I')
        'E': 'E',  # Extended strand -> Sheet
        'B': 'E',  # Beta bridge -> Sheet
        'T': 'C',  # Turn -> Coil
        'S': 'C',  # Bend -> Coil
        '-': 'C',  # Loop -> Coil
        ' ': 'C',  # Space -> Coil
        'X': 'X',  # Unknown -> Unknown (preserved)
    }

    # Convert each residue
    dssp_3_array = [mapping.get(ss, 'C') for ss in dssp_8_array]

    return dssp_3_array


def compute_dssp_for_pdb(pdb_file, model_index=0, chain_id=None):
    """
    Compute DSSP secondary structure and extract sequence from a PDB file.

    Args:
        pdb_file (str): Path to the PDB file
        model_index (int): Model index to use (default: 0)
        chain_id (str): Specific chain ID to process (default: None, uses first chain)

    Returns:
        tuple: (sequence, dssp_8_array, dssp_3_array)
            sequence (str): Amino acid sequence
            dssp_8_array (list): 8-state secondary structure assignments for each residue
            dssp_3_array (list): 3-state secondary structure assignments for each residue

    Raises:
        Exception: If no valid chain is found or DSSP computation fails
    """
    # Parse the PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_file)
    model = structure[model_index]

    # Run DSSP with explicit PDB format to avoid mmCIF issues
    try:
        dssp = DSSP(model, pdb_file, file_type="PDB")
    except TypeError:
        # Fallback for older BioPython versions that don't support file_type parameter
        dssp = DSSP(model, pdb_file)

    # Get the target chain (first chain if chain_id not specified)
    target_chain = None
    actual_chain_id = None

    for chain in model:
        if chain_id is None or chain.id == chain_id:
            target_chain = chain
            actual_chain_id = chain.id
            break

    if target_chain is None:
        raise Exception(f"No target chain found in {pdb_file}")

    # Get all protein residues (excluding heteroatoms)
    # Residues with ID[0] == ' ' are standard amino acids
    protein_residues = [
        r for r in target_chain.get_residues() if r.get_id()[0] == " "
    ]

    if not protein_residues:
        raise Exception(f"No protein residues found in chain {actual_chain_id}")

    # Extract sequence and DSSP assignments
    sequence = []
    dssp_array = []

    for residue in protein_residues:
        # Create DSSP key (chain_id, residue_id)
        dssp_key = (actual_chain_id, residue.get_id())

        if dssp_key in dssp:
            # DSSP data available
            dssp_data = dssp[dssp_key]
            amino_acid = dssp_data[1]  # Amino acid one-letter code
            secondary_structure = dssp_data[2]  # Secondary structure assignment

            sequence.append(amino_acid)
            dssp_array.append(secondary_structure)
        else:
            # DSSP couldn't compute for this residue (e.g., incomplete coordinates)
            # Use 'X' for unknown amino acid and secondary structure
            sequence.append('X')
            dssp_array.append('X')

    # Convert sequence list to string
    sequence_str = ''.join(sequence)

    # Convert 8-state DSSP to 3-state DSSP
    dssp_3_array = convert_dssp_8_to_3(dssp_array)

    return sequence_str, dssp_array, dssp_3_array


def process_single_pdb_file(pdb_file_path):
    """
    Process a single PDB file and return results.
    This function is designed to be called by multiprocessing workers.

    Args:
        pdb_file_path (Path): Path object for the PDB file

    Returns:
        tuple: (chain_name, result_dict) where result_dict contains:
            - status: 'success' or 'error'
            - sequence: amino acid sequence (if successful)
            - dssp_8: 8-state DSSP array (if successful)
            - dssp_3: 3-state DSSP array (if successful)
            - error: error message (if failed)
            - traceback_str: full traceback (if failed)
    """
    try:
        # Extract chain name from filename
        chain_name = extract_chain_name_from_filename(pdb_file_path.name)

        if chain_name is None:
            return None, {
                'status': 'error',
                'filename': pdb_file_path.name,
                'error': 'Invalid filename format',
                'traceback_str': ''
            }

        # Compute DSSP (both 8-state and 3-state)
        sequence, dssp_8_array, dssp_3_array = compute_dssp_for_pdb(str(pdb_file_path))

        # Return successful result
        return chain_name, {
            'status': 'success',
            'sequence': sequence,
            'dssp_8': dssp_8_array,
            'dssp_3': dssp_3_array
        }

    except Exception as e:
        # Return error result
        return None, {
            'status': 'error',
            'filename': pdb_file_path.name,
            'error': str(e),
            'traceback_str': traceback.format_exc()
        }


def process_pdb_directory(input_dir, output_file, num_workers=None):
    """
    Process all PDB files in a directory and generate DSSP data.

    Args:
        input_dir (str): Path to directory containing PDB files
        output_file (str): Path to output JSON file
        num_workers (int): Number of parallel workers. If None, uses cpu_count().
                          If 1, processes sequentially (no multiprocessing).

    Returns:
        dict: Dictionary with chain names as keys and sequence/dssp data as values
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # Find all PDB files in the directory
    pdb_files = list(input_path.glob("*.pdb"))

    if not pdb_files:
        print(f"Warning: No PDB files found in {input_dir}")
        return {}

    print(f"Found {len(pdb_files)} PDB files to process")

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    elif num_workers <= 0:
        # If 0 or negative, default to cpu_count()
        num_workers = cpu_count()

    # Dictionary to store results
    results = {}

    # Statistics tracking
    successful = 0
    failed = 0
    failed_entries = []

    # Process files based on num_workers setting
    if num_workers == 1:
        # Sequential processing (no multiprocessing overhead)
        print("Processing files sequentially (num_workers=1)")
        processing_results = []
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            processing_results.append(process_single_pdb_file(pdb_file))
    else:
        # Parallel processing with multiprocessing
        print(f"Processing files in parallel using {num_workers} workers")
        # Use chunksize for better performance with large datasets
        # Chunksize of 10-50 works well for most cases
        chunksize = max(1, len(pdb_files) // (num_workers * 4))

        with Pool(processes=num_workers) as pool:
            # Use imap for memory efficiency with progress bar
            processing_results = list(
                tqdm(
                    pool.imap(process_single_pdb_file, pdb_files, chunksize=chunksize),
                    total=len(pdb_files),
                    desc="Processing PDB files"
                )
            )

    # Aggregate results from processing
    for chain_name, result in processing_results:
        if result['status'] == 'success':
            # Store successful result
            results[chain_name] = {
                "sequence": result['sequence'],
                "dssp_8": result['dssp_8'],
                "dssp_3": result['dssp_3']
            }
            successful += 1
        else:
            # Track failed entry
            filename = result.get('filename', 'unknown')
            error = result.get('error', 'Unknown error')
            failed += 1
            failed_entries.append((filename, error))

            # Print error message for visibility
            if result.get('traceback_str'):
                print(f"\nError processing {filename}: {error}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Processing Summary:")
    print(f"  Total PDB files: {len(pdb_files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 60)

    # Save failed entries log if any failures occurred
    if failed_entries:
        log_file = output_file.replace('.json', '_errors.log')
        with open(log_file, 'w') as f:
            f.write("Failed PDB Processing Log\n")
            f.write("=" * 60 + "\n\n")
            for filename, error in failed_entries:
                f.write(f"File: {filename}\n")
                f.write(f"Error: {error}\n")
                f.write("-" * 60 + "\n")
        print(f"\nError log saved to: {log_file}")

    return results


def save_results(results, output_file, format='json'):
    """
    Save results to file in specified format.

    Args:
        results (dict): Dictionary of chain names and their sequence/dssp data
        output_file (str): Path to output file
        format (str): Output format ('json' or 'pickle')
    """
    output_path = Path(output_file)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        print(f"Format: JSON (human-readable)")
    elif format == 'pickle':
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {output_file}")
        print(f"Format: Pickle (Python-optimized)")
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Print sample of results
    if results:
        sample_chain = list(results.keys())[0]
        sample_data = results[sample_chain]
        print(f"\nSample entry (chain: {sample_chain}):")
        print(f"  Sequence length: {len(sample_data['sequence'])}")
        print(f"  DSSP-8 length: {len(sample_data['dssp_8'])}")
        print(f"  DSSP-3 length: {len(sample_data['dssp_3'])}")
        print(f"  First 50 residues:")
        print(f"    Sequence: {sample_data['sequence'][:50]}")
        print(f"    DSSP-8:   {''.join(sample_data['dssp_8'][:50])}")
        print(f"    DSSP-3:   {''.join(sample_data['dssp_3'][:50])}")


def main():
    """
    Main function to orchestrate DSSP generation from PDB directory.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate DSSP secondary structure assignments from PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDB files and save as JSON (default: uses all CPU cores)
  python generate_dssp_from_pdbs.py --input_dir ./esmfold_predictions --output_file ./dssp_results.json

  # Process PDB files with 8 parallel workers
  python generate_dssp_from_pdbs.py --input_dir ./predictions --output_file ./dssp.json --num_workers 8

  # Process sequentially (no parallelization)
  python generate_dssp_from_pdbs.py --input_dir ./predictions --output_file ./dssp.json --num_workers 1

  # Process PDB files and save as pickle (more efficient for large datasets)
  python generate_dssp_from_pdbs.py --input_dir ./predictions --output_file ./dssp.pkl --format pickle
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to directory containing PDB files'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output file (will create parent directories if needed)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'pickle'],
        default='json',
        help='Output format: json (human-readable) or pickle (more efficient for Python)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers for processing. Default: cpu_count(). Use 1 for sequential processing.'
    )

    args = parser.parse_args()

    # Determine number of workers for display
    display_workers = args.num_workers if args.num_workers is not None else cpu_count()

    print("DSSP Generator for ESMfold PDB Files")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Output format: {args.format}")
    print(f"Number of workers: {display_workers}" + (" (sequential)" if display_workers == 1 else " (parallel)"))
    print("=" * 60 + "\n")

    try:
        # Process all PDB files in the directory
        results = process_pdb_directory(args.input_dir, args.output_file, num_workers=args.num_workers)

        if not results:
            print("No results to save. Exiting.")
            return 1

        # Save results to file
        save_results(results, args.output_file, format=args.format)

        print("\nProcessing completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

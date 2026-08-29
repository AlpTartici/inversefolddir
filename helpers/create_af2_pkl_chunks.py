# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
create_af2_pkl_chunks.py

Create fast-access pickle files from AlphaFold2 CIF structures for efficient batch loading.
Each protein appears in multiple pickle files for random sampling without replacement issues.

Usage:
    python create_af2_pkl_chunks.py --input_dir /path/to/af2_cifs --cluster_dir /path/to/af_clusters --output_dir /path/to/pkl_chunks
"""
import argparse
import json
import os
import pickle
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cif_parser import parse_cif_backbone_auto
from data.graph_builder import normalize_uncertainty


def extract_uniprot_id(cif_filename: str) -> str:
    """Extract UniProt ID from AF2 filename: AF-{uniprot_id}-F1-model_v4.cif"""
    if cif_filename.startswith("AF-") and cif_filename.endswith("-F1-model_v4.cif"):
        return cif_filename[3:-16]  # Remove 'AF-' prefix and '-F1-model_v4.cif' suffix
    else:
        # Fallback: use filename without extension
        return Path(cif_filename).stem


def process_single_cif(cif_path: Path, max_length: int = 600) -> Optional[Dict]:
    """
    Process a single AF2 CIF file into optimized format.

    Args:
        cif_path: Path to CIF file
        max_length: Maximum protein length to include

    Returns:
        Dictionary with protein data or None if processing failed/filtered
    """
    try:
        # Parse CIF file
        coords, plddt_scores, residue_types, source = parse_cif_backbone_auto(
            str(cif_path)
        )

        if coords is None or len(coords) == 0:
            return None

        # Apply length filter
        if len(coords) > max_length:
            return None

        # Convert to numpy for better pickle efficiency
        if hasattr(coords, "numpy"):
            coords_np = coords.numpy()
        else:
            coords_np = np.array(coords)

        if hasattr(plddt_scores, "numpy"):
            plddt_np = plddt_scores.numpy()
        else:
            plddt_np = np.array(plddt_scores)

        # Create sequence string from residue types
        three_to_one = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }
        sequence = "".join([three_to_one.get(rt, "X") for rt in residue_types])

        # Extract UniProt ID
        uniprot_id = extract_uniprot_id(cif_path.name)

        # Create protein entry in CATH format
        # coords_np shape is [L, 4, 3] where the 4 atoms are N, CA, C, O
        coords_dict = {
            "N": coords_np[:, 0, :].astype(np.float32),  # [L, 3]
            "CA": coords_np[:, 1, :].astype(np.float32),  # [L, 3]
            "C": coords_np[:, 2, :].astype(np.float32),  # [L, 3]
            "O": coords_np[:, 3, :].astype(np.float32),  # [L, 3]
        }

        protein_data = {
            "name": uniprot_id,
            "seq": sequence,  # Use 'seq' to match CATH format
            "coords": coords_dict,  # Dictionary format like CATH
            "plddt": plddt_np.astype(np.float32),  # pLDDT scores instead of B-factors
            "source": "alphafold2",
            "num_chains": 1,
            "CATH": ["unknown"],  # Placeholder for CATH classification
        }

        return protein_data

    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        return None


def process_cif_batch(
    cif_paths: List[Path], max_length: int, worker_id: int
) -> Tuple[List[Dict], List[str]]:
    """Process a batch of CIF files in a single worker."""
    processed = []
    failed = []

    for cif_path in tqdm(cif_paths, desc=f"Worker {worker_id}", leave=False):
        result = process_single_cif(cif_path, max_length)
        if result is not None:
            processed.append(result)
        else:
            failed.append(str(cif_path))

    return processed, failed


def create_pkl_chunks(
    protein_data: List[Dict],
    output_dir: Path,
    chunk_size: int = 1100,
    coverage_per_protein: int = 10,
    random_seed: int = 42,
) -> Dict:
    """
    Create pickle chunks with each protein appearing in multiple files.

    Args:
        protein_data: List of processed protein dictionaries
        output_dir: Directory to save pickle files
        chunk_size: Number of proteins per pickle file
        coverage_per_protein: How many times each protein appears across files
        random_seed: Random seed for reproducible assignment

    Returns:
        Statistics dictionary
    """
    print(f"Creating pickle chunks with {len(protein_data)} proteins...")
    print(f"Target: {coverage_per_protein}x coverage, {chunk_size} proteins per chunk")

    # Calculate number of chunks needed
    total_appearances = len(protein_data) * coverage_per_protein
    num_chunks = (total_appearances + chunk_size - 1) // chunk_size

    print(f"Will create {num_chunks} pickle files")

    # Create assignment list: each protein appears coverage_per_protein times
    protein_assignments = []
    for i, protein in enumerate(protein_data):
        for _ in range(coverage_per_protein):
            protein_assignments.append((i, protein))

    # Shuffle assignments
    random.seed(random_seed)
    random.shuffle(protein_assignments)

    # Create chunks
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_stats = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(protein_assignments))

        chunk_proteins = []
        for i in range(start_idx, end_idx):
            _, protein = protein_assignments[i]
            chunk_proteins.append(protein)

        # Save chunk
        chunk_path = output_dir / f"af2_chunk_{chunk_idx:06d}.pkl"
        with open(chunk_path, "wb") as f:
            pickle.dump(chunk_proteins, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Collect stats
        chunk_stats.append(
            {
                "chunk_id": chunk_idx,
                "path": str(chunk_path),
                "num_proteins": len(chunk_proteins),
                "avg_length": np.mean([len(p["seq"]) for p in chunk_proteins]),
                "size_mb": chunk_path.stat().st_size / (1024 * 1024),
            }
        )

        print(
            f"Created chunk {chunk_idx:6d}: {len(chunk_proteins):4d} proteins, "
            f"{chunk_stats[-1]['size_mb']:.1f}MB"
        )

    # Create metadata
    metadata = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "coverage_per_protein": coverage_per_protein,
        "total_proteins": len(protein_data),
        "total_appearances": total_appearances,
        "random_seed": random_seed,
        "chunk_stats": chunk_stats,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_path}")
    print(f"Average chunk size: {np.mean([s['size_mb'] for s in chunk_stats]):.1f}MB")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create AF2 pickle chunks for fast batch loading"
    )

    # Input/Output
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing AF2 CIF files",
    )
    parser.add_argument(
        "--cluster_dir",
        type=str,
        required=True,
        help="Directory containing cluster metadata (flat_members.npy, etc.)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save pickle chunks"
    )

    # Processing parameters
    parser.add_argument(
        "--max_length",
        type=int,
        default=600,
        help="Maximum protein length (default: 600)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1100,
        help="Number of proteins per pickle file (default: 1100)",
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=10,
        help="How many times each protein appears (default: 10)",
    )

    # Performance
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of files to process per worker batch (default: 1000)",
    )

    # Other
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible chunk assignment",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just count files and estimate output size",
    )

    args = parser.parse_args()

    # Setup
    input_dir = Path(args.input_dir)
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1

    if not cluster_dir.exists():
        print(f"Error: Cluster directory {cluster_dir} does not exist")
        return 1

    # Load protein names from flat_members.npy (same as AF2 sampler uses)
    flat_members_path = cluster_dir / "flat_members.npy"
    if not flat_members_path.exists():
        print(f"Error: flat_members.npy not found at {flat_members_path}")
        return 1

    print(f"Loading protein names from {flat_members_path}...")
    try:
        flat_members = np.load(flat_members_path, mmap_mode="r")
        protein_names = [str(name) for name in flat_members]  # Convert to strings
        print(f"Loaded {len(protein_names):,} protein names")
    except Exception as e:
        print(f"Error loading flat_members.npy: {e}")
        return 1

    # Construct CIF file paths using the same pattern as AF2Dataset
    print(f"Constructing CIF file paths...")
    cif_files = []
    for uniprot_id in protein_names:
        # Same pattern as AF2Dataset._construct_cif_path()
        filename = f"AF-{uniprot_id}-F1-model_v4.cif"
        cif_path = input_dir / filename
        cif_files.append(cif_path)

    print(f"Constructed {len(cif_files):,} CIF file paths")

    if len(cif_files) == 0:
        print("No CIF files to process!")
        return 1

    if args.dry_run:
        # Estimate output
        total_appearances = len(cif_files) * args.coverage
        num_chunks = (total_appearances + args.chunk_size - 1) // args.chunk_size
        est_size_gb = num_chunks * 50 / 1024  # Rough estimate: 50MB per chunk

        print(f"\nDry run estimates:")
        print(f"  Input files: {len(cif_files):,}")
        print(
            f"  After filtering (max {args.max_length} residues): ~{len(cif_files)*0.8:.0f}"
        )
        print(f"  Coverage: {args.coverage}x per protein")
        print(f"  Chunk size: {args.chunk_size} proteins")
        print(f"  Estimated chunks: {num_chunks}")
        print(f"  Estimated total size: {est_size_gb:.1f} GB")
        return 0

    # Process CIF files in parallel
    if args.num_workers is None:
        args.num_workers = min(os.cpu_count(), 8)  # Cap at 8 to avoid memory issues

    print(f"Processing {len(cif_files)} files with {args.num_workers} workers...")

    # Split files into batches for workers
    batch_size = args.batch_size
    file_batches = [
        cif_files[i : i + batch_size] for i in range(0, len(cif_files), batch_size)
    ]

    all_protein_data = []
    all_failed = []

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all jobs
        futures = []
        for i, batch in enumerate(file_batches):
            future = executor.submit(process_cif_batch, batch, args.max_length, i)
            futures.append(future)

        # Collect results
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing batches"
        ):
            try:
                protein_batch, failed_batch = future.result()
                all_protein_data.extend(protein_batch)
                all_failed.extend(failed_batch)
            except Exception as e:
                print(f"Batch processing error: {e}")

    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {len(all_protein_data)}")
    print(f"  Failed/filtered: {len(all_failed)}")

    if len(all_protein_data) == 0:
        print("No proteins successfully processed!")
        return 1

    # Save failed files list
    if all_failed:
        failed_path = output_dir / "failed_files.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(failed_path, "w") as f:
            f.write("\n".join(all_failed))
        print(f"Saved {len(all_failed)} failed files to {failed_path}")

    # Create pickle chunks
    metadata = create_pkl_chunks(
        all_protein_data,
        output_dir,
        chunk_size=args.chunk_size,
        coverage_per_protein=args.coverage,
        random_seed=args.random_seed,
    )

    print(f"\n[ok] Successfully created {metadata['num_chunks']} pickle chunks!")
    print(f"Output directory: {output_dir}")
    print(f"Total proteins: {metadata['total_proteins']:,}")
    print(
        f"Average chunk size: {np.mean([s['size_mb'] for s in metadata['chunk_stats']]):.1f} MB"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

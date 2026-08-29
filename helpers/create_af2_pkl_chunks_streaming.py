#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
create_af2_pkl_chunks_streaming.py

Memory-efficient version that streams processed proteins directly to temporary files,
then creates chunks without loading everything into memory at once.

Usage:
    python create_af2_pkl_chunks_streaming.py --input_dir /path/to/af2_cifs --cluster_dir /path/to/af_clusters --output_dir /path/to/pkl_chunks
"""
import argparse
import json
import os
import pickle
import random
import shutil
import signal
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cif_parser import parse_cif_backbone_auto

# (normalize_uncertainty import removed; not used here)


@contextmanager
def timeout(duration):
    """Context manager for timing out operations."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def extract_uniprot_id(cif_filename: str) -> str:
    """Extract UniProt ID from AF2 filename: AF-{uniprot_id}-F1-model_v4.cif"""
    if cif_filename.startswith("AF-") and cif_filename.endswith("-F1-model_v4.cif"):
        return cif_filename[3:-16]  # Remove 'AF-' prefix and '-F1-model_v4.cif' suffix
    else:
        # Fallback: use filename without extension
        return Path(cif_filename).stem


def process_single_cif(
    cif_path: Path,
    max_length: int = 600,
    min_plddt: float = 70.0,
    parse_timeout: int = 60,
) -> Optional[Dict]:
    """
    Process a single AF2 CIF file into optimized format.

    Args:
        cif_path: Path to CIF file
        max_length: Maximum protein length to include
        min_plddt: Minimum average pLDDT score to include (default: 70.0)
        parse_timeout: Maximum seconds to spend parsing each file (default: 60)

    Returns:
        Dictionary with protein data or None if processing failed/filtered
    """
    try:
        # Parse CIF file with timeout protection
        with timeout(parse_timeout):
            coords, plddt_scores, residue_types, _ = parse_cif_backbone_auto(
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

        # Apply pLDDT quality filter - check average confidence
        avg_plddt = float(np.mean(plddt_np))
        if avg_plddt < min_plddt:
            return None

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
            "seq": sequence,
            "coords": coords_dict,
            "plddt": plddt_np.astype(np.float32),
            "source": "alphafold2",
            "num_chains": 1,
            "name": uniprot_id,
            "CATH": ["unknown"],
        }

        return uniprot_id, protein_data

    except (FileNotFoundError, OSError, ValueError, TimeoutError):
        # Silent failure - let the worker handle logging to avoid stdout spam
        # TimeoutError will catch files that take too long to parse
        return None


def process_and_save_batch(
    cif_paths: List[Path],
    max_length: int,
    worker_id: int,
    temp_dir: Path,
    min_plddt: float = 70.0,
    disable_tqdm: bool = True,
    fail_buffer_flush: int = 200,
    parse_timeout: int = 60,
) -> Tuple[str, int, int, str]:
    """Process a batch of CIF files and save directly to temporary file.

    Returns: (temp_file_path, num_processed, failed_count, failed_log_path)

    Optimizations vs original:
      * Optional suppression of per-worker tqdm to reduce lock contention / overhead.
      * Write failures to a per-worker text file instead of returning large lists over IPC.
      * Buffered failure writes to limit small I/O ops.
    """
    processed_count = 0
    failed_count = 0

    # Create temporary file for this batch
    temp_file = temp_dir / f"batch_{worker_id:06d}.pkl"
    failed_log_path = temp_dir / f"failed_{worker_id:06d}.txt"

    processed_proteins: Dict[str, Dict] = {}
    failure_buffer: List[str] = []

    iterator = (
        cif_paths
        if disable_tqdm
        else tqdm(cif_paths, desc=f"Worker {worker_id}", leave=False)
    )
    for cif_path in iterator:
        result = process_single_cif(cif_path, max_length, min_plddt, parse_timeout)
        if result is not None:
            protein_id, protein_data = result
            processed_proteins[protein_id] = protein_data
            processed_count += 1
        else:
            failed_count += 1
            failure_buffer.append(str(cif_path))

        # Flush failure buffer if large
        if len(failure_buffer) >= fail_buffer_flush:
            with open(failed_log_path, "a", encoding="utf-8") as fl:
                fl.write("\n".join(failure_buffer) + "\n")
            failure_buffer.clear()

    # Final failure buffer flush
    if failure_buffer:
        with open(failed_log_path, "a", encoding="utf-8") as fl:
            fl.write("\n".join(failure_buffer) + "\n")

    # Save batch to temporary file
    if processed_proteins:
        with open(temp_file, "wb") as f:
            pickle.dump(processed_proteins, f, protocol=pickle.HIGHEST_PROTOCOL)

    return (
        str(temp_file),
        processed_count,
        failed_count,
        str(failed_log_path if failed_log_path.exists() else ""),
    )


def stream_proteins_from_temp_files(
    temp_files: List[str],
) -> Iterator[Tuple[str, Dict]]:
    """Stream proteins from temporary files one at a time as (protein_id, protein_data) tuples."""
    for temp_file in temp_files:
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            try:
                with open(temp_file, "rb") as f:
                    proteins_dict = pickle.load(f)
                for protein_id, protein_data in proteins_dict.items():
                    yield protein_id, protein_data
            except (pickle.PickleError, OSError) as e:
                print(f"Error reading temp file {temp_file}: {e}")


def create_pkl_chunks_streaming(
    temp_files: List[str],
    output_dir: Path,
    total_proteins: int,
    chunk_size: int = 1100,
    coverage_per_protein: int = 10,
    random_seed: int = 42,
) -> Dict:
    """
    Create pickle chunks by streaming from temporary files to avoid memory overload.

    Args:
        temp_files: List of temporary files containing processed proteins
        output_dir: Directory to save pickle files
        total_proteins: Total number of proteins processed
        chunk_size: Number of proteins per pickle file
        coverage_per_protein: How many times each protein appears across files
        random_seed: Random seed for reproducible assignment

    Returns:
        Statistics dictionary
    """
    print(f"Creating pickle chunks with {total_proteins} proteins (streaming mode)...")
    print(f"Target: {coverage_per_protein}x coverage, {chunk_size} proteins per chunk")

    # Calculate number of chunks needed
    total_appearances = total_proteins * coverage_per_protein
    num_chunks = (total_appearances + chunk_size - 1) // chunk_size

    print(f"Will create {num_chunks} pickle files")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create index mapping protein_id -> chunk assignments
    print("Creating protein-to-chunk assignments...")
    random.seed(random_seed)

    protein_to_chunks = {}  # protein_id -> list of chunk_ids it should appear in

    # For each protein, randomly assign it to 'coverage_per_protein' chunks
    for protein_id in range(total_proteins):
        # Randomly choose which chunks this protein will appear in
        assigned_chunks = random.sample(
            range(num_chunks), min(coverage_per_protein, num_chunks)
        )
        protein_to_chunks[protein_id] = assigned_chunks

    # Step 2: Initialize chunk files
    print("Initializing chunk files...")
    chunk_files = {}
    chunk_counts = [0] * num_chunks

    for chunk_idx in range(num_chunks):
        chunk_path = output_dir / f"af2_chunk_{chunk_idx:06d}.pkl"
        chunk_files[chunk_idx] = {"path": chunk_path, "proteins": {}}  # Change to dict

    # Step 3: Stream proteins and assign to chunks
    print("Streaming proteins and building chunks...")
    protein_id = 0

    for protein_key, protein_data in tqdm(
        stream_proteins_from_temp_files(temp_files),
        total=total_proteins,
        desc="Building chunks",
    ):
        # Get chunk assignments for this protein
        assigned_chunks = protein_to_chunks.get(protein_id, [])

        # Add protein to each assigned chunk (using protein_key as the dict key)
        for chunk_idx in assigned_chunks:
            chunk_files[chunk_idx]["proteins"][protein_key] = protein_data
            chunk_counts[chunk_idx] += 1

        protein_id += 1

        # Periodically flush chunks that are getting full to manage memory
        if protein_id % 10000 == 0:  # Every 10k proteins
            for chunk_idx in range(num_chunks):
                if len(chunk_files[chunk_idx]["proteins"]) >= chunk_size:
                    # Save and clear this chunk
                    chunk_path = chunk_files[chunk_idx]["path"]
                    proteins = chunk_files[chunk_idx]["proteins"]

                    with open(chunk_path, "wb") as f:
                        pickle.dump(proteins, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # Clear from memory
                    chunk_files[chunk_idx]["proteins"] = {}
                    print(f"Saved chunk {chunk_idx} ({len(proteins)} proteins)")

    # Step 4: Save remaining chunks
    print("Saving remaining chunks...")
    chunk_stats = []

    for chunk_idx in range(num_chunks):
        chunk_path = chunk_files[chunk_idx]["path"]
        proteins = chunk_files[chunk_idx]["proteins"]

        if proteins:  # Only save if there are proteins left
            with open(chunk_path, "wb") as f:
                pickle.dump(proteins, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Collect stats
        if chunk_path.exists():
            chunk_stats.append(
                {
                    "chunk_id": chunk_idx,
                    "path": str(chunk_path),
                    "num_proteins": chunk_counts[chunk_idx],
                    "size_mb": chunk_path.stat().st_size / (1024 * 1024),
                }
            )
            print(
                f"Final chunk {chunk_idx:6d}: {chunk_counts[chunk_idx]:4d} proteins, "
                f"{chunk_stats[-1]['size_mb']:.1f}MB"
            )

    # Create metadata
    metadata = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "coverage_per_protein": coverage_per_protein,
        "total_proteins": total_proteins,
        "total_appearances": sum(chunk_counts),
        "random_seed": random_seed,
        "chunk_stats": chunk_stats,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_path}")
    print(f"Average chunk size: {np.mean([s['size_mb'] for s in chunk_stats]):.1f}MB")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create AF2 pickle chunks for fast batch loading (memory-efficient)"
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
        help="Directory containing cluster metadata",
    )
    parser.add_argument(
        "--flat_members_file",
        type=str,
        required=True,
        help="Name of the flat members file (e.g., 'flat_members.npy', 'flat_members_thresholded.npy')",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save pickle chunks"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory for intermediate files (default: system temp)",
    )

    # Processing parameters
    parser.add_argument(
        "--max_length",
        type=int,
        default=600,
        help="Maximum protein length (default: 600)",
    )
    parser.add_argument(
        "--min_plddt",
        type=float,
        default=70.0,
        help="Minimum average pLDDT score to include (default: 70.0)",
    )
    parser.add_argument(
        "--parse_timeout",
        type=int,
        default=60,
        help="Maximum seconds to spend parsing each CIF file (default: 60)",
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
        "--enable_worker_tqdm",
        action="store_true",
        help="Show per-worker progress bars (disabled by default for speed)",
    )
    parser.add_argument(
        "--fail_buffer_flush",
        type=int,
        default=200,
        help="Flush failure paths to disk every N failures (default: 200)",
    )
    parser.add_argument(
        "--no_global_tqdm",
        action="store_true",
        help="Disable the global tqdm batch progress bar and use periodic log lines instead",
    )
    parser.add_argument(
        "--progress_log_every",
        type=int,
        default=120,
        help="Seconds between periodic progress log lines when --no_global_tqdm is set (default: 120)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just count files and estimate output size",
    )
    parser.add_argument(
        "--keep_temp_files",
        action="store_true",
        help="Keep temporary files after processing (for debugging)",
    )

    args = parser.parse_args()

    # Setup
    input_dir = Path(args.input_dir)
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)

    print(f"Input directory: {input_dir}", flush=True)
    print(f"Cluster directory: {cluster_dir}", flush=True)

    # Load protein names from the specified flat members file
    flat_members_path = cluster_dir / args.flat_members_file
    print(f"Loading from: {flat_members_path}", flush=True)

    print(f"Loading protein names from {flat_members_path}...", flush=True)
    try:
        flat_members = np.load(flat_members_path, mmap_mode="r")
        protein_names = [str(name) for name in flat_members]  # Convert to strings
        print(f"Loaded {len(protein_names):,} protein names", flush=True)
    except (OSError, ValueError) as e:
        print(f"Error loading {args.flat_members_file}: {e}")
        return 1

    # Construct CIF file paths using the same pattern as AF2Dataset
    print("Constructing CIF file paths...", flush=True)
    cif_files = []
    for uniprot_id in protein_names:
        # Same pattern as AF2Dataset._construct_cif_path()
        filename = f"AF-{uniprot_id}-F1-model_v4.cif"
        cif_path = input_dir / filename
        cif_files.append(cif_path)

    print(f"Constructed {len(cif_files):,} CIF file paths", flush=True)

    # Skip existence filtering - let workers handle missing files during processing
    print(
        "Skipping existence pre-filtering (workers will handle missing files)",
        flush=True,
    )

    if len(cif_files) == 0:
        print("No CIF files to process!")
        return 1

    if args.dry_run:
        # Estimate output
        total_appearances = len(cif_files) * args.coverage
        num_chunks = (total_appearances + args.chunk_size - 1) // args.chunk_size
        est_size_gb = num_chunks * 50 / 1024  # Rough estimate: 50MB per chunk

        print("\nDry run estimates:")
        print(f"  Input files: {len(cif_files):,}")
        print(
            f"  After filtering (max {args.max_length} residues, min {args.min_plddt} pLDDT): ~{len(cif_files)*0.6:.0f}"
        )
        print(f"  Coverage: {args.coverage}x per protein")
        print(f"  Chunk size: {args.chunk_size} proteins")
        print(f"  Estimated chunks: {num_chunks}")
        print(f"  Estimated total size: {est_size_gb:.1f} GB")
        print("  Memory usage: <10 GB (streaming mode)")
        return 0

    # Create temporary directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="af2_chunks_"))
        print(f"Created temporary directory: {temp_dir}")

    try:
        # Process CIF files in parallel and save to temporary files
        if args.num_workers is None:
            args.num_workers = min(os.cpu_count(), 8)
        print(
            f"Processing {len(cif_files)} files with {args.num_workers} workers (cpu_count={os.cpu_count()})..."
        )

        # Split files into batches for workers
        batch_size = args.batch_size
        file_batches = [
            cif_files[i : i + batch_size] for i in range(0, len(cif_files), batch_size)
        ]

        temp_files: List[str] = []
        total_processed = 0
        total_failed = 0
        failed_logs: List[str] = []

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for i, batch in enumerate(file_batches):
                futures.append(
                    executor.submit(
                        process_and_save_batch,
                        batch,
                        args.max_length,
                        i,
                        temp_dir,
                        args.min_plddt,
                        not args.enable_worker_tqdm,
                        args.fail_buffer_flush,
                        args.parse_timeout,
                    )
                )

            start_time = time.time()
            last_log = start_time
            completed_futures = 0

            if not args.no_global_tqdm:
                # Standard tqdm progress bar over completed batches
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing batches"
                ):
                    try:
                        temp_file, processed_count, failed_count, failed_log_path = (
                            future.result()
                        )
                        if temp_file:
                            temp_files.append(temp_file)
                        total_processed += processed_count
                        total_failed += failed_count
                        if failed_log_path:
                            failed_logs.append(failed_log_path)
                    except (RuntimeError, OSError, ValueError) as e:
                        print(f"Batch processing error: {e}")
            else:
                # Quiet mode: periodic aggregated log lines (time-based)
                total_batches = len(futures)
                for future in as_completed(futures):
                    completed_futures += 1
                    try:
                        temp_file, processed_count, failed_count, failed_log_path = (
                            future.result()
                        )
                        if temp_file:
                            temp_files.append(temp_file)
                        total_processed += processed_count
                        total_failed += failed_count
                        if failed_log_path:
                            failed_logs.append(failed_log_path)
                    except (RuntimeError, OSError, ValueError) as e:
                        print(f"Batch processing error: {e}")

                    now = time.time()
                    if (
                        (now - last_log) >= args.progress_log_every
                        or completed_futures == total_batches
                    ):
                        elapsed = now - start_time
                        rate = completed_futures / elapsed if elapsed > 0 else 0
                        remaining_batches = total_batches - completed_futures
                        eta_sec = remaining_batches / rate if rate > 0 else 0
                        eta_min = eta_sec / 60
                        pct = (completed_futures / total_batches) * 100
                        print(
                            (
                                f"[Progress] Batches: {completed_futures}/{total_batches} ({pct:.1f}%) | "
                                f"Proteins OK: {total_processed:,} | Failed/Filtered: {total_failed:,} | "
                                f"Rate: {rate:.2f} batches/s | Elapsed: {elapsed/60:.1f} min | "
                                f"ETA: {eta_min:.1f} min"
                            )
                        )
                        last_log = now

        print("\nProcessing complete:")
        print(f"  Successfully processed (passed filters): {total_processed:,}")
        print(f"  Failed / filtered within workers: {total_failed:,}")
        print(f"  Temporary files created: {len(temp_files):,}")

        if total_processed == 0:
            print("No proteins successfully processed!")
            return 1

        if failed_logs:
            failed_path = output_dir / "failed_files.txt"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Stream failures from log files to avoid loading huge lists into memory
            print("Consolidating failure logs...", flush=True)
            unique_failed = set()
            total_failed_entries = 0

            for fl in failed_logs:
                try:
                    with open(fl, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                unique_failed.add(line)
                                total_failed_entries += 1

                                # Flush to disk every 50k unique failures to avoid memory issues
                                if len(unique_failed) >= 50000:
                                    with open(
                                        failed_path, "a", encoding="utf-8"
                                    ) as out_f:
                                        out_f.write(
                                            "\n".join(sorted(unique_failed)) + "\n"
                                        )
                                    unique_failed.clear()

                except FileNotFoundError:
                    continue

            # Write remaining failures
            if unique_failed:
                with open(failed_path, "a", encoding="utf-8") as out_f:
                    out_f.write("\n".join(sorted(unique_failed)) + "\n")

            print(
                f"Saved consolidated failure log ({total_failed_entries:,} total failures) to {failed_path}",
                flush=True,
            )

        metadata = create_pkl_chunks_streaming(
            temp_files,
            output_dir,
            total_processed,
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

    finally:
        # Clean up temporary files
        if not args.keep_temp_files:
            print(f"Cleaning up temporary files...")
            try:
                shutil.rmtree(temp_dir)
                print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temp directory {temp_dir}: {e}")
        else:
            print(f"Keeping temporary files in: {temp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

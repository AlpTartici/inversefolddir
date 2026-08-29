# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
create_af2_pkl_chunks_streaming_with_dssp.py

Memory-efficient version that streams processed proteins directly to temporary files,
then creates chunks without loading everything into memory at once.
Now includes DSSP secondary structure prediction on AlphaFold2 structures.

Usage:
    python create_af2_pkl_chunks_streaming_with_dssp.py --input_dir /path/to/af2_cifs --cluster_dir /path/to/af_clusters --output_dir /path/to/pkl_chunks
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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback

# BioPython imports for DSSP
from Bio.PDB import DSSP, PDBIO, Atom, Chain, Model, PDBParser, Residue, Structure
from Bio.PDB.mmcifio import MMCIFIO

from data.cif_parser import parse_cif_backbone_auto

# Suppress BioPython DSSP warnings
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
warnings.filterwarnings("ignore", message=".*mmCIF.*")
warnings.filterwarnings("ignore", message=".*Unknown or untrusted program.*")


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


def create_temp_pdb_from_coords(
    coords_dict: Dict, residue_types: List[str], temp_dir: Path
) -> str:
    """
    Create a temporary PDB file from parsed coordinates for DSSP processing.

    Args:
        coords_dict: Dictionary with 'N', 'CA', 'C', 'O' coordinates
        residue_types: List of residue types (3-letter codes)
        temp_dir: Directory for temporary files

    Returns:
        Path to temporary PDB file
    """
    # Create a BioPython structure
    structure = Structure.Structure("temp_structure")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    # Three to one letter code mapping for atom naming
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

    atom_names = ["N", "CA", "C", "O"]
    elements = ["N", "C", "C", "O"]  # Correct chemical elements

    for i, res_type in enumerate(residue_types):
        # Create residue
        res_id = (" ", i + 1, " ")  # (hetflag, resseq, icode)
        residue = Residue.Residue(res_id, res_type, " ")

        # Add atoms with proper elements
        for j, (atom_name, element) in enumerate(zip(atom_names, elements)):
            if (
                i < coords_dict[atom_name].shape[0]
            ):  # Ensure we have coordinates for this residue
                coord = coords_dict[atom_name][i].astype(float)  # Ensure float type
                atom = Atom.Atom(
                    name=atom_name,
                    coord=coord,
                    bfactor=0.0,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=f" {atom_name:<3}",
                    serial_number=i * 4 + j + 1,
                    element=element,
                )
                residue.add(atom)

        chain.add(residue)

    model.add(chain)
    structure.add(model)

    # Write to temporary PDB file
    temp_pdb = temp_dir / f"temp_structure_{os.getpid()}_{time.time():.6f}.pdb"

    # Create PDB file with proper headers for DSSP compatibility
    with open(temp_pdb, "w") as f:
        # Add required CRYST1 header for DSSP
        f.write(
            "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
        )

        # Now write the structure using BioPython
        io = PDBIO()
        io.set_structure(structure)
        io.save(f)

    return str(temp_pdb)


def compute_dssp_af2(
    coords_dict: Dict,
    residue_types: List[str],
    sequence: str,
    cif_path: Optional[str] = None,
    temp_dir: Optional[Path] = None,
    dssp_timeout: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Compute DSSP on AlphaFold2 structure.

    Uses direct mmCIF parsing which is the most reliable approach for AlphaFold2 files.
    Falls back to coordinate-based PDB creation if needed.

    Args:
        coords_dict: Dictionary with 'N', 'CA', 'C', 'O' coordinates (already parsed)
        residue_types: List of residue types (3-letter codes)
        sequence: Expected protein sequence
        cif_path: Path to original CIF file (required for direct mmCIF approach)
        temp_dir: Directory for temporary files
        dssp_timeout: Timeout for DSSP computation

    Returns:
        Tuple of (dssp_array, seq_dssp) - both as lists

    Raises:
        RuntimeError: If DSSP fails or returns only undefined assignments
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    dssp_array = []
    seq_dssp = []
    temp_pdb_path = None
    last_error = None

    try:
        # Strategy 1: Direct DSSP on mmCIF (fastest and most reliable)
        if cif_path and os.path.exists(cif_path):
            try:
                with timeout(dssp_timeout):
                    # Parse mmCIF structure
                    parser = MMCIFParser(QUIET=True)
                    structure = parser.get_structure("af2_structure", cif_path)
                    model = structure[0]

                    # Run DSSP directly on mmCIF with explicit file type
                    dssp = DSSP(model, cif_path, file_type="MMCIF")

                    # Extract DSSP data
                    chain = model.get_chains().__next__()  # Get first chain
                    chain_id = chain.get_id()

                    for residue in chain.get_residues():
                        if residue.get_id()[0] == " ":  # Standard residue
                            dssp_key = (chain_id, residue.get_id())
                            if dssp_key in dssp:
                                ss_data = dssp[dssp_key]
                                dssp_array.append(ss_data[2])  # Secondary structure
                                seq_dssp.append(ss_data[1])  # Amino acid
                            else:
                                dssp_array.append("X")
                                seq_dssp.append("X")

                # Check if we got valid results
                if len(dssp_array) > 0:
                    non_x_count = sum(1 for x in dssp_array if x != "X")
                    if non_x_count > 0:
                        return dssp_array, seq_dssp
                    else:
                        last_error = "Direct mmCIF approach produced only X values"
                else:
                    last_error = "Direct mmCIF approach produced no data"

            except Exception as e:
                last_error = f"Direct mmCIF approach failed: {e}"

        # Strategy 2: Fallback to coordinate-based PDB creation
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir())

        temp_pdb_path = create_temp_pdb_from_coords(
            coords_dict, residue_types, temp_dir
        )

        with timeout(dssp_timeout):
            # Parse temporary PDB
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("temp_structure", temp_pdb_path)
            model = structure[0]

            # Run DSSP
            dssp = DSSP(model, temp_pdb_path, file_type="PDB")

            # Extract DSSP data
            chain = model.get_chains().__next__()  # Get first chain
            chain_id = chain.get_id()

            for residue in chain.get_residues():
                if residue.get_id()[0] == " ":  # Standard residue
                    dssp_key = (chain_id, residue.get_id())
                    if dssp_key in dssp:
                        ss_data = dssp[dssp_key]
                        dssp_array.append(ss_data[2])  # Secondary structure
                        seq_dssp.append(ss_data[1])  # Amino acid
                    else:
                        dssp_array.append("X")
                        seq_dssp.append("X")

        # Clean up temp file
        if temp_pdb_path and os.path.exists(temp_pdb_path):
            os.remove(temp_pdb_path)

        # Return results only if we got meaningful assignments
        if len(dssp_array) > 0:
            # Check if we got any valid secondary structure assignments
            non_x_count = sum(1 for x in dssp_array if x != "X")
            if non_x_count > 0:
                return dssp_array, seq_dssp
            else:
                # All X means DSSP failed to assign any secondary structure
                raise RuntimeError(
                    f"DSSP produced only undefined ('X') assignments. {last_error or ''}"
                )
        else:
            # If we got no data at all, that's a real failure
            raise RuntimeError(
                f"DSSP produced no secondary structure data. {last_error or ''}"
            )

    except Exception as e:
        # Clean up temp file on error
        if temp_pdb_path and os.path.exists(temp_pdb_path):
            try:
                os.remove(temp_pdb_path)
            except:
                pass

        # Re-raise the error so it gets properly logged as a DSSP failure
        raise RuntimeError(f"DSSP computation failed: {str(e)}. {last_error or ''}")


def process_single_cif(
    cif_path: Path,
    max_length: int = 600,
    min_plddt: float = 70.0,
    parse_timeout: int = 60,
    dssp_timeout: int = 10,
    temp_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Process a single AF2 CIF file into optimized format with DSSP secondary structure.

    Args:
        cif_path: Path to CIF file
        max_length: Maximum protein length to include
        min_plddt: Minimum average pLDDT score to include (default: 70.0)
        parse_timeout: Maximum seconds to spend parsing each file (default: 60)
        dssp_timeout: Maximum seconds for DSSP computation (default: 10)
        temp_dir: Directory for temporary files
        verbose: If True, include detailed error information in failures

    Returns:
        Dictionary with protein data or None if processing failed/filtered
    """
    failure_reason = None
    try:
        # Parse CIF file with timeout protection
        with timeout(parse_timeout):
            try:
                coords, plddt_scores, residue_types, _ = parse_cif_backbone_auto(
                    str(cif_path)
                )
            except Exception as e:
                failure_reason = f"CIF parsing failed: {e}"
                if verbose:
                    import traceback

                    failure_reason += f" | Traceback: {traceback.format_exc()}"
                raise ValueError(failure_reason)

        if coords is None or len(coords) == 0:
            failure_reason = "No coordinates found in CIF file"
            raise ValueError(failure_reason)

        # Apply length filter
        if len(coords) > max_length:
            failure_reason = f"Sequence too long: {len(coords)} > {max_length}"
            raise ValueError(failure_reason)

        # Convert to numpy for better pickle efficiency
        try:
            if hasattr(coords, "numpy"):
                coords_np = coords.numpy()
            else:
                coords_np = np.array(coords)

            if hasattr(plddt_scores, "numpy"):
                plddt_np = plddt_scores.numpy()
            else:
                plddt_np = np.array(plddt_scores)
        except Exception as e:
            failure_reason = f"Failed to convert coordinates to numpy: {e}"
            if verbose:
                import traceback

                failure_reason += f" | Traceback: {traceback.format_exc()}"
            raise ValueError(failure_reason)

        # Apply pLDDT quality filter - check average confidence
        try:
            avg_plddt = float(np.mean(plddt_np))
            if avg_plddt < min_plddt:
                failure_reason = f"Average pLDDT too low: {avg_plddt:.2f} < {min_plddt}"
                raise ValueError(failure_reason)
        except Exception as e:
            failure_reason = f"Failed to compute average pLDDT: {e}"
            if verbose:
                import traceback

                failure_reason += f" | Traceback: {traceback.format_exc()}"
            raise ValueError(failure_reason)

        # Create sequence string from residue types
        try:
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
        except Exception as e:
            failure_reason = f"Failed to create sequence string: {e}"
            if verbose:
                import traceback

                failure_reason += f" | Traceback: {traceback.format_exc()}"
            raise ValueError(failure_reason)

        # NEW: Compute DSSP secondary structure
        try:
            coords_dict = {
                "N": coords_np[:, 0, :].astype(np.float32),  # [L, 3]
                "CA": coords_np[:, 1, :].astype(np.float32),  # [L, 3]
                "C": coords_np[:, 2, :].astype(np.float32),  # [L, 3]
                "O": coords_np[:, 3, :].astype(np.float32),  # [L, 3]
            }
        except Exception as e:
            failure_reason = f"Failed to create coordinate dictionary: {e}"
            if verbose:
                import traceback

                failure_reason += f" | Traceback: {traceback.format_exc()}"
            raise ValueError(failure_reason)

        dssp_result = None
        dssp_error = None

        try:
            dssp_array, seq_dssp = compute_dssp_af2(
                coords_dict,
                residue_types,
                sequence,
                str(cif_path),
                temp_dir,
                dssp_timeout,
            )

            # Align DSSP results with sequence if needed
            if len(dssp_array) == len(sequence):
                dssp_result = dssp_array
            else:
                # Length mismatch - pad or truncate as needed
                if len(dssp_array) < len(sequence):
                    # Pad with 'X'
                    dssp_result = dssp_array + ["X"] * (len(sequence) - len(dssp_array))
                else:
                    # Truncate
                    dssp_result = dssp_array[: len(sequence)]

        except Exception as e:
            dssp_error = f"DSSP computation failed: {e}"
            if verbose:
                import traceback

                dssp_error += f" | Traceback: {traceback.format_exc()}"
            dssp_result = ["X"] * len(sequence)

        # Extract UniProt ID
        try:
            uniprot_id = extract_uniprot_id(cif_path.name)
        except Exception as e:
            failure_reason = f"Failed to extract UniProt ID: {e}"
            if verbose:
                import traceback

                failure_reason += f" | Traceback: {traceback.format_exc()}"
            raise ValueError(failure_reason)

        # Create protein entry in CATH format with DSSP
        try:
            protein_data = {
                "seq": sequence,
                "coords": coords_dict,
                "plddt": plddt_np.astype(np.float32),
                "dssp": dssp_result,  # NEW: DSSP secondary structure
                "source": "alphafold2",
                "num_chains": 1,
                "name": uniprot_id,
                "CATH": ["unknown"],
            }
        except Exception as e:
            failure_reason = f"Failed to create protein data dictionary: {e}"
            if verbose:
                import traceback

                failure_reason += f" | Traceback: {traceback.format_exc()}"
            raise ValueError(failure_reason)

        # Return result with DSSP status
        result = (uniprot_id, protein_data)
        if dssp_error:
            result = result + (dssp_error,)  # Add error info for logging

        return result

    except TimeoutError as e:
        failure_reason = f"Processing timeout after {parse_timeout}s: {e}"
        if verbose:
            import traceback

            failure_reason += f" | Traceback: {traceback.format_exc()}"
        if verbose:
            print(f"⏰ TIMEOUT: {cif_path.name} - {failure_reason}")
        return None

    except ValueError as e:
        # These are expected filtering failures (length, pLDDT, etc.)
        if verbose:
            print(f"FILTERED: {cif_path.name} - {e}")
        return None

    except (FileNotFoundError, OSError) as e:
        failure_reason = f"File access error: {e}"
        if verbose:
            import traceback

            failure_reason += f" | Traceback: {traceback.format_exc()}"
            print(f"FILE ERROR: {cif_path.name} - {failure_reason}")
        return None

    except Exception as e:
        # Unexpected errors
        failure_reason = f"Unexpected error: {e}"
        if verbose:
            import traceback

            failure_reason += f" | Traceback: {traceback.format_exc()}"
            print(f"[error] UNEXPECTED ERROR: {cif_path.name} - {failure_reason}")
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
    dssp_timeout: int = 10,
    verbose: bool = False,
) -> Tuple[str, int, int, int, str, str]:
    """
    Process a batch of CIF files and save directly to temporary file.

    Returns: (temp_file_path, num_processed, failed_count, dssp_failed_count, failed_log_path, dssp_failed_log_path)
    """
    processed_count = 0
    failed_count = 0
    dssp_failed_count = 0

    # Create temporary files for this batch
    temp_file = temp_dir / f"batch_{worker_id:06d}.pkl"
    failed_log_path = temp_dir / f"failed_{worker_id:06d}.txt"
    dssp_failed_log_path = temp_dir / f"dssp_failed_{worker_id:06d}.txt"

    processed_proteins: Dict[str, Dict] = {}
    failure_buffer: List[str] = []
    dssp_failure_buffer: List[str] = []

    # Create worker-specific temp directory for DSSP
    worker_temp_dir = temp_dir / f"worker_{worker_id}_temp"
    worker_temp_dir.mkdir(exist_ok=True)

    iterator = (
        cif_paths
        if disable_tqdm
        else tqdm(cif_paths, desc=f"Worker {worker_id}", leave=False)
    )
    for cif_path in iterator:
        result = process_single_cif(
            cif_path,
            max_length,
            min_plddt,
            parse_timeout,
            dssp_timeout,
            worker_temp_dir,
            verbose,
        )

        if result is not None:
            if len(result) == 3:  # Has DSSP error
                protein_id, protein_data, dssp_error = result
                dssp_failed_count += 1
                dssp_failure_buffer.append(f"{str(cif_path)}: {dssp_error}")

                if verbose:
                    print(f"DSSP WARNING: {cif_path.name} - {dssp_error}")
            else:
                protein_id, protein_data = result

            processed_proteins[protein_id] = protein_data
            processed_count += 1

            if verbose and processed_count % 100 == 0:
                print(f"[ok] Worker {worker_id}: Processed {processed_count} proteins")
        else:
            failed_count += 1
            failure_buffer.append(str(cif_path))

        # Flush failure buffers if large
        if len(failure_buffer) >= fail_buffer_flush:
            with open(failed_log_path, "a", encoding="utf-8") as fl:
                fl.write("\n".join(failure_buffer) + "\n")
            failure_buffer.clear()

        if len(dssp_failure_buffer) >= fail_buffer_flush:
            with open(dssp_failed_log_path, "a", encoding="utf-8") as dl:
                dl.write("\n".join(dssp_failure_buffer) + "\n")
            dssp_failure_buffer.clear()

    # Final buffer flushes
    if failure_buffer:
        with open(failed_log_path, "a", encoding="utf-8") as fl:
            fl.write("\n".join(failure_buffer) + "\n")

    if dssp_failure_buffer:
        with open(dssp_failed_log_path, "a", encoding="utf-8") as dl:
            dl.write("\n".join(dssp_failure_buffer) + "\n")

    # Clean up worker temp directory
    try:
        shutil.rmtree(worker_temp_dir)
    except:
        pass

    # Save batch to temporary file
    if processed_proteins:
        with open(temp_file, "wb") as f:
            pickle.dump(processed_proteins, f, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(
                f"Worker {worker_id}: Saved {len(processed_proteins)} proteins to {temp_file}"
            )

    if verbose:
        print(
            f"Worker {worker_id} SUMMARY: {processed_count} processed, {failed_count} failed, {dssp_failed_count} DSSP warnings"
        )

    return (
        str(temp_file),
        processed_count,
        failed_count,
        dssp_failed_count,
        str(failed_log_path if failed_log_path.exists() else ""),
        str(dssp_failed_log_path if dssp_failed_log_path.exists() else ""),
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
        "includes_dssp": True,  # Mark that this dataset includes DSSP
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_path}")
    print(f"Average chunk size: {np.mean([s['size_mb'] for s in chunk_stats]):.1f}MB")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create AF2 pickle chunks with DSSP secondary structure (memory-efficient)"
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
        "--dssp_timeout",
        type=int,
        default=10,
        help="Maximum seconds for DSSP computation per protein (default: 10)",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose error reporting for debugging failures",
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
        # Estimate output with DSSP overhead
        total_appearances = len(cif_files) * args.coverage
        num_chunks = (total_appearances + args.chunk_size - 1) // args.chunk_size
        est_size_gb = (
            num_chunks * 60 / 1024
        )  # Rough estimate: 60MB per chunk (with DSSP)

        print("\nDry run estimates:")
        print(f"  Input files: {len(cif_files):,}")
        print(
            f"  After filtering (max {args.max_length} residues, min {args.min_plddt} pLDDT): ~{len(cif_files)*0.6:.0f}"
        )
        print(f"  Coverage: {args.coverage}x per protein")
        print(f"  Chunk size: {args.chunk_size} proteins")
        print(f"  Estimated chunks: {num_chunks}")
        print(f"  Estimated total size: {est_size_gb:.1f} GB (including DSSP data)")
        print(f"  Memory usage: <10 GB (streaming mode)")
        print(f"  DSSP timeout: {args.dssp_timeout}s per protein")
        return 0

    # Create temporary directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="af2_chunks_dssp_"))
        print(f"Created temporary directory: {temp_dir}")

    try:
        # Process CIF files in parallel and save to temporary files
        if args.num_workers is None:
            args.num_workers = min(os.cpu_count(), 8)
        print(
            f"Processing {len(cif_files)} files with {args.num_workers} workers (cpu_count={os.cpu_count()})..."
        )
        print(f"DSSP timeout: {args.dssp_timeout}s per protein")

        if args.verbose:
            print("VERBOSE MODE: Detailed error reporting enabled")
            print("   - Will show specific reasons for filtering/failures")
            print("   - Will report DSSP computation issues")
            print("   - Will track worker progress in detail")

        # Split files into batches for workers
        batch_size = args.batch_size
        file_batches = [
            cif_files[i : i + batch_size] for i in range(0, len(cif_files), batch_size)
        ]

        temp_files: List[str] = []
        total_processed = 0
        total_failed = 0
        total_dssp_failed = 0
        failed_logs: List[str] = []
        dssp_failed_logs: List[str] = []

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
                        args.dssp_timeout,
                        args.verbose,  # Pass verbose flag
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
                        (
                            temp_file,
                            processed_count,
                            failed_count,
                            dssp_failed_count,
                            failed_log_path,
                            dssp_failed_log_path,
                        ) = future.result()
                        if temp_file:
                            temp_files.append(temp_file)
                        total_processed += processed_count
                        total_failed += failed_count
                        total_dssp_failed += dssp_failed_count
                        if failed_log_path:
                            failed_logs.append(failed_log_path)
                        if dssp_failed_log_path:
                            dssp_failed_logs.append(dssp_failed_log_path)
                    except (RuntimeError, OSError, ValueError) as e:
                        print(f"Batch processing error: {e}")
            else:
                # Quiet mode: periodic aggregated log lines (time-based)
                total_batches = len(futures)
                for future in as_completed(futures):
                    completed_futures += 1
                    try:
                        (
                            temp_file,
                            processed_count,
                            failed_count,
                            dssp_failed_count,
                            failed_log_path,
                            dssp_failed_log_path,
                        ) = future.result()
                        if temp_file:
                            temp_files.append(temp_file)
                        total_processed += processed_count
                        total_failed += failed_count
                        total_dssp_failed += dssp_failed_count
                        if failed_log_path:
                            failed_logs.append(failed_log_path)
                        if dssp_failed_log_path:
                            dssp_failed_logs.append(dssp_failed_log_path)
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
                                f"DSSP Failed: {total_dssp_failed:,} | "
                                f"Rate: {rate:.2f} batches/s | Elapsed: {elapsed/60:.1f} min | "
                                f"ETA: {eta_min:.1f} min"
                            )
                        )
                        last_log = now

        print("\nProcessing complete:")
        print(f"  Successfully processed (passed filters): {total_processed:,}")
        print(f"  Failed / filtered within workers: {total_failed:,}")
        print(f"  DSSP computation failures: {total_dssp_failed:,}")
        print(f"  Temporary files created: {len(temp_files):,}")

        if total_processed == 0:
            print("No proteins successfully processed!")
            return 1

        # Save consolidated failure logs
        output_dir.mkdir(parents=True, exist_ok=True)

        if failed_logs:
            failed_path = output_dir / "failed_files.txt"
            print("Consolidating general failure logs...", flush=True)
            unique_failed = set()

            for fl in failed_logs:
                try:
                    with open(fl, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                unique_failed.add(line)
                except FileNotFoundError:
                    continue

            with open(failed_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(sorted(unique_failed)) + "\n")

            print(
                f"Saved general failure log ({len(unique_failed):,} unique failures) to {failed_path}"
            )

        # Save DSSP failure logs
        if dssp_failed_logs:
            dssp_failed_path = output_dir / "dssp_failed_files.txt"
            print("Consolidating DSSP failure logs...", flush=True)
            unique_dssp_failed = set()

            for dfl in dssp_failed_logs:
                try:
                    with open(dfl, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                unique_dssp_failed.add(line)
                except FileNotFoundError:
                    continue

            with open(dssp_failed_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(sorted(unique_dssp_failed)) + "\n")

            print(
                f"Saved DSSP failure log ({len(unique_dssp_failed):,} unique DSSP failures) to {dssp_failed_path}"
            )

        metadata = create_pkl_chunks_streaming(
            temp_files,
            output_dir,
            total_processed,
            chunk_size=args.chunk_size,
            coverage_per_protein=args.coverage,
            random_seed=args.random_seed,
        )

        print(
            f"\n[ok] Successfully created {metadata['num_chunks']} pickle chunks with DSSP!"
        )
        print(f"Output directory: {output_dir}")
        print(f"Total proteins: {metadata['total_proteins']:,}")
        print(
            f"DSSP failures: {total_dssp_failed:,} ({100*total_dssp_failed/total_processed:.1f}%)"
        )
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

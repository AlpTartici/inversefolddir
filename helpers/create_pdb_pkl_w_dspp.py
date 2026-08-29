# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import pickle
import sys
import traceback
import warnings
from functools import partial
from multiprocessing import Manager, Pool, cpu_count

from Bio.PDB import DSSP, PDBParser
from tqdm import tqdm

# Suppress BioPython DSSP warnings about mmCIF validation
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
warnings.filterwarnings("ignore", message=".*mmCIF.*")
warnings.filterwarnings("ignore", message=".*Unknown or untrusted program.*")


def compute_dssp_with_missing_handling(
    pdb_file, reference_sequence, model_index=0, chain_id=None
):
    """
    Compute DSSP while handling missing residues by assigning 'X'.

    Args:
        pdb_file: Path to PDB file
        reference_sequence: The complete expected sequence (including missing residues)
        model_index: Model index (default 0)
        chain_id: Chain ID to process

    Returns:
        dssp_array: List of secondary structure assignments (including 'X' for missing)
        seq_dssp: List of amino acids from DSSP (including 'X' for missing)
    """
    # Parse the PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_file)
    model = structure[model_index]

    # Run DSSP
    try:
        dssp = DSSP(model, pdb_file, file_type="PDB")
    except TypeError:
        dssp = DSSP(model, pdb_file)

    # Get the target chain
    target_chain = None
    for chain in model:
        if chain_id is None or chain.id == chain_id:
            target_chain = chain
            actual_chain_id = chain.id
            break

    if target_chain is None:
        # No chain found - raise an exception
        raise Exception("No target chain found")
    # Get residues from PDB
    protein_residues = [r for r in target_chain.get_residues() if r.get_id()[0] == " "]

    # Create mapping of PDB residues by position
    pdb_residues = {}
    for residue in protein_residues:
        res_num = residue.get_id()[1]
        pdb_residues[res_num] = residue

    # Extract DSSP sequence to align with reference
    dssp_seq_from_pdb = ""
    for residue in protein_residues:
        dssp_key = (actual_chain_id, residue.get_id())
        if dssp_key in dssp:
            dssp_seq_from_pdb += dssp[dssp_key][1]  # amino acid from DSSP
        else:
            dssp_seq_from_pdb += "X"  # DSSP failed for this residue

    # Try to align DSSP sequence with reference sequence
    dssp_array = []
    seq_dssp = []

    # Simple approach: if lengths match, assume direct correspondence
    if len(dssp_seq_from_pdb) == len(reference_sequence):
        # Direct mapping
        for i, (ref_aa, pdb_residue) in enumerate(
            zip(reference_sequence, protein_residues)
        ):
            dssp_key = (actual_chain_id, pdb_residue.get_id())
            if dssp_key in dssp:
                ss = dssp[dssp_key]
                dssp_array.append(ss[2])  # Secondary structure
                seq_dssp.append(ss[1])  # Amino acid
            else:
                dssp_array.append("X")
                seq_dssp.append("X")
    else:
        # Length mismatch - fill missing positions with 'X'
        # This is a simplified approach; you might need more sophisticated alignment
        ref_pos = 0
        pdb_pos = 0

        while ref_pos < len(reference_sequence):
            if pdb_pos < len(protein_residues):
                pdb_residue = protein_residues[pdb_pos]
                dssp_key = (actual_chain_id, pdb_residue.get_id())

                if dssp_key in dssp:
                    ss = dssp[dssp_key]
                    pdb_aa = ss[1]

                    # Check if this PDB residue matches the reference
                    if (
                        pdb_aa == reference_sequence[ref_pos]
                        or reference_sequence[ref_pos] == "X"
                    ):
                        dssp_array.append(ss[2])
                        seq_dssp.append(pdb_aa)
                        pdb_pos += 1
                    else:
                        # Mismatch - assume missing residue in PDB
                        dssp_array.append("X")
                        seq_dssp.append("X")
                else:
                    # DSSP failed for this residue
                    dssp_array.append("X")
                    seq_dssp.append("X")
                    pdb_pos += 1
            else:
                # No more PDB residues - rest are missing
                dssp_array.append("X")
                seq_dssp.append("X")

            ref_pos += 1

    return dssp_array, seq_dssp


def compute_dssp(pdb_file, model_index=0, chain_id=None):
    # Parse the PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_file)
    model = structure[model_index]

    # Run DSSP with explicit classic format (avoids mmCIF issues)
    try:
        dssp = DSSP(model, pdb_file, file_type="PDB")
    except TypeError:
        # Fallback for older BioPython versions
        dssp = DSSP(model, pdb_file)

    # Get the target chain
    target_chain = None
    for chain in model:
        if chain_id is None or chain.id == chain_id:
            target_chain = chain
            actual_chain_id = chain.id
            break

    if target_chain is None:
        return [], []

    # Get all residue positions in the chain (including missing ones)
    all_residues = list(target_chain.get_residues())
    protein_residues = [
        r for r in all_residues if r.get_id()[0] == " "
    ]  # Only standard amino acids

    if not protein_residues:
        return [], []

    # Get the range of residue numbers
    residue_numbers = [r.get_id()[1] for r in protein_residues]
    min_res = min(residue_numbers)
    max_res = max(residue_numbers)

    # Create a mapping of existing residues
    existing_residues = {r.get_id()[1]: r for r in protein_residues}

    # Initialize arrays for the complete sequence range
    dssp_array = []
    seq_dssp = []

    # Process each position in the sequence range
    for res_num in range(min_res, max_res + 1):
        if res_num in existing_residues:
            # Residue exists - try to get DSSP data
            residue = existing_residues[res_num]
            dssp_key = (actual_chain_id, residue.get_id())

            if dssp_key in dssp:
                # DSSP data available
                ss = dssp[dssp_key]
                dssp_array.append(ss[2])  # Secondary structure
                seq_dssp.append(ss[1])  # Amino acid
            else:
                # Residue exists but DSSP couldn't compute (e.g., incomplete coordinates)
                print(f"added X at res_num {res_num} bc dssp_key is not in dssp")
                dssp_array.append("X")
                seq_dssp.append("X")
        else:
            # Missing residue - assign 'X'
            print(
                f"added X at res_num {res_num} bc res_num is not in existing residues"
            )
            dssp_array.append("X")
            seq_dssp.append("X")

    return dssp_array, seq_dssp


def process_single_entry(args):
    """
    Process a single entry for parallel execution.
    Returns a tuple: (entry, result_dict)
    """
    entry, sample = args
    try:
        pdb_id, chain_id = sample["name"].split(".")
        path = f"../datasets/all_pdbs/{pdb_id}.pdb"

        # Check if PDB file exists
        if not os.path.exists(path):
            # Create fallback DSSP array with 'X' values for missing PDB files
            seq = sample.get("seq", "")
            fallback_dssp = ["X"] * len(seq)

            return entry, {
                "status": "missing_pdb",
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "seq_len": len(seq),
                "dssp_len": len(fallback_dssp),
                "dssp": fallback_dssp,
                "error": None,
            }

        # Compute DSSP - choose between two approaches:

        # Approach 1: Simple version (current)
        # dssp, seq_dssp = compute_dssp(path, chain_id=chain_id)

        # Approach 2: Advanced version with reference sequence alignment
        dssp, seq_dssp = compute_dssp_with_missing_handling(
            path, sample["seq"], chain_id=chain_id
        )

        seq = sample["seq"]

        # Check alignment
        if len(seq) != len(seq_dssp):
            return entry, {
                "status": "misaligned",
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "seq_len": len(seq),
                "seq_dssp_len": len(seq_dssp),
                "dssp_len": len(dssp),
                "dssp": dssp,
                "error": None,
            }
        else:
            return entry, {
                "status": "aligned",
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "seq_len": len(seq),
                "seq_dssp_len": len(seq_dssp),
                "dssp_len": len(dssp),
                "dssp": dssp,
                "error": None,
            }

    except Exception as e:
        # Create fallback DSSP array with 'X' values matching sequence length
        seq = sample.get("seq", "")
        fallback_dssp = ["X"] * len(seq)

        return entry, {
            "status": "error",
            "pdb_id": pdb_id if "pdb_id" in locals() else "unknown",
            "chain_id": chain_id if "chain_id" in locals() else "unknown",
            "seq_len": len(seq),
            "dssp_len": len(fallback_dssp),
            "dssp": fallback_dssp,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    """Main function to orchestrate the parallel processing."""
    print("Starting DSSP Processing with Multiprocessing")
    print("=" * 50)

    # Load data
    print("Loading data...")
    with open("../datasets/cath-4.2/chain_set_map_with_b_factors.pkl", "rb") as file:
        data_pdb = pickle.load(file)

    print(f"Total entries to process: {len(data_pdb)}")

    # Determine number of processes
    num_processes = min(cpu_count(), 16)  # Cap at 8 to avoid overwhelming the system
    print(f"Using {num_processes} processes")

    # Prepare data for parallel processing
    entries = list(data_pdb.items())  # [:5]

    # Initialize result containers
    missing_pdbs = []
    misaligned = []
    aligned = []
    errors = []

    # Process in parallel with progress bar
    print("Processing entries in parallel...")

    with Pool(processes=num_processes) as pool:
        # Use imap for better memory efficiency with large datasets
        results = list(
            tqdm(
                pool.imap(process_single_entry, entries, chunksize=50),
                total=len(entries),
                desc="Processing entries",
            )
        )

    # Process results and update data_pdb
    print("Processing results...")

    for entry, result in tqdm(results, desc="Updating data"):
        if result["status"] == "missing_pdb":
            # Still save the fallback DSSP data (all 'X' values) for missing PDB files
            data_pdb[entry]["dssp"] = result["dssp"]
            missing_pdbs.append(
                (
                    entry,
                    result["pdb_id"],
                    result["chain_id"],
                    result["seq_len"],
                    result["dssp_len"],
                )
            )

        elif result["status"] == "misaligned":
            misaligned.append(
                (
                    entry,
                    result["pdb_id"],
                    result["chain_id"],
                    result["seq_len"],
                    result["seq_dssp_len"],
                    result["dssp_len"],
                )
            )
            data_pdb[entry]["dssp"] = result["dssp"]

        elif result["status"] == "aligned":
            aligned.append(
                (
                    entry,
                    result["pdb_id"],
                    result["chain_id"],
                    result["seq_len"],
                    result["seq_dssp_len"],
                    result["dssp_len"],
                )
            )
            data_pdb[entry]["dssp"] = result["dssp"]

        elif result["status"] == "error":
            # Still save the fallback DSSP data (all 'X' values)
            data_pdb[entry]["dssp"] = result["dssp"]
            errors.append(
                (
                    entry,
                    result["pdb_id"],
                    result["chain_id"],
                    result["error"],
                    result.get("traceback", ""),
                    result["seq_len"],
                    result["dssp_len"],
                )
            )

    # Print summary statistics
    print("\n Processing Summary:")
    print(f"[ok] Successfully aligned: {len(aligned)}")
    print(f"[warn] Misaligned sequences: {len(misaligned)}")
    print(f"[error] Missing PDB files: {len(missing_pdbs)} (assigned 'X' values)")
    print(f"Processing errors: {len(errors)} (assigned 'X' values)")
    print(f"Total processed: {len(aligned) + len(misaligned)}")
    print(
        f"Total with DSSP data: {len(aligned) + len(misaligned) + len(missing_pdbs) + len(errors)}"
    )
    print(f"   All entries now have 'dssp' field with proper length!")

    # Save results
    print("\n Saving results...")

    # Save updated data with DSSP
    with open(
        "../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl", "wb"
    ) as file:
        pickle.dump(data_pdb, file)
    print("[ok] Saved main dataset with DSSP data")

    # Save detailed logs
    with open("missing_pdbs.txt", "w") as file:
        for entry, pdb_id, chain_id, seq_len, dssp_len in missing_pdbs:
            file.write(
                f"{entry}: {pdb_id}.{chain_id}, len(seq): {seq_len}, len(dssp): {dssp_len} (X values)\n"
            )
    print(f"[ok] Saved missing PDBs log ({len(missing_pdbs)} entries)")
    print(f"   Note: Missing PDB entries still have DSSP data (all 'X' values)")

    with open("misaligned.txt", "w") as file:
        for entry, pdb_id, chain_id, seq_len, seq_dssp_len, dssp_len in misaligned:
            file.write(
                f"{entry}: {pdb_id}.{chain_id}, len(seq): {seq_len}, len(seq_dssp): {seq_dssp_len}, len(dssp): {dssp_len}\n"
            )
    print(f"Saved misaligned sequences log ({len(misaligned)} entries)")

    with open("aligned.txt", "w") as file:
        for entry, pdb_id, chain_id, seq_len, seq_dssp_len, dssp_len in aligned:
            file.write(
                f"{entry}: {pdb_id}.{chain_id}, len(seq): {seq_len}, len(seq_dssp): {seq_dssp_len}, len(dssp): {dssp_len}\n"
            )
    print(f"Saved aligned sequences log ({len(aligned)} entries)")

    # Save error log if there are errors
    if errors:
        with open("processing_errors.txt", "w") as file:
            for (
                entry,
                pdb_id,
                chain_id,
                error,
                traceback_str,
                seq_len,
                dssp_len,
            ) in errors:
                file.write(f"Entry: {entry}\n")
                file.write(f"PDB: {pdb_id}.{chain_id}\n")
                file.write(f"Error: {error}\n")
                file.write(
                    f"Sequence length: {seq_len}, DSSP length (X values): {dssp_len}\n"
                )
                file.write(f"Traceback:\n{traceback_str}\n")
                file.write("-" * 50 + "\n")
        print(f"Saved error log ({len(errors)} entries)")
        print(f"   Note: Error entries still have DSSP data (all 'X' values)")

    print("\n Processing completed successfully!")
    return len(aligned) + len(misaligned), len(missing_pdbs), len(errors)


if __name__ == "__main__":
    try:
        processed, missing, errors = main()
        print(
            f"\n Final Results: {processed} processed, {missing} missing, {errors} errors"
        )
    except KeyboardInterrupt:
        print("\n[warn] Processing interrupted by user")
    except Exception as e:
        print(f"\n Fatal error: {e}")
        traceback.print_exc()

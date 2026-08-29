# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# read ../datasets/cath-4.2/chain_set_splits.json into data
import json
import os

import requests
from Bio import PDB
from tqdm import tqdm

print("Starting PDB chain extraction script...")
with open("../datasets/cath-4.2/chain_set_splits.json") as f:
    data = json.load(f)


data_test = data["test"]
data_validation = data["validation"]
combined_data = data_test + data_validation
data_train = data["train"]


parser = PDB.PDBParser()

io = PDB.PDBIO()


# Chain selector class for proper PDB formatting
class ChainSelect(PDB.Select):
    """Select only a specific chain while maintaining proper PDB structure."""
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        """Accept only the specified chain."""
        return chain.get_id() == self.chain_id


# wrap that in a function
def extract_chain_from_pdb(pdb_code, chain):
    """
    Extracts a specific chain from a PDB file and writes it to a new file.

    Parameters:
    - pdb_code: PDB code to download.
    - chain: The chain identifier to keep (e.g., 'A').
    """
    DATA_DIR = "../datasets"
    full_pdb_path = f"{DATA_DIR}/val_full_pdbs/{pdb_code}.pdb"
    chain_pdb_path = f"{DATA_DIR}/val_chain_pdbs/{pdb_code}_chain{chain}.pdb"

    # if the path exists, don't do anything
    if os.path.exists(chain_pdb_path):
        return 3

    pdb_code = pdb_code.lower()
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    response = requests.get(url)

    if response.status_code == 200:

        with open(full_pdb_path, "w") as file:
            file.write(response.text)
    else:
        print(f"Failed to download PDB file for code: {pdb_code}")
        return pdb_code
    try:
        parser = PDB.PDBParser()
        structure = parser.get_structure("protein", full_pdb_path)

        io = PDB.PDBIO()
        io.set_structure(structure)  # Set the full structure
        # Use ChainSelect to save only the specified chain in proper PDB format
        io.save(chain_pdb_path, ChainSelect(chain))
    except Exception as e:
        print(f"Error processing {pdb_code}: {e}")
        return pdb_code
    return 3


failed_entries = []
for entry in tqdm(data_validation):
    pdb_code, chain_id = entry.split(".")
    output = extract_chain_from_pdb(pdb_code, chain_id)
    if output != 3:
        failed_entries.append(entry)

# Save failed entries to a file
with open("failed_entries_validation.txt", "w") as f:
    for entry in failed_entries:
        f.write(f"{entry}\n")

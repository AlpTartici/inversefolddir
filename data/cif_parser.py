# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
cif_parser.py

This script provides functions to parse Crystallographic Information Files (CIF)
and extract key structural information for protein backbones. It is designed
to handle both traditional PDB files and AlphaFold2 prediction files.

This file is a dependency for the `build` method in `data/graph_builder.py`,
but note that the `build` method itself is not used in the main training
pipeline, which uses pre-processed data instead.
"""

"""
TTD: extract residue too
"""


import re

import torch
from Bio.PDB import MMCIFParser


def parse_cif_backbone(cif_path: str):
    """
    Parses a single mmCIF file to extract backbone atom coordinates and B-factors.

    This function focuses on the four main backbone atoms: N, CA, C, and O. It
    iterates through the first model and chain in the CIF file, collecting the
    coordinates for these atoms, the B-factor of the alpha-carbon (CA), and the
    three-letter code for each residue.

    Args:
        cif_path: The file path to the mmCIF file.

    Returns:
        A tuple containing:
        - coords (torch.Tensor): A tensor of shape [L, 4, 3] containing the
          xyz coordinates of the N, CA, C, and O atoms for each of the L residues.
        - b_factors (torch.Tensor): A tensor of shape [L] containing the B-factor
          of the CA atom for each residue.
        - residue_types (List[str]): A list of L strings, where each string is the
          three-letter code of the residue (e.g., "ALA", "CYS").
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", cif_path)
    model = structure[0]
    # assume first chain only
    chain = next(model.get_chains())
    coords, b_factors, residue_types = [], [], []
    for res in chain:
        if not res.has_id("CA"):
            continue
        atom_coords = []
        for name in ("N", "CA", "C", "O"):
            atom = res[name]
            atom_coords.append(atom.get_coord())
        coords.append(atom_coords)
        b_factors.append(res["CA"].get_bfactor())
        residue_types.append(res.get_resname())
    coords = torch.tensor(coords, dtype=torch.float)    # [L,4,3]
    b_factors = torch.tensor(b_factors, dtype=torch.float)  # [L]
    return coords, b_factors, residue_types


def parse_alphafold2_cif(cif_path: str):
    """
    Parses an AlphaFold2 CIF file to extract backbone coordinates and pLDDT scores.

    AlphaFold2 CIF files use the mmCIF format with structured data columns, where
    pLDDT scores are stored in the B-factor column. This function handles both
    PDB ATOM record format and mmCIF format.

    Args:
        cif_path: The file path to the AlphaFold2 CIF file.

    Returns:
        A tuple containing:
        - coords (torch.Tensor): A tensor of shape [L, 4, 3] containing the
          xyz coordinates of the N, CA, C, and O atoms for each of the L residues.
        - plddt_scores (torch.Tensor): A tensor of shape [L] containing the pLDDT
          scores (confidence measures) for each residue.
        - residue_types (List[str]): A list of L strings, where each string is the
          three-letter code of the residue (e.g., "ALA", "CYS").
    """
    coords, plddt_scores, residue_types = [], [], []

    with open(cif_path, 'r') as f:
        lines = f.readlines()

    # Check if this is mmCIF format (AlphaFold2) or PDB ATOM format
    is_mmcif = False
    atom_lines = []

    for line in lines:
        if line.startswith('ATOM') and not is_mmcif:
            # Check if this looks like mmCIF format (lots of columns)
            parts = line.split()
            if len(parts) > 15:  # mmCIF has many more columns than PDB ATOM
                is_mmcif = True
            atom_lines.append(line)
        elif line.startswith('ATOM'):
            atom_lines.append(line)

    if not atom_lines:
        raise ValueError(f"No ATOM records found in {cif_path}")

    # Group atoms by residue
    current_residue = None
    current_coords = {}
    current_plddt = None
    current_resname = None

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 11:
            continue

        if is_mmcif:
            # mmCIF format: ATOM id type atom_name alt residue_name chain entity seq_id ins_code x y z occupancy b_factor ...
            # Example: ATOM 1 N N . MET A 1 1 ? -6.743 9.797 -14.974 1.0 58.00 ...
            if len(parts) < 15:
                continue
            atom_name = parts[3]  # N, CA, C, O
            res_name = parts[5]   # MET, LEU, etc.
            try:
                res_seq = int(parts[8])   # sequence number
                x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                temp_factor = float(parts[14])  # pLDDT score
            except (ValueError, IndexError):
                continue
        else:
            # PDB ATOM format: ATOM serial name altLoc resName chainID resSeq iCode x y z occupancy tempFactor element charge
            atom_name = parts[2]
            res_name = parts[3]
            try:
                res_seq = int(parts[5])
                x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                temp_factor = float(parts[10])  # This is pLDDT in AlphaFold2 files
            except (ValueError, IndexError):
                continue

        # Only process backbone atoms
        if atom_name not in ['N', 'CA', 'C', 'O']:
            continue

        residue_id = (res_seq, res_name)

        # If we've moved to a new residue, save the previous one
        if current_residue is not None and residue_id != current_residue:
            if len(current_coords) == 4:  # All backbone atoms present
                atom_coords = [
                    current_coords['N'],
                    current_coords['CA'],
                    current_coords['C'],
                    current_coords['O']
                ]
                coords.append(atom_coords)
                plddt_scores.append(current_plddt)
                residue_types.append(current_resname)

            # Reset for new residue
            current_coords = {}

        # Update current residue info
        current_residue = residue_id
        current_coords[atom_name] = [x, y, z]
        current_plddt = temp_factor
        current_resname = res_name

    # Don't forget the last residue
    if current_residue is not None and len(current_coords) == 4:
        atom_coords = [
            current_coords['N'],
            current_coords['CA'],
            current_coords['C'],
            current_coords['O']
        ]
        coords.append(atom_coords)
        plddt_scores.append(current_plddt)
        residue_types.append(current_resname)

    if not coords:
        raise ValueError(f"No complete backbone residues found in {cif_path}")

    # Convert to tensors
    coords = torch.tensor(coords, dtype=torch.float)      # [L, 4, 3]
    plddt_scores = torch.tensor(plddt_scores, dtype=torch.float)  # [L]

    return coords, plddt_scores, residue_types


def detect_file_type(file_path: str) -> str:
    """
    Detects whether a CIF file is from AlphaFold2 or a traditional PDB source.

    Args:
        file_path: Path to the CIF file.

    Returns:
        'alphafold2' if the file appears to be from AlphaFold2, 'pdb' otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            first_lines = [f.readline() for _ in range(50)]

        # Check for AlphaFold2 indicators
        for line in first_lines:
            if 'AlphaFold' in line or 'AF-' in line:
                return 'alphafold2'
            if 'data_AF-' in line:  # AlphaFold2 data block
                return 'alphafold2'

        # Check if file uses ATOM records (AlphaFold2 format) vs mmCIF format
        for line in first_lines:
            if line.startswith('ATOM'):
                return 'alphafold2'
            if line.startswith('_atom_site'):
                return 'pdb'

        # Default to PDB if we can't determine
        return 'pdb'

    except Exception:
        # Default to PDB if we can't read the file
        return 'pdb'


def parse_cif_backbone_auto(cif_path: str):
    """
    Automatically detects the file type and parses accordingly.

    Args:
        cif_path: The file path to the CIF file.

    Returns:
        A tuple containing:
        - coords (torch.Tensor): Backbone coordinates [L, 4, 3]
        - scores (torch.Tensor): B-factors (PDB) or pLDDT scores (AlphaFold2) [L]
        - residue_types (List[str]): Residue names
        - source (str): 'pdb' or 'alphafold2'
    """
    file_type = detect_file_type(cif_path)

    if file_type == 'alphafold2':
        coords, scores, residue_types = parse_alphafold2_cif(cif_path)
        return coords, scores, residue_types, 'alphafold2'
    else:
        coords, scores, residue_types = parse_cif_backbone(cif_path)
        return coords, scores, residue_types, 'pdb'

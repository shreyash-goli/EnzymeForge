"""
Utility functions for EnzymeForge

Provides PDB manipulation, subprocess handling, and other utilities.
"""

from enzymeforge.utils.pdb_utils import (
    get_motifs,
    get_ligand_index,
    get_ca_rmsd,
    get_motif_ca_rmsd,
    get_motif_all_atom_rmsd,
    superimpose_motif_all_atom,
    add_sidechain_coordinates,
    add_ligand_coordinates,
    add_sidechain_and_ligand_coordinates,
    remove_chain_from_pdb,
    get_ligand_residue,
    get_ligand_residue_from_path,
    get_close_backbone_atoms,
    residue_dist_to_ligand,
    get_close_protein_atoms,
    atom_sort_key,
)

from enzymeforge.utils.process import (
    run_command,
    run_parallel,
    check_executable_exists,
    check_path_exists,
)

__all__ = [
    # PDB utilities
    "get_motifs",
    "get_ligand_index",
    "get_ca_rmsd",
    "get_motif_ca_rmsd",
    "get_motif_all_atom_rmsd",
    "superimpose_motif_all_atom",
    "add_sidechain_coordinates",
    "add_ligand_coordinates",
    "add_sidechain_and_ligand_coordinates",
    "remove_chain_from_pdb",
    "get_ligand_residue",
    "get_ligand_residue_from_path",
    "get_close_backbone_atoms",
    "residue_dist_to_ligand",
    "get_close_protein_atoms",
    "atom_sort_key",
    # Process utilities
    "run_command",
    "run_parallel",
    "check_executable_exists",
    "check_path_exists",
]

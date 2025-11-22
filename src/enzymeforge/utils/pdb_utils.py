"""
PDB manipulation utilities for EnzymeForge

Adapted from ProtDesign2 with improvements:
- Type hints throughout
- pathlib for path handling
- Better error handling and logging
- Comprehensive docstrings
"""

import copy
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
from Bio.PDB import PDBParser, Superimposer, PDBIO, Chain, Structure, Model, Residue, Atom

logger = logging.getLogger(__name__)


def atom_sort_key(atom: Atom) -> str:
    """Sort key for atoms in PDB file

    Args:
        atom: BioPython Atom object

    Returns:
        Atom ID for sorting
    """
    return atom.id


def get_motifs(contig_str: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse RFdiffusion contig string to extract motif information

    Contig format: "N-term/MotifChainStart-End/C-term"
    Example: "50-50/A45-47/50-50" means:
    - 50 residues N-terminal
    - Fixed motif from chain A, residues 45-47
    - 50 residues C-terminal

    Args:
        contig_str: RFdiffusion contig string (e.g., "50-50/A45-47/50-50")

    Returns:
        Tuple of (design_motif, ref_motif, redesigned_residues)
        - design_motif: Residue IDs in designed structure (e.g., ["A75", "A76", "A77"])
        - ref_motif: Residue IDs in reference structure (e.g., ["A45", "A46", "A47"])
        - redesigned_residues: All residues NOT in motif (e.g., ["A1", "A2", ..., "A74", "A78", ...])

    Example:
        >>> get_motifs("50-50/A45-47/50-50")
        (["A51", "A52", "A53"], ["A45", "A46", "A47"], ["A1", ..., "A50", "A54", ..., "A150"])
    """
    design_motif = []
    ref_motif = []
    all_residues = []
    idx = 0

    # Parse each block in contig string
    for block in contig_str.split(" ")[0].split("/"):
        if block == "0":
            # Ignore chain breaks
            continue

        if block[0].isalpha():
            # This is a fixed motif block (e.g., "A45-47")
            chain_id = block[0]
            lb = int(block.split("-")[0][1:])  # Lower bound
            ub = int(block.split("-")[1])       # Upper bound
            diff = ub - lb + 1

            for i in range(diff):
                idx += 1
                ref_motif.append(f"{chain_id}{lb + i}")
                design_motif.append(f"A{idx}")
        else:
            # This is a de novo block (e.g., "50-50")
            idx = idx + int(block.split("-")[0])

    # Generate list of all residues
    for i in range(idx):
        all_residues.append(f"A{i+1}")

    # Redesigned residues = all residues - fixed motif
    redesigned_residues = list(set(all_residues) - set(design_motif))

    return design_motif, ref_motif, redesigned_residues


def get_ligand_index(pdb_file: Path, ligand_name: str) -> Optional[int]:
    """Get residue index of ligand in PDB file

    Args:
        pdb_file: Path to PDB file
        ligand_name: 3-letter ligand code (e.g., "FOA" for PFOA)

    Returns:
        Residue index of ligand, or None if not found
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', str(pdb_file))
    chain = structure[0]["B"]  # Ligand is typically in chain B

    for residue in chain.get_residues():
        if residue.get_resname() == ligand_name:
            return residue.get_id()[1]

    logger.warning(f"Ligand {ligand_name} not found in {pdb_file}")
    return None


def get_ca_rmsd(design_path: Path, ref_path: Path) -> float:
    """Calculate CA-RMSD between two structures

    Args:
        design_path: Path to design PDB file
        ref_path: Path to reference PDB file

    Returns:
        CA-RMSD in Angstroms (rounded to 2 decimals)
    """
    parser = PDBParser(QUIET=True)
    design_structure = parser.get_structure('design', str(design_path))[0]['A']
    ref_structure = parser.get_structure('ref', str(ref_path))[0]['A']

    design_ca_atoms = [atom for atom in design_structure.get_atoms() if atom.get_id() == "CA"]
    ref_ca_atoms = [atom for atom in ref_structure.get_atoms() if atom.get_id() == "CA"]

    super_imposer = Superimposer()
    super_imposer.set_atoms(design_ca_atoms, ref_ca_atoms)

    return round(super_imposer.rms, 2)


def get_motif_ca_rmsd(
    design_path: Path,
    ref_path: Path,
    design_motif: List[str],
    ref_motif: List[str]
) -> float:
    """Calculate CA-RMSD for catalytic motif only

    This is the key quality metric - checks if the catalytic residues
    are positioned correctly in the designed structure.

    Args:
        design_path: Path to design PDB file
        ref_path: Path to reference PDB file
        design_motif: Motif residues in design (e.g., ["A51", "A52", "A53"])
        ref_motif: Motif residues in reference (e.g., ["A45", "A46", "A47"])

    Returns:
        Motif CA-RMSD in Angstroms (rounded to 2 decimals)

    Example:
        >>> rmsd = get_motif_ca_rmsd(
        ...     design_path="design_0.pdb",
        ...     ref_path="input.pdb",
        ...     design_motif=["A51", "A52", "A53"],
        ...     ref_motif=["A45", "A46", "A47"]
        ... )
        >>> rmsd < 2.0  # Good design
        True
    """
    parser = PDBParser(QUIET=True)
    design_structure = parser.get_structure('design', str(design_path))[0]['A']
    ref_model = parser.get_structure('ref', str(ref_path))[0]

    ref_atoms = []
    design_atoms = []

    # Extract CA atoms from reference motif
    for resi in ref_motif:
        ref_chain = resi[0]
        ref_atom_list = [
            atom for atom in ref_model[ref_chain][int(resi[1:])].get_atoms()
            if atom.get_id() == 'CA'
        ]
        ref_atoms.extend(ref_atom_list)

    # Extract CA atoms from design motif
    for resi in design_motif:
        design_atom_list = [
            atom for atom in design_structure[int(resi[1:])].get_atoms()
            if atom.get_id() == 'CA'
        ]
        design_atoms.extend(design_atom_list)

    super_imposer = Superimposer()
    super_imposer.set_atoms(design_atoms, ref_atoms)

    return round(super_imposer.rms, 2)


def superimpose_motif_all_atom(
    design_model: Model,
    ref_model: Model,
    design_motif: List[str],
    ref_motif: List[str],
    apply_change: bool
) -> Union[float, Model]:
    """Superimpose structures based on all-atom motif alignment

    Args:
        design_model: Design structure model
        ref_model: Reference structure model
        design_motif: Motif residues in design
        ref_motif: Motif residues in reference
        apply_change: If True, apply transformation to ref_model and return it
                     If False, just return RMSD

    Returns:
        If apply_change=False: All-atom RMSD (float)
        If apply_change=True: Transformed ref_model (Model)
    """
    design_structure = design_model['A']
    ref_atoms = []
    design_atoms = []

    # Extract non-hydrogen atoms from reference motif
    for resi in ref_motif:
        ref_chain = resi[0]
        ref_atom_list = [
            atom for atom in ref_model[ref_chain][int(resi[1:])].get_atoms()
            if atom.element != 'H'
        ]
        ref_atom_list.sort(key=atom_sort_key)
        ref_atoms.extend(ref_atom_list)

    # Extract non-hydrogen atoms from design motif
    for resi in design_motif:
        design_atom_list = [
            atom for atom in design_structure[int(resi[1:])].get_atoms()
            if atom.element != 'H'
        ]
        design_atom_list.sort(key=atom_sort_key)
        design_atoms.extend(design_atom_list)

    super_imposer = Superimposer()
    super_imposer.set_atoms(design_atoms, ref_atoms)

    if not apply_change:
        return round(super_imposer.rms, 2)
    else:
        super_imposer.apply(ref_model.get_atoms())
        return ref_model


def get_motif_all_atom_rmsd(
    design_path: Path,
    ref_path: Path,
    design_motif: List[str],
    ref_motif: List[str]
) -> float:
    """Calculate all-atom RMSD for catalytic motif

    More stringent than CA-RMSD - checks if sidechain geometry matches.

    Args:
        design_path: Path to design PDB file
        ref_path: Path to reference PDB file
        design_motif: Motif residues in design
        ref_motif: Motif residues in reference

    Returns:
        All-atom motif RMSD in Angstroms (rounded to 2 decimals)
    """
    parser = PDBParser(QUIET=True)
    design_model = parser.get_structure('design', str(design_path))[0]
    ref_model = parser.get_structure('ref', str(ref_path))[0]

    return superimpose_motif_all_atom(
        design_model=design_model,
        ref_model=ref_model,
        design_motif=design_motif,
        ref_motif=ref_motif,
        apply_change=False
    )


def add_sidechain_coordinates(
    design_model: Model,
    ref_model: Model,
    design_resi: str,
    ref_resi: str
) -> Model:
    """Add sidechain coordinates from reference to design for a single residue

    Superimposes backbone atoms (CA, N, C, O), then transfers sidechain atoms.

    Args:
        design_model: Design structure (backbone only)
        ref_model: Reference structure (with sidechains)
        design_resi: Residue ID in design (e.g., "A51")
        ref_resi: Residue ID in reference (e.g., "A45")

    Returns:
        Updated design model with sidechain added
    """
    ref_chain = ref_resi[0]
    bb_atoms = ["CA", "N", "C", "O"]

    # Get backbone atoms
    design_atoms = [
        atom for atom in design_model["A"][int(design_resi[1:])].get_atoms()
        if atom.get_id() in bb_atoms
    ]
    ref_atoms = [
        atom for atom in ref_model[ref_chain][int(ref_resi[1:])].get_atoms()
        if atom.get_id() in bb_atoms
    ]

    design_atoms.sort(key=atom_sort_key)
    ref_atoms.sort(key=atom_sort_key)

    # Superimpose backbones
    super_imposer = Superimposer()
    super_imposer.set_atoms(design_atoms, ref_atoms)

    # Get sidechain atoms (non-backbone, non-hydrogen)
    sidechain_atoms = [
        atom for atom in ref_model[ref_chain][int(ref_resi[1:])].get_atoms()
        if (atom.get_id() not in bb_atoms) and (atom.element != 'H')
    ]

    # Transform sidechains to match design backbone
    super_imposer.apply(sidechain_atoms)

    # Add sidechain atoms to design
    for atom in sidechain_atoms:
        design_model["A"][int(design_resi[1:])].add(atom)

    return design_model


def get_ligand_residue(model: Model, ligand_name: str) -> Optional[Residue]:
    """Get ligand residue from model

    Args:
        model: BioPython Model object
        ligand_name: 3-letter ligand code

    Returns:
        Ligand Residue object, or None if not found
    """
    logger.info(f"Searching for ligand: {ligand_name}")

    for chain in model:
        logger.debug(f"Checking chain {chain.id}")
        for residue in chain:
            if residue.resname == ligand_name:
                logger.info(f"Ligand {ligand_name} found in Chain {chain.id}, Residue {residue.id}")
                return residue

    logger.warning(f"Ligand {ligand_name} not found in the PDB file")
    return None


def get_ligand_residue_from_path(path: Path, ligand_name: str) -> Optional[Residue]:
    """Get ligand residue from PDB file path

    Args:
        path: Path to PDB file
        ligand_name: 3-letter ligand code

    Returns:
        Ligand Residue object, or None if not found
    """
    parser = PDBParser(QUIET=True)
    model = parser.get_structure('structure', str(path))[0]
    return get_ligand_residue(model=model, ligand_name=ligand_name)


def add_ligand_coordinates(
    design_model: Model,
    ref_model: Model,
    design_motif: List[str],
    ref_motif: List[str],
    ligand_name: str
) -> Model:
    """Add ligand coordinates from reference to design

    Superimposes structures on motif, then transfers ligand.

    Args:
        design_model: Design structure (no ligand)
        ref_model: Reference structure (with ligand)
        design_motif: Motif residues in design
        ref_motif: Motif residues in reference
        ligand_name: 3-letter ligand code

    Returns:
        Updated design model with ligand in chain B
    """
    # Superimpose reference onto design
    ref_model = superimpose_motif_all_atom(
        design_model=design_model,
        ref_model=ref_model,
        design_motif=design_motif,
        ref_motif=ref_motif,
        apply_change=True
    )

    # Get ligand from reference
    ligand_residue = get_ligand_residue(model=ref_model, ligand_name=ligand_name)

    if ligand_residue is None:
        logger.error(f"Cannot add ligand - {ligand_name} not found in reference")
        return design_model

    # Add ligand to chain B
    new_chain = Chain.Chain("B")
    design_model.add(new_chain)
    design_model["B"].add(ligand_residue)

    return design_model


def add_sidechain_and_ligand_coordinates(
    design_path: Path,
    ref_path: Path,
    design_motif: List[str],
    ref_motif: List[str],
    ligand_name: str
) -> None:
    """Add motif sidechains and ligand to RFdiffusion backbone

    This is the key postprocessing step after RFdiffusion:
    1. Adds sidechains to motif residues (from reference)
    2. Adds ligand positioned relative to motif
    3. Saves updated structure back to design_path

    Args:
        design_path: Path to design PDB (backbone only) - will be overwritten
        ref_path: Path to reference PDB (with sidechains and ligand)
        design_motif: Motif residues in design
        ref_motif: Motif residues in reference
        ligand_name: 3-letter ligand code

    Example:
        >>> add_sidechain_and_ligand_coordinates(
        ...     design_path="design_0.pdb",
        ...     ref_path="input.pdb",
        ...     design_motif=["A51", "A52", "A53"],
        ...     ref_motif=["A45", "A46", "A47"],
        ...     ligand_name="FOA"
        ... )
    """
    io = PDBIO()
    parser = PDBParser(QUIET=True)
    design_model = parser.get_structure('design', str(design_path))[0]
    ref_model = parser.get_structure('ref', str(ref_path))[0]

    # Add sidechains for each motif residue
    for i, design_resi in enumerate(design_motif):
        design_model = add_sidechain_coordinates(
            design_model=design_model,
            ref_model=copy.deepcopy(ref_model),
            design_resi=design_resi,
            ref_resi=ref_motif[i]
        )

    # Add ligand
    design_model = add_ligand_coordinates(
        design_model=design_model,
        ref_model=copy.deepcopy(ref_model),
        design_motif=design_motif,
        ref_motif=ref_motif,
        ligand_name=ligand_name
    )

    # Save updated structure
    io.set_structure(design_model)
    io.save(str(design_path))
    logger.info(f"Added sidechains and ligand to {design_path}")


def remove_chain_from_pdb(design_path: Path, chain_to_remove: str) -> None:
    """Remove a chain from PDB file

    Args:
        design_path: Path to PDB file (will be overwritten)
        chain_to_remove: Chain ID to remove (e.g., "B")
    """
    parser = PDBParser(QUIET=True)
    design_model = parser.get_structure('design', str(design_path))[0]
    new_model = design_model.copy()

    for chain in list(new_model):
        logger.debug(f"Checking chain {chain.id}")
        if chain.id == chain_to_remove:
            logger.info(f"Removing chain {chain_to_remove}")
            new_model.detach_child(chain_to_remove)
            break

    # Save updated structure
    io = PDBIO()
    io.set_structure(new_model)
    io.save(str(design_path))


def residue_dist_to_ligand(
    protein_residue: Residue,
    ligand_residue: Residue,
    atomtype: str = 'CA'
) -> Optional[float]:
    """Calculate minimum distance from protein residue to ligand

    Args:
        protein_residue: Protein residue
        ligand_residue: Ligand residue
        atomtype: Atom type to measure from (default: "CA")

    Returns:
        Minimum distance in Angstroms, or None if atom not found
    """
    distances = []

    for atom in ligand_residue:
        if atomtype in protein_residue:
            vector = protein_residue[atomtype].coord - atom.coord
            distances.append(np.sqrt(np.sum(vector * vector)))

    if len(distances) > 0:
        return min(distances)
    else:
        return None


def get_close_protein_atoms(
    ligand: Residue,
    distance: float,
    model: Model,
    atom_list: List[str]
) -> List[Tuple[str, int, str, float]]:
    """Get protein atoms within distance threshold of ligand

    Args:
        ligand: Ligand residue
        distance: Distance threshold in Angstroms
        model: Structure model
        atom_list: Atom types to check (e.g., ["CA", "C", "O", "N"])

    Returns:
        List of (resname, resid, atomtype, distance) tuples
    """
    close_atoms = []
    chains = model.child_dict

    for chain_id in chains:
        for protein_res in chains[chain_id].child_list:
            if not protein_res.resname == ligand.resname:
                for atom in atom_list:
                    dist = residue_dist_to_ligand(protein_res, ligand, atom)
                    if dist and dist < distance:
                        close_atoms.append((protein_res.resname, protein_res.id[1], atom, dist))

    return close_atoms


def get_close_backbone_atoms(
    ligand: Residue,
    distance: float,
    design_path: Path
) -> List[Tuple[str, int, str, float]]:
    """Get backbone atoms within distance threshold of ligand

    Used for clash detection - designs with backbone clashes are rejected.

    Args:
        ligand: Ligand residue
        distance: Distance threshold in Angstroms (typically 2.0 for clashes)
        design_path: Path to design PDB file

    Returns:
        List of (resname, resid, atomtype, distance) tuples for clashing atoms

    Example:
        >>> ligand = get_ligand_residue_from_path("design_0.pdb", "FOA")
        >>> clashes = get_close_backbone_atoms(ligand, 2.0, "design_0.pdb")
        >>> len(clashes) == 0  # Good design has no clashes
        True
    """
    parser = PDBParser(QUIET=True)
    design_model = parser.get_structure('design', str(design_path))[0]

    close_backbone_atoms = get_close_protein_atoms(
        ligand=ligand,
        distance=distance,
        model=design_model,
        atom_list=["CA", "C", "O", "N"]
    )

    return close_backbone_atoms

"""
Structure validation module

Validates designed structures for quality metrics
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from enzymeforge.utils.pdb_utils import (
    get_motif_ca_rmsd,
    get_motif_all_atom_rmsd,
    get_close_backbone_atoms,
    get_ligand_residue_from_path
)

logger = logging.getLogger(__name__)


class StructureValidator:
    """Validate designed structures for quality metrics

    Checks:
    - Motif RMSD (are catalytic residues positioned correctly?)
    - Backbone clashes with ligand (is substrate binding pocket valid?)
    """

    def __init__(
        self,
        rmsd_cutoff: float = 2.0,
        clash_cutoff: float = 2.0
    ):
        """Initialize validator

        Args:
            rmsd_cutoff: Maximum allowed motif CA-RMSD (Angstroms)
            clash_cutoff: Minimum allowed distance for backbone-ligand (Angstroms)
        """
        self.rmsd_cutoff = rmsd_cutoff
        self.clash_cutoff = clash_cutoff

    def check_motif_rmsd(
        self,
        design_pdb: Path,
        reference_pdb: Path,
        design_motif: List[str],
        ref_motif: List[str],
        all_atom: bool = False
    ) -> float:
        """Calculate RMSD for catalytic motif

        Args:
            design_pdb: Designed structure
            reference_pdb: Reference structure (substrate complex)
            design_motif: Motif residues in design (e.g., ["A51", "A52", "A53"])
            ref_motif: Motif residues in reference (e.g., ["A45", "A46", "A47"])
            all_atom: If True, use all-atom RMSD. If False, use CA-only (faster)

        Returns:
            RMSD in Angstroms

        Example:
            >>> validator = StructureValidator(rmsd_cutoff=2.0)
            >>> rmsd = validator.check_motif_rmsd(
            ...     design_pdb=Path("design_0.pdb"),
            ...     reference_pdb=Path("input.pdb"),
            ...     design_motif=["A51", "A52", "A53"],
            ...     ref_motif=["A45", "A46", "A47"]
            ... )
            >>> rmsd < 2.0  # Good design
            True
        """
        if all_atom:
            rmsd = get_motif_all_atom_rmsd(
                design_path=design_pdb,
                ref_path=reference_pdb,
                design_motif=design_motif,
                ref_motif=ref_motif
            )
        else:
            rmsd = get_motif_ca_rmsd(
                design_path=design_pdb,
                ref_path=reference_pdb,
                design_motif=design_motif,
                ref_motif=ref_motif
            )

        logger.debug(f"Motif RMSD for {design_pdb.name}: {rmsd} Å")
        return rmsd

    def check_clashes(
        self,
        pdb_path: Path,
        ligand_name: str,
        distance_cutoff: Optional[float] = None
    ) -> List[Tuple[str, int, str, float]]:
        """Check for backbone clashes with ligand

        Args:
            pdb_path: PDB file to check
            ligand_name: 3-letter ligand code (e.g., "FOA")
            distance_cutoff: Min distance (Å). If None, uses self.clash_cutoff

        Returns:
            List of (resname, resid, atomtype, distance) tuples for clashing atoms

        Example:
            >>> validator = StructureValidator(clash_cutoff=2.0)
            >>> clashes = validator.check_clashes(
            ...     pdb_path=Path("design_0.pdb"),
            ...     ligand_name="FOA"
            ... )
            >>> len(clashes) == 0  # Good design has no clashes
            True
        """
        if distance_cutoff is None:
            distance_cutoff = self.clash_cutoff

        # Get ligand from structure
        ligand = get_ligand_residue_from_path(pdb_path, ligand_name)

        if ligand is None:
            logger.warning(f"Ligand {ligand_name} not found in {pdb_path}, cannot check clashes")
            return []

        # Check for backbone atoms too close to ligand
        clashes = get_close_backbone_atoms(
            ligand=ligand,
            distance=distance_cutoff,
            design_path=pdb_path
        )

        if clashes:
            logger.warning(f"Found {len(clashes)} backbone-ligand clashes in {pdb_path.name}")
            for resname, resid, atomtype, dist in clashes:
                logger.debug(f"  {resname}{resid} {atomtype}: {dist:.2f} Å")
        else:
            logger.debug(f"No backbone clashes in {pdb_path.name}")

        return clashes

    def validate_design(
        self,
        design_pdb: Path,
        reference_pdb: Path,
        design_motif: List[str],
        ref_motif: List[str],
        ligand_name: str
    ) -> Tuple[bool, Dict]:
        """Run all validation checks on a design

        Args:
            design_pdb: Designed structure
            reference_pdb: Reference structure
            design_motif: Motif residues in design
            ref_motif: Motif residues in reference
            ligand_name: 3-letter ligand code

        Returns:
            Tuple of (passes_validation, metrics_dict)
            - passes_validation: True if design passes all checks
            - metrics_dict: Dictionary with validation metrics

        Example:
            >>> validator = StructureValidator(rmsd_cutoff=2.0, clash_cutoff=2.0)
            >>> passes, metrics = validator.validate_design(
            ...     design_pdb=Path("design_0.pdb"),
            ...     reference_pdb=Path("input.pdb"),
            ...     design_motif=["A51", "A52", "A53"],
            ...     ref_motif=["A45", "A46", "A47"],
            ...     ligand_name="FOA"
            ... )
            >>> if passes:
            ...     print(f"Good design! RMSD: {metrics['rmsd']:.2f}")
        """
        metrics = {}

        # Check motif RMSD
        rmsd = self.check_motif_rmsd(
            design_pdb=design_pdb,
            reference_pdb=reference_pdb,
            design_motif=design_motif,
            ref_motif=ref_motif
        )
        metrics['rmsd'] = rmsd

        # Check clashes
        clashes = self.check_clashes(
            pdb_path=design_pdb,
            ligand_name=ligand_name
        )
        metrics['num_clashes'] = len(clashes)
        metrics['clashes'] = clashes

        # Determine if design passes
        passes_rmsd = rmsd <= self.rmsd_cutoff
        passes_clashes = len(clashes) == 0
        passes_validation = passes_rmsd and passes_clashes

        metrics['passes_rmsd'] = passes_rmsd
        metrics['passes_clashes'] = passes_clashes
        metrics['passes_validation'] = passes_validation

        if passes_validation:
            logger.info(f"✓ {design_pdb.name} passes validation (RMSD: {rmsd:.2f} Å)")
        else:
            logger.warning(f"✗ {design_pdb.name} fails validation:")
            if not passes_rmsd:
                logger.warning(f"  RMSD {rmsd:.2f} > {self.rmsd_cutoff:.2f} Å")
            if not passes_clashes:
                logger.warning(f"  {len(clashes)} backbone clashes")

        return passes_validation, metrics

    def filter_designs(
        self,
        design_pdbs: List[Path],
        reference_pdb: Path,
        design_motif: List[str],
        ref_motif: List[str],
        ligand_name: str
    ) -> Tuple[List[Path], List[Dict]]:
        """Filter list of designs, keeping only those that pass validation

        Args:
            design_pdbs: List of design PDB files
            reference_pdb: Reference structure
            design_motif: Motif residues in design
            ref_motif: Motif residues in reference
            ligand_name: 3-letter ligand code

        Returns:
            Tuple of (passing_designs, all_metrics)
            - passing_designs: List of PDB paths that pass validation
            - all_metrics: List of metrics dicts for all designs

        Example:
            >>> validator = StructureValidator(rmsd_cutoff=2.0, clash_cutoff=2.0)
            >>> designs = [Path(f"design_{i}.pdb") for i in range(10)]
            >>> passing, metrics = validator.filter_designs(
            ...     design_pdbs=designs,
            ...     reference_pdb=Path("input.pdb"),
            ...     design_motif=["A51", "A52", "A53"],
            ...     ref_motif=["A45", "A46", "A47"],
            ...     ligand_name="FOA"
            ... )
            >>> print(f"{len(passing)}/{len(designs)} designs passed")
        """
        logger.info(f"Filtering {len(design_pdbs)} designs...")

        passing_designs = []
        all_metrics = []

        for design_pdb in design_pdbs:
            passes, metrics = self.validate_design(
                design_pdb=design_pdb,
                reference_pdb=reference_pdb,
                design_motif=design_motif,
                ref_motif=ref_motif,
                ligand_name=ligand_name
            )

            all_metrics.append(metrics)

            if passes:
                passing_designs.append(design_pdb)

        logger.info(f"Filtering complete: {len(passing_designs)}/{len(design_pdbs)} designs passed")

        return passing_designs, all_metrics

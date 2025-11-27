"""
Rosetta FastRelax runner for EnzymeForge

Handles iterative cycles of FastRelax + sequence design.
Adapted from ProtDesign2 with improvements.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from enzymeforge.utils.process import run_command, check_executable_exists

logger = logging.getLogger(__name__)


@dataclass
class RelaxConfig:
    """Rosetta FastRelax configuration

    Attributes:
        params_file: Rosetta params file for ligand (optional)
        cst_file: Constraint file for Rosetta (optional)
        num_workers: Parallel workers for relaxation (default: 8)
        constrain_to_start: Constrain to starting coordinates (default: True)
        use_pdb2pqr: Protonate structures before relaxation (default: True)
    """
    params_file: Optional[Path] = None
    cst_file: Optional[Path] = None
    num_workers: int = 8
    constrain_to_start: bool = True
    use_pdb2pqr: bool = True


@dataclass
class RelaxResult:
    """Result from Rosetta FastRelax"""
    design_id: str
    pdb_path: Path
    energy_before: float
    energy_after: float
    energy_delta: float
    metadata: Dict = field(default_factory=dict)


class RosettaRunner:
    """Run Rosetta FastRelax for structure refinement

    Supports:
    - FastRelax with constraints
    - Ligand parameter files
    - Structure protonation
    - Parallel execution
    - Integration with LigandMPNN
    """

    def __init__(self, check_dependencies: bool = True):
        """Initialize Rosetta runner

        Args:
            check_dependencies: Check for PyRosetta and pdb2pqr (default: True)
        """
        self.pyrosetta_available = False
        self.pdb2pqr_available = False

        if check_dependencies:
            # Check PyRosetta
            try:
                import pyrosetta
                self.pyrosetta_available = True
                logger.info("PyRosetta detected")
            except ImportError:
                logger.warning("PyRosetta not found - FastRelax will not be available")
                logger.warning("Install with: pip install pyrosetta")

            # Check pdb2pqr
            if check_executable_exists("pdb2pqr"):
                self.pdb2pqr_available = True
                logger.info("pdb2pqr detected")
            else:
                logger.warning("pdb2pqr not found - protonation will be skipped")
                logger.warning("Install with: conda install -c conda-forge pdb2pqr")

    def run_relax_cycle(
        self,
        pdb_files: List[Path],
        config: RelaxConfig,
        design_motif: List[str],
        ref_motif: List[str],
        ligand_name: str,
        output_dir: Path,
        cycle_number: int
    ) -> List[RelaxResult]:
        """Run one cycle of FastRelax on structures

        Args:
            pdb_files: List of PDB files to relax
            config: Relaxation configuration
            design_motif: Motif residues in design (e.g., ["A51", "A52"])
            ref_motif: Motif residues in reference (e.g., ["A45", "A46"])
            ligand_name: Three-letter ligand code
            output_dir: Output directory for relaxed structures
            cycle_number: Cycle number for naming

        Returns:
            List of RelaxResult objects
        """
        if not self.pyrosetta_available:
            raise RuntimeError("PyRosetta not available - cannot run FastRelax")

        logger.info(f"Starting FastRelax cycle {cycle_number}")
        logger.info(f"Relaxing {len(pdb_files)} structures")

        # Create output directory
        relaxed_dir = output_dir / "relaxed"
        relaxed_dir.mkdir(parents=True, exist_ok=True)

        results = []

        # Run in parallel
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            futures = [
                executor.submit(
                    self._relax_single_structure,
                    pdb_file,
                    config,
                    design_motif,
                    ref_motif,
                    ligand_name,
                    relaxed_dir,
                    cycle_number
                )
                for pdb_file in pdb_files
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Relaxed {result.design_id}: ΔE = {result.energy_delta:.2f}")
                except Exception as e:
                    logger.error(f"Relaxation failed: {e}")

        logger.info(f"Completed FastRelax cycle {cycle_number}: {len(results)} structures")
        return results

    def _relax_single_structure(
        self,
        pdb_file: Path,
        config: RelaxConfig,
        design_motif: List[str],
        ref_motif: List[str],
        ligand_name: str,
        output_dir: Path,
        cycle_number: int
    ) -> RelaxResult:
        """Relax a single structure with Rosetta FastRelax

        Adapted from ProtDesign2's relax() function.
        """
        import pyrosetta as pyr

        # Extract design ID
        design_id = pdb_file.stem
        if "_packed" in design_id:
            design_id = design_id.split("_packed")[0]

        # Add cycle number to ID
        if "_c" in design_id:
            index = design_id.index("_c")
            design_id = design_id[:index+2] + str(cycle_number)
        else:
            # Extract sequence number from packed filename
            seq_num = pdb_file.name.split("packed_")[1].split(".pdb")[0]
            design_id = f"{design_id}_n{seq_num}_c{cycle_number}"

        # Protonate structure if requested
        if config.use_pdb2pqr and self.pdb2pqr_available:
            pdb_file = self._protonate_structure(pdb_file)

        # Create design-specific constraint file
        cst_file = None
        if config.cst_file:
            cst_file = self._create_cst_file(
                design_motif,
                ref_motif,
                config.cst_file,
                pdb_file,
                ligand_name,
                output_dir,
                design_id
            )

        # Initialize PyRosetta
        extra_res = f"-extra_res_fa {config.params_file}" if config.params_file else ""
        constraints = f"-constraints:cst_fa_file {cst_file}" if cst_file else ""
        pyr.init(f"-ignore_zero_occupancy false -ex1 -ex2 {extra_res} {constraints}")

        # Define score function
        scorefxn = pyr.get_fa_scorefxn()
        scorefxn.set_weight(pyr.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1)
        scorefxn.set_weight(pyr.rosetta.core.scoring.score_type_from_name("coordinate_constraint"), 1)

        # Load structure
        pose = pyr.pose_from_file(str(pdb_file))
        pose2 = pose.clone()

        # Add constraints
        if cst_file:
            constraint_mover = pyr.rosetta.protocols.constraint_movers.ConstraintSetMover()
            constraint_mover.constraint_file(str(cst_file))
            constraint_mover.apply(pose2)

        # Set up task factory (prevent repacking of motif and ligand)
        tf = pyr.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyr.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyr.rosetta.core.pack.task.operation.RestrictToRepacking())
        tf.push_back(pyr.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyr.rosetta.core.pack.task.operation.NoRepackDisulfides())

        # Prevent repacking of catalytic residues and ligand
        residue_selector = pyr.rosetta.core.select.residue_selector.ResidueIndexSelector()
        for resi in design_motif:
            residue = int(resi[1:])
            residue_selector.append_index(residue)
        residue_selector.apply(pose2)

        chain_selector = pyr.rosetta.core.select.residue_selector.ChainSelector("B")
        chain_selector.apply(pose2)

        no_repacking_selector = pyr.rosetta.core.select.residue_selector.OrResidueSelector()
        no_repacking_selector.add_residue_selector(residue_selector)
        no_repacking_selector.add_residue_selector(chain_selector)
        no_repacking_selector.apply(pose2)

        prevent_repacking_rlt = pyr.rosetta.core.pack.task.operation.PreventRepackingRLT()
        prevent_repacking_op = pyr.rosetta.core.pack.task.operation.OperateOnResidueSubset(
            prevent_repacking_rlt, no_repacking_selector, False
        )
        tf.push_back(prevent_repacking_op)

        # Set up move map (disable minimization of motif and ligand)
        mmf = pyr.rosetta.core.select.movemap.MoveMapFactory()
        mmf.add_chi_action(pyr.rosetta.core.select.movemap.mm_disable, no_repacking_selector)
        mmf.add_bb_action(pyr.rosetta.core.select.movemap.mm_disable, no_repacking_selector)

        # Run FastRelax
        fastRelax = pyr.rosetta.protocols.relax.FastRelax()
        fastRelax.constrain_relax_to_start_coords(config.constrain_to_start)
        fastRelax.set_scorefxn(scorefxn)
        fastRelax.set_movemap_factory(mmf)
        fastRelax.set_task_factory(tf)

        energy_before = scorefxn.score(pose2)
        fastRelax.apply(pose2)
        energy_after = scorefxn.score(pose2)

        # Save relaxed structure
        output_path = output_dir / f"{design_id}.pdb"
        pose2.dump_pdb(str(output_path))

        logger.debug(f"{design_id}: Energy {energy_before:.2f} → {energy_after:.2f} (Δ{energy_after-energy_before:.2f})")

        return RelaxResult(
            design_id=design_id,
            pdb_path=output_path,
            energy_before=energy_before,
            energy_after=energy_after,
            energy_delta=energy_after - energy_before,
            metadata={
                "input_pdb": str(pdb_file),
                "cycle": cycle_number
            }
        )

    def _protonate_structure(self, pdb_file: Path) -> Path:
        """Add hydrogens to structure using pdb2pqr

        Args:
            pdb_file: Input PDB file

        Returns:
            Path to protonated PDB file
        """
        output_path = pdb_file.parent / f"{pdb_file.stem}_prot.pdb"
        pqr_path = pdb_file.parent / f"{pdb_file.stem}_pqr.pdb"

        cmd = f"pdb2pqr --keep-chain --ff=AMBER --pdb-output {output_path} {pdb_file} {pqr_path}"

        returncode, stdout, stderr = run_command(cmd)

        if returncode != 0:
            logger.warning(f"pdb2pqr failed for {pdb_file.name}, using original")
            return pdb_file

        # Clean up pqr file
        pqr_path.unlink(missing_ok=True)

        return output_path

    def _create_cst_file(
        self,
        design_motif: List[str],
        ref_motif: List[str],
        template_cst: Path,
        pdb_file: Path,
        ligand_name: str,
        output_dir: Path,
        design_id: str
    ) -> Path:
        """Create design-specific constraint file

        Maps reference residue numbers to design residue numbers.

        Args:
            design_motif: Motif in design (e.g., ["A51", "A52"])
            ref_motif: Motif in reference (e.g., ["A45", "A46"])
            template_cst: Template constraint file
            pdb_file: PDB file to get ligand index
            ligand_name: Three-letter ligand code
            output_dir: Output directory
            design_id: Design identifier

        Returns:
            Path to created constraint file
        """
        # Convert motif format from A45 to 45A
        design_motif_conv = [resi[1:] + resi[0] for resi in design_motif]
        ref_motif_conv = [resi[1:] + resi[0] for resi in ref_motif]

        # Get ligand index
        ligand_index = self._get_ligand_index(pdb_file, ligand_name)

        # Read template constraints
        with open(template_cst, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            blocks = line.split()
            if len(blocks) < 9:
                continue

            resi1, resi2 = blocks[2].strip(), blocks[4].strip()

            # Map residues
            new_residues = []
            for resi in [resi1, resi2]:
                if resi in ref_motif_conv:
                    index = ref_motif_conv.index(resi)
                    new_resi = design_motif_conv[index]
                else:
                    new_resi = f"{ligand_index}B"
                new_residues.append(new_resi)

            # Rebuild line
            new_line = f"{blocks[0]} {blocks[1]} {new_residues[0]} {blocks[3]} {new_residues[1]} "
            new_line += f"{blocks[5]} {blocks[6]} {blocks[7]} {blocks[8]}\n"
            new_lines.append(new_line)

        # Write new constraint file
        output_path = output_dir / f"{design_id}_constraints.cst"
        with open(output_path, 'w') as f:
            f.writelines(new_lines)

        return output_path

    def _get_ligand_index(self, pdb_file: Path, ligand_name: str) -> int:
        """Get ligand residue index from PDB file

        Args:
            pdb_file: PDB file
            ligand_name: Three-letter ligand code

        Returns:
            Ligand residue index
        """
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("HETATM") and ligand_name in line:
                    # Extract residue number
                    return int(line[22:26].strip())

        logger.warning(f"Ligand {ligand_name} not found in {pdb_file.name}")
        return 1

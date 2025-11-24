"""
RFdiffusion runner for EnzymeForge

Handles RFdiffusion and RFdiffusion all-atom execution.

Adapted from ProtDesign2 with improvements:
- Type hints and dataclasses
- pathlib for path handling
- Better error handling
- Proper logging
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
from glob import glob

from enzymeforge.utils.process import run_command, check_path_exists
from enzymeforge.utils.pdb_utils import (
    get_motifs,
    add_sidechain_and_ligand_coordinates,
    remove_chain_from_pdb
)

logger = logging.getLogger(__name__)


@dataclass
class DiffusionResult:
    """Output from RFdiffusion"""
    design_id: str
    pdb_path: Path
    contig_string: str
    rmsd: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class DiffusionConfig:
    """RFdiffusion parameters

    Attributes:
        contigs: Contig string (e.g., "50-50/A45-47/50-50")
        pdb: Input PDB file path (optional for de novo, required for motif scaffolding)
        substrate: Substrate/ligand name (e.g., "FOA")
        iterations: Number of diffusion steps (default: 50)
        num_designs: Number of designs to generate (default: 10)
        guide_potentials: Guide potential string (optional)
        guide_scale: Scaling factor for guide potential (default: 1.0)
        noise_scale: Noise scale for CA and frame (default: 1.0)
        deterministic: Use deterministic initialization (default: False)
        partial_diffusion: Use partial diffusion (default: False)
        ckpt_override_path: Override checkpoint path (optional)
    """
    contigs: str
    pdb: Optional[Path] = None
    substrate: Optional[str] = None
    iterations: int = 50
    num_designs: int = 10
    guide_potentials: Optional[str] = None
    guide_scale: float = 1.0
    noise_scale: float = 1.0
    deterministic: bool = False
    partial_diffusion: bool = False
    ckpt_override_path: Optional[Path] = None


class RFdiffusionRunner:
    """Run RFdiffusion for protein backbone generation

    Supports both standard RFdiffusion (backbone-only) and RFdiffusion all-atom
    (with ligand).
    """

    def __init__(
        self,
        rfdiffusion_path: Path,
        rfdiffusion_aa_path: Optional[Path] = None
    ):
        """Initialize RFdiffusion runner

        Args:
            rfdiffusion_path: Path to RFdiffusion installation
            rfdiffusion_aa_path: Path to RFdiffusion all-atom installation (optional)
        """
        self.rfdiffusion_path = Path(rfdiffusion_path)
        self.rfdiffusion_aa_path = Path(rfdiffusion_aa_path) if rfdiffusion_aa_path else None

        # Validate paths
        if not check_path_exists(self.rfdiffusion_path, "directory"):
            raise ValueError(f"RFdiffusion path not found: {self.rfdiffusion_path}")

        logger.info(f"Initialized RFdiffusionRunner with path: {self.rfdiffusion_path}")

    def run_diffusion(
        self,
        config: DiffusionConfig,
        name: str,
        output_dir: Path,
        enzyme_design: bool = False
    ) -> List[DiffusionResult]:
        """Run standard RFdiffusion (backbone-only)

        Adapted from ProtDesign2's run_diffusion() function.

        Args:
            config: Diffusion configuration
            name: Experiment name (used for output files)
            output_dir: Directory for output files
            enzyme_design: If True, use guide potentials for enzyme design

        Returns:
            List of DiffusionResult objects for generated designs

        Example:
            >>> runner = RFdiffusionRunner(Path("/path/to/RFdiffusion"))
            >>> config = DiffusionConfig(
            ...     contigs="50-50/A45-47/50-50",
            ...     pdb=Path("input.pdb"),
            ...     substrate="FOA",
            ...     iterations=100,
            ...     num_designs=20
            ... )
            >>> results = runner.run_diffusion(config, "pfas_design", Path("output"))
        """
        # Create output directory
        full_path = output_dir / name / "Diffusion"
        full_path.mkdir(parents=True, exist_ok=True)
        output_prefix = full_path / name

        logger.info(f"Running RFdiffusion: {name}")
        logger.info(f"Output directory: {full_path}")
        logger.info(f"Contigs: {config.contigs}")

        # Build hydra config overrides
        opts = [
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={config.num_designs}",
            f"denoiser.noise_scale_ca={config.noise_scale}",
            f"denoiser.noise_scale_frame={config.noise_scale}",
            f"'contigmap.contigs=[{config.contigs}]'"
        ]

        # Add enzyme design options
        if enzyme_design and config.guide_potentials:
            opts.append(f"potentials.guide_scale={config.guide_scale}")
            opts.append(f"'potentials.guiding_potentials=[\"{config.guide_potentials}\"]'")
            if config.substrate:
                opts.append(f"potentials.substrate={config.substrate}")

        # Add diffusion steps
        if config.partial_diffusion:
            opts.append(f"diffuser.partial_T={config.iterations}")
        else:
            opts.append(f"diffuser.T={config.iterations}")

        # Deterministic run
        if config.deterministic:
            opts.append("inference.deterministic=True")

        # Copy input PDB to output directory
        if config.pdb:
            if not check_path_exists(config.pdb, "file"):
                raise ValueError(f"Input PDB not found: {config.pdb}")

            input_pdb = full_path / "input.pdb"
            shutil.copy(config.pdb, input_pdb)
            opts.append(f"inference.input_pdb={input_pdb}")
            logger.info(f"Using input PDB: {config.pdb}")

        # Override checkpoint if specified
        if config.ckpt_override_path:
            opts.append(f"inference.ckpt_override_path={config.ckpt_override_path}")
            logger.info(f"Using checkpoint: {config.ckpt_override_path}")

        # Run RFdiffusion
        opts_str = " ".join(opts)
        cmd = f"cd {self.rfdiffusion_path} && python3.9 run_inference.py {opts_str}"

        logger.info(f"Executing: {cmd}")
        returncode, stdout, stderr = run_command(cmd, capture_output=False)

        if returncode != 0:
            logger.error(f"RFdiffusion failed with return code {returncode}")
            raise RuntimeError(f"RFdiffusion execution failed")

        # Postprocessing: add sidechains and ligand
        if config.pdb and config.substrate:
            logger.info("Running postprocessing: adding sidechains and ligand")
            self._postprocess_designs(
                contigs=config.contigs,
                name=name,
                output_dir=output_dir,
                ref_pdb=config.pdb,
                substrate=config.substrate
            )

        # Collect results
        results = self._collect_results(full_path, name, config.contigs)
        logger.info(f"Generated {len(results)} designs")

        return results

    def run_diffusion_allatom(
        self,
        config: DiffusionConfig,
        name: str,
        output_dir: Path
    ) -> List[DiffusionResult]:
        """Run RFdiffusion all-atom (with ligand)

        Adapted from ProtDesign2's run_diffusion_aa() function.

        Args:
            config: Diffusion configuration
            name: Experiment name
            output_dir: Directory for output files

        Returns:
            List of DiffusionResult objects

        Note:
            Requires RFdiffusion all-atom installation. The contigs string
            will be reformatted (/ → ,) for all-atom compatibility.
        """
        if not self.rfdiffusion_aa_path:
            raise ValueError("RFdiffusion all-atom path not set")

        if not check_path_exists(self.rfdiffusion_aa_path, "directory"):
            raise ValueError(f"RFdiffusion all-atom path not found: {self.rfdiffusion_aa_path}")

        if not config.pdb:
            raise ValueError("Input PDB required for all-atom diffusion")

        if not config.substrate:
            raise ValueError("Substrate name required for all-atom diffusion")

        # Create output directory
        full_path = output_dir / name / "Diffusion"
        full_path.mkdir(parents=True, exist_ok=True)
        output_prefix = full_path / name

        logger.info(f"Running RFdiffusion all-atom: {name}")
        logger.info(f"Output directory: {full_path}")

        # Preprocess contig string for all-atom (replace / with ,)
        contigs_aa = config.contigs.replace("/", ",")
        logger.info(f"Contigs (all-atom format): {contigs_aa}")

        # Build hydra config overrides
        opts = [
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={config.num_designs}",
            f"denoiser.noise_scale_ca={config.noise_scale}",
            f"denoiser.noise_scale_frame={config.noise_scale}",
            f"diffuser.T={config.iterations}",
            f"inference.ligand={config.substrate}",
            f"contigmap.contigs=[\\'{contigs_aa}\\']"
        ]

        # Deterministic run
        if config.deterministic:
            opts.append("inference.deterministic=True")

        # Copy input PDB
        input_pdb = full_path / "input.pdb"
        shutil.copy(config.pdb, input_pdb)
        opts.append(f"inference.input_pdb={input_pdb}")
        logger.info(f"Using input PDB: {config.pdb}")

        # Run RFdiffusion all-atom
        opts_str = " ".join(opts)
        cmd = f"cd {self.rfdiffusion_aa_path} && python run_inference.py {opts_str}"

        logger.info(f"Executing: {cmd}")
        returncode, stdout, stderr = run_command(cmd, capture_output=False)

        if returncode != 0:
            logger.error(f"RFdiffusion all-atom failed with return code {returncode}")
            raise RuntimeError(f"RFdiffusion all-atom execution failed")

        # Collect results
        results = self._collect_results(full_path, name, config.contigs)
        logger.info(f"Generated {len(results)} designs")

        return results

    def _postprocess_designs(
        self,
        contigs: str,
        name: str,
        output_dir: Path,
        ref_pdb: Path,
        substrate: str
    ) -> None:
        """Add sidechains and ligand to RFdiffusion backbones

        Adapted from ProtDesign2's postprocessing() function.

        Args:
            contigs: Contig string
            name: Experiment name
            output_dir: Output directory
            ref_pdb: Reference PDB with sidechains and ligand
            substrate: Substrate/ligand name
        """
        full_path = output_dir / name / "Diffusion"
        design_motif, ref_motif, _ = get_motifs(contigs)

        # Find all design PDB files
        design_files = list(full_path.glob(f"{name}*.pdb"))
        logger.info(f"Postprocessing {len(design_files)} designs")

        for design_path in design_files:
            # Remove chain B if chain break in contigs
            if "/0" in contigs:
                logger.debug(f"Removing chain B from {design_path.name}")
                remove_chain_from_pdb(design_path, "B")

            # Add sidechains and ligand
            logger.debug(f"Adding sidechains and ligand to {design_path.name}")
            add_sidechain_and_ligand_coordinates(
                design_path=design_path,
                ref_path=ref_pdb,
                design_motif=design_motif,
                ref_motif=ref_motif,
                ligand_name=substrate
            )

        logger.info("Postprocessing complete")

    def _collect_results(
        self,
        output_path: Path,
        name: str,
        contigs: str
    ) -> List[DiffusionResult]:
        """Collect DiffusionResult objects from output directory

        Args:
            output_path: Path to Diffusion output directory
            name: Experiment name
            contigs: Contig string

        Returns:
            List of DiffusionResult objects
        """
        results = []
        design_files = sorted(output_path.glob(f"{name}_*.pdb"))

        for pdb_file in design_files:
            # Extract design ID from filename (e.g., "pfas_0.pdb" -> "pfas_0")
            design_id = pdb_file.stem

            result = DiffusionResult(
                design_id=design_id,
                pdb_path=pdb_file,
                contig_string=contigs,
                metadata={
                    "output_path": str(output_path),
                    "name": name
                }
            )
            results.append(result)

        return results

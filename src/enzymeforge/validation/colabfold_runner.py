"""
ColabFold runner for EnzymeForge

Handles ColabFold structure prediction and metrics calculation.

Adapted from ProtDesign2 with improvements:
- Type hints and dataclasses
- pathlib for path handling
- Better error handling
- Proper logging
- Integrated metrics calculation
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from statistics import mean
from glob import glob as globlib

from enzymeforge.utils.process import run_command, check_executable_exists
from enzymeforge.utils.pdb_utils import (
    get_motifs,
    get_ca_rmsd,
    get_motif_ca_rmsd,
    get_motif_all_atom_rmsd
)

logger = logging.getLogger(__name__)


@dataclass
class FoldingConfig:
    """ColabFold configuration parameters

    Attributes:
        msa_mode: MSA generation mode (default: "single_sequence")
        num_models: Number of models to generate (default: 2)
        num_recycles: Number of recycles (default: 3)
        use_gpu: Use GPU if available (default: True)
        max_msa: Maximum MSA sequences (optional)
    """
    msa_mode: str = "single_sequence"
    num_models: int = 2
    num_recycles: int = 3
    use_gpu: bool = True
    max_msa: Optional[int] = None


@dataclass
class FoldingResult:
    """Output from structure prediction"""
    design_id: str
    sequence: str
    pdb_path: Path
    mean_plddt: float
    mean_motif_plddt: float
    ca_rmsd: Optional[float] = None
    motif_ca_rmsd: Optional[float] = None
    motif_all_atom_rmsd: Optional[float] = None
    contig_str: Optional[str] = None
    motif_residues: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)


class ColabFoldRunner:
    """Run ColabFold for structure prediction and validation

    Supports:
    - Structure prediction from sequences
    - pLDDT extraction
    - RMSD calculation
    - Metrics aggregation
    """

    def __init__(self, colabfold_command: str = "colabfold_batch"):
        """Initialize ColabFold runner

        Args:
            colabfold_command: Command to run ColabFold (default: "colabfold_batch")
        """
        self.colabfold_command = colabfold_command

        # Validate ColabFold is available
        if not check_executable_exists(colabfold_command):
            logger.warning(f"ColabFold command '{colabfold_command}' not found in PATH")
            logger.warning("Structure prediction will fail unless ColabFold is installed")

        logger.info(f"Initialized ColabFoldRunner with command: {colabfold_command}")

    def run_folding(
        self,
        fasta_path: Path,
        output_dir: Path,
        config: FoldingConfig,
        name: str
    ) -> List[FoldingResult]:
        """Run ColabFold structure prediction

        Adapted from ProtDesign2's fold.py.

        Args:
            fasta_path: Path to FASTA file with sequences
            output_dir: Directory for output files
            config: Folding configuration
            name: Experiment name

        Returns:
            List of FoldingResult objects

        Example:
            >>> runner = ColabFoldRunner()
            >>> config = FoldingConfig(
            ...     msa_mode="single_sequence",
            ...     num_models=2,
            ...     num_recycles=3
            ... )
            >>> results = runner.run_folding(
            ...     fasta_path=Path("sequences.fa"),
            ...     output_dir=Path("folding_output"),
            ...     config=config,
            ...     name="pfas_design"
            ... )
        """
        logger.info(f"Running ColabFold: {name}")
        logger.info(f"Input FASTA: {fasta_path}")
        logger.info(f"Output directory: {output_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command options
        opts = [
            f"--msa-mode {config.msa_mode}",
            f"--num-models {config.num_models}",
            f"--num-recycle {config.num_recycles}"
        ]

        if config.max_msa:
            opts.append(f"--max-msa {config.max_msa}")

        opts.append(f"{fasta_path} {output_dir}")
        opts_str = " ".join(opts)

        cmd = f"{self.colabfold_command} {opts_str}"

        logger.info(f"Executing: {cmd}")
        returncode, stdout, stderr = run_command(cmd, capture_output=False)

        if returncode != 0:
            logger.error(f"ColabFold failed with return code {returncode}")
            raise RuntimeError("ColabFold execution failed")

        logger.info("ColabFold completed successfully")

        # Parse sequences from FASTA
        seq_dict = self._read_fasta_sequences(fasta_path)

        # Postprocess results
        self._postprocess_results(output_dir)

        # Parse results
        results = self._parse_results(output_dir, seq_dict)

        logger.info(f"Generated {len(results)} folding results")
        return results

    def calculate_metrics(
        self,
        folded_pdb: Path,
        reference_pdb: Path,
        diffusion_pdb: Path,
        design_motif: List[str],
        ref_motif: List[str],
        sequence: str,
        contig_str: str,
        design_id: str,
        output_dir: Path
    ) -> FoldingResult:
        """Calculate comprehensive metrics for a folded structure

        Adapted from ProtDesign2's get_scores() function.

        Args:
            folded_pdb: Path to folded structure (rank_001)
            reference_pdb: Reference structure with motif
            diffusion_pdb: RFdiffusion output backbone
            design_motif: Motif residues in design
            ref_motif: Motif residues in reference
            sequence: Protein sequence
            contig_str: Contig string
            design_id: Design identifier
            output_dir: Directory containing ColabFold outputs

        Returns:
            FoldingResult with comprehensive metrics
        """
        logger.debug(f"Calculating metrics for {design_id}")

        # Get pLDDT scores
        design_motif_indices = [int(resi[1:]) for resi in design_motif]
        mean_plddt, mean_motif_plddt = self._get_plddt_scores(output_dir, design_motif_indices)

        # Calculate RMSD metrics
        ca_rmsd = get_ca_rmsd(design_path=folded_pdb, ref_path=diffusion_pdb)
        motif_ca_rmsd = get_motif_ca_rmsd(
            design_path=folded_pdb,
            ref_path=reference_pdb,
            design_motif=design_motif,
            ref_motif=ref_motif
        )
        motif_all_atom_rmsd = get_motif_all_atom_rmsd(
            design_path=folded_pdb,
            ref_path=reference_pdb,
            design_motif=design_motif,
            ref_motif=ref_motif
        )

        logger.debug(f"  pLDDT: {mean_plddt:.2f}, Motif pLDDT: {mean_motif_plddt:.2f}")
        logger.debug(f"  CA-RMSD: {ca_rmsd:.2f} Å, Motif CA-RMSD: {motif_ca_rmsd:.2f} Å")

        result = FoldingResult(
            design_id=design_id,
            sequence=sequence,
            pdb_path=folded_pdb,
            mean_plddt=mean_plddt,
            mean_motif_plddt=mean_motif_plddt,
            ca_rmsd=ca_rmsd,
            motif_ca_rmsd=motif_ca_rmsd,
            motif_all_atom_rmsd=motif_all_atom_rmsd,
            contig_str=contig_str,
            motif_residues=design_motif,
            metadata={
                "reference_pdb": str(reference_pdb),
                "diffusion_pdb": str(diffusion_pdb)
            }
        )

        return result

    def create_scores_file(
        self,
        output_dir: Path,
        reference_pdb: Path,
        contig_str: str,
        seq_dict: Dict[str, str],
        diffusion_dir: Optional[Path] = None
    ) -> Path:
        """Create comprehensive scores JSON file

        Adapted from ProtDesign2's create_score_file() function.

        Args:
            output_dir: ColabFold output directory
            reference_pdb: Reference PDB with motif
            contig_str: Contig string
            seq_dict: Dictionary of {design_id: sequence}
            diffusion_dir: Directory with RFdiffusion outputs (optional)

        Returns:
            Path to scores.json file
        """
        logger.info("Creating scores file...")

        design_motif, ref_motif, _ = get_motifs(contig_str)

        # Find all result subdirectories
        subfolders = [Path(f) for f in globlib(str(output_dir / "*")) if Path(f).is_dir()]
        logger.info(f"Found {len(subfolders)} result directories")

        combined_scores = {}

        for folder in subfolders:
            design_id = folder.name

            # Find corresponding diffusion PDB
            if diffusion_dir:
                # Extract base diffusion ID (e.g., "pfas_0_n1_c0" -> "pfas_0.pdb")
                diff_id = f"{design_id.split('_')[0]}_{design_id.split('_')[1]}.pdb"
                diff_path = diffusion_dir / diff_id
            else:
                # Assume diffusion output is in sibling directory
                diff_path = output_dir.parent / "Diffusion" / f"{design_id.split('_')[0]}_{design_id.split('_')[1]}.pdb"

            if not diff_path.exists():
                logger.warning(f"Diffusion PDB not found: {diff_path}, using reference for CA-RMSD")
                diff_path = reference_pdb

            # Get sequence
            sequence = seq_dict.get(design_id, "")

            # Find rank_001 PDB
            rank_pdbs = list(folder.glob("*rank_001*.pdb"))
            if not rank_pdbs:
                logger.warning(f"No rank_001 PDB found for {design_id}")
                continue

            folded_pdb = rank_pdbs[0]

            # Calculate metrics
            result = self.calculate_metrics(
                folded_pdb=folded_pdb,
                reference_pdb=reference_pdb,
                diffusion_pdb=diff_path,
                design_motif=design_motif,
                ref_motif=ref_motif,
                sequence=sequence,
                contig_str=contig_str,
                design_id=design_id,
                output_dir=folder
            )

            # Convert to dictionary
            score_dict = {
                "id": result.design_id,
                "seq": result.sequence,
                "contig_str": result.contig_str,
                "motif": result.motif_residues,
                "mean-plddt": result.mean_plddt,
                "mean-motif-plddt": result.mean_motif_plddt,
                "ca-rmsd": result.ca_rmsd,
                "motif-ca-rmsd": result.motif_ca_rmsd,
                "motif-all-atom-rmsd": result.motif_all_atom_rmsd
            }

            combined_scores[design_id] = score_dict

        # Write scores file
        scores_path = output_dir / "scores.json"
        with open(scores_path, 'w') as f:
            json.dump(combined_scores, f, indent=4)

        logger.info(f"Created scores file: {scores_path}")
        logger.info(f"Scored {len(combined_scores)} designs")

        return scores_path

    def _get_plddt_scores(
        self,
        output_dir: Path,
        motif_indices: List[int]
    ) -> tuple[float, float]:
        """Extract pLDDT scores from ColabFold JSON output

        Adapted from ProtDesign2's get_mean_plddt() function.

        Args:
            output_dir: Directory containing rank_001 JSON
            motif_indices: Residue indices for motif (0-indexed)

        Returns:
            Tuple of (mean_plddt, mean_motif_plddt)
        """
        # Find rank_001 JSON file
        json_files = list(output_dir.glob("*rank_001*.json"))

        if not json_files:
            logger.warning(f"No rank_001 JSON found in {output_dir}")
            return 0.0, 0.0

        json_file = json_files[0]

        with open(json_file, 'r') as f:
            data = json.load(f)

        plddt_scores = data.get("plddt", [])

        if not plddt_scores:
            logger.warning(f"No pLDDT scores in {json_file}")
            return 0.0, 0.0

        mean_plddt = round(mean(plddt_scores), 2)

        # Extract motif pLDDT scores
        motif_plddt = [plddt_scores[i] for i in motif_indices if i < len(plddt_scores)]

        if motif_plddt:
            mean_motif_plddt = round(mean(motif_plddt), 2)
        else:
            mean_motif_plddt = 0.0

        return mean_plddt, mean_motif_plddt

    def _read_fasta_sequences(self, fasta_path: Path) -> Dict[str, str]:
        """Read sequences from FASTA file

        Args:
            fasta_path: Path to FASTA file

        Returns:
            Dictionary of {sequence_id: sequence}
        """
        seq_dict = {}

        with open(fasta_path, 'r') as f:
            lines = f.readlines()

        current_id = None
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                # Header line
                current_id = line[1:].split()[0]  # Remove '>' and take first field
            elif current_id:
                # Sequence line
                seq_dict[current_id] = line

        logger.debug(f"Read {len(seq_dict)} sequences from {fasta_path.name}")
        return seq_dict

    def _postprocess_results(self, output_dir: Path) -> None:
        """Organize ColabFold output files into subdirectories

        Adapted from ProtDesign2's postprocessing() function.

        Args:
            output_dir: ColabFold output directory
        """
        logger.info("Organizing ColabFold output files...")

        # Remove temporary files
        for suffix in [".done.txt", ".a3m"]:
            for file in output_dir.glob(f"*{suffix}"):
                file.unlink()
                logger.debug(f"Removed {file.name}")

        # Keep config and log files in main directory
        keep_suffixes = ["bibtex", "config.json", "log.txt", "scores.json"]

        # Move result files to subdirectories
        for file in output_dir.iterdir():
            if file.is_dir():
                continue

            if any(file.name.endswith(suffix) for suffix in keep_suffixes):
                continue

            # Extract design ID from filename
            # Typically: designname_0_n1_c0_unrelaxed_rank_001...
            parts = file.name.split("_")
            if len(parts) >= 4:
                design_id = "_".join(parts[:4])  # e.g., "pfas_0_n1_c0"

                # Create subdirectory
                subdir = output_dir / design_id
                subdir.mkdir(exist_ok=True)

                # Move file
                shutil.move(str(file), str(subdir / file.name))
                logger.debug(f"Moved {file.name} to {design_id}/")

        logger.info("File organization complete")

    def _parse_results(
        self,
        output_dir: Path,
        seq_dict: Dict[str, str]
    ) -> List[FoldingResult]:
        """Parse ColabFold results from output directory

        Args:
            output_dir: ColabFold output directory
            seq_dict: Dictionary of sequences

        Returns:
            List of FoldingResult objects
        """
        logger.info("Parsing ColabFold results...")

        results = []
        subfolders = [Path(f) for f in globlib(str(output_dir / "*")) if Path(f).is_dir()]

        for folder in subfolders:
            design_id = folder.name

            # Find rank_001 PDB and JSON
            rank_pdbs = list(folder.glob("*rank_001*.pdb"))
            if not rank_pdbs:
                logger.warning(f"No rank_001 PDB for {design_id}")
                continue

            pdb_path = rank_pdbs[0]

            # Get pLDDT scores (no motif info at this stage)
            mean_plddt, _ = self._get_plddt_scores(folder, [])

            # Get sequence
            sequence = seq_dict.get(design_id, "")

            result = FoldingResult(
                design_id=design_id,
                sequence=sequence,
                pdb_path=pdb_path,
                mean_plddt=mean_plddt,
                mean_motif_plddt=0.0,  # Will be calculated later if needed
                metadata={"output_dir": str(folder)}
            )

            results.append(result)

        logger.info(f"Parsed {len(results)} results")
        return results

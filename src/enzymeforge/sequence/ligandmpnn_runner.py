"""
LigandMPNN runner for EnzymeForge

Handles LigandMPNN/ProteinMPNN sequence design with fixed motif residues.

Adapted from ProtDesign2 with improvements:
- Type hints and dataclasses
- pathlib for path handling
- Better error handling
- Proper logging
- Filtering integration with StructureValidator
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from glob import glob

from enzymeforge.utils.process import run_command, check_path_exists
from enzymeforge.utils.pdb_utils import get_motifs
from enzymeforge.validation.structure_validator import StructureValidator

logger = logging.getLogger(__name__)


@dataclass
class SequenceDesignConfig:
    """LigandMPNN sequence design parameters

    Attributes:
        model_type: Model to use ("ligand_mpnn" or "protein_mpnn")
        num_seqs: Number of sequences to generate per design
        temperature: Sampling temperature (default: 0.1)
        seed: Random seed (optional)
        pack_side_chains: Pack side chains (default: True)
        pack_with_ligand_context: Consider ligand when packing (default: True)
        number_of_packs_per_design: Packing attempts per design (default: 1)
        repack_everything: Repack all residues vs. only designed (default: False)
        zero_indexed: Use zero-indexed residue numbering (default: True)
    """
    model_type: str = "ligand_mpnn"
    num_seqs: int = 4
    temperature: float = 0.1
    seed: Optional[int] = None
    pack_side_chains: bool = True
    pack_with_ligand_context: bool = True
    number_of_packs_per_design: int = 1
    repack_everything: bool = False
    zero_indexed: bool = True


@dataclass
class SequenceDesignResult:
    """Output from sequence design"""
    design_id: str
    pdb_path: Path
    sequences: List[str]
    fasta_path: Optional[Path] = None
    metadata: Dict = field(default_factory=dict)


class LigandMPNNRunner:
    """Run LigandMPNN for sequence design with fixed motif residues

    Supports:
    - Fixed motif residues (catalytic residues)
    - Ligand-aware packing
    - Pre-filtering by RMSD and clashes
    - Batch processing
    """

    def __init__(self, ligandmpnn_path: Path):
        """Initialize LigandMPNN runner

        Args:
            ligandmpnn_path: Path to LigandMPNN installation
        """
        self.ligandmpnn_path = Path(ligandmpnn_path)

        # Validate path
        if not check_path_exists(self.ligandmpnn_path, "directory"):
            raise ValueError(f"LigandMPNN path not found: {self.ligandmpnn_path}")

        logger.info(f"Initialized LigandMPNNRunner with path: {self.ligandmpnn_path}")

    def run_design(
        self,
        design_pdbs: List[Path],
        config: SequenceDesignConfig,
        contig_str: str,
        reference_pdb: Path,
        ligand_name: str,
        output_dir: Path,
        name: str,
        rmsd_cutoff: float = 2.0,
        clash_cutoff: float = 2.0,
        filter_first: bool = True
    ) -> List[SequenceDesignResult]:
        """Run sequence design on protein backbones

        Adapted from ProtDesign2's design() function.

        Args:
            design_pdbs: List of design PDB files from RFdiffusion
            config: Sequence design configuration
            contig_str: Contig string (for motif extraction)
            reference_pdb: Reference PDB with motif
            ligand_name: Substrate/ligand name
            output_dir: Output directory
            name: Experiment name
            rmsd_cutoff: RMSD threshold for filtering (Angstroms)
            clash_cutoff: Clash distance threshold (Angstroms)
            filter_first: If True, filter designs before sequence design

        Returns:
            List of SequenceDesignResult objects

        Example:
            >>> runner = LigandMPNNRunner(Path("/path/to/LigandMPNN"))
            >>> config = SequenceDesignConfig(num_seqs=8, temperature=0.1)
            >>> results = runner.run_design(
            ...     design_pdbs=[Path("design_0.pdb"), Path("design_1.pdb")],
            ...     config=config,
            ...     contig_str="50-50/A45-47/50-50",
            ...     reference_pdb=Path("input.pdb"),
            ...     ligand_name="FOA",
            ...     output_dir=Path("output"),
            ...     name="pfas_design"
            ... )
        """
        logger.info(f"Running sequence design: {name}")
        logger.info(f"Input designs: {len(design_pdbs)}")

        # Create output directories
        input_dir = output_dir / "inputs"
        output_subdir = output_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Extract motifs from contig string
        design_motif, ref_motif, redesigned_residues = get_motifs(contig_str)
        logger.info(f"Fixed motif residues: {len(design_motif)}")
        logger.info(f"Redesigned residues: {len(redesigned_residues)}")

        # Filter designs by RMSD and clashes
        if filter_first:
            design_pdbs = self._filter_designs(
                design_pdbs=design_pdbs,
                reference_pdb=reference_pdb,
                design_motif=design_motif,
                ref_motif=ref_motif,
                ligand_name=ligand_name,
                rmsd_cutoff=rmsd_cutoff,
                clash_cutoff=clash_cutoff
            )

            if len(design_pdbs) == 0:
                logger.warning("No designs passed filtering!")
                return []

            logger.info(f"Designs after filtering: {len(design_pdbs)}")

        # Prepare JSON input files
        self._prepare_input_files(
            design_pdbs=design_pdbs,
            design_motif=design_motif,
            redesigned_residues=redesigned_residues,
            output_dir=input_dir
        )

        # Run LigandMPNN
        self._run_ligandmpnn(
            config=config,
            input_dir=input_dir,
            output_dir=output_subdir
        )

        # Parse results
        results = self._parse_fasta_results(
            output_dir=output_subdir,
            name=name,
            design_pdbs=design_pdbs
        )

        logger.info(f"Generated {len(results)} sequence design results")
        return results

    def _filter_designs(
        self,
        design_pdbs: List[Path],
        reference_pdb: Path,
        design_motif: List[str],
        ref_motif: List[str],
        ligand_name: str,
        rmsd_cutoff: float,
        clash_cutoff: float
    ) -> List[Path]:
        """Filter designs by RMSD and clashes

        Adapted from ProtDesign2's preprocessing() function.

        Args:
            design_pdbs: List of design PDB files
            reference_pdb: Reference PDB
            design_motif: Motif residues in designs
            ref_motif: Motif residues in reference
            ligand_name: Ligand name
            rmsd_cutoff: RMSD threshold
            clash_cutoff: Clash threshold

        Returns:
            List of passing design PDB paths
        """
        logger.info(f"Filtering {len(design_pdbs)} designs...")
        logger.info(f"RMSD cutoff: {rmsd_cutoff} Å, Clash cutoff: {clash_cutoff} Å")

        validator = StructureValidator(
            rmsd_cutoff=rmsd_cutoff,
            clash_cutoff=clash_cutoff
        )

        passing_designs, metrics = validator.filter_designs(
            design_pdbs=design_pdbs,
            reference_pdb=reference_pdb,
            design_motif=design_motif,
            ref_motif=ref_motif,
            ligand_name=ligand_name
        )

        logger.info(f"Filtering complete: {len(passing_designs)}/{len(design_pdbs)} passed")
        return passing_designs

    def _prepare_input_files(
        self,
        design_pdbs: List[Path],
        design_motif: List[str],
        redesigned_residues: List[str],
        output_dir: Path
    ) -> None:
        """Prepare JSON input files for LigandMPNN

        Adapted from ProtDesign2's preprocessing() function.

        Creates three JSON files:
        - pdb_ids.json: {pdb_path: ""}
        - fix_residues_multi.json: {pdb_path: [fixed_residues]}
        - redesigned_residues_multi.json: {pdb_path: [redesigned_residues]}

        Args:
            design_pdbs: List of design PDB files
            design_motif: Fixed motif residues
            redesigned_residues: Residues to redesign
            output_dir: Directory for JSON files
        """
        logger.info("Preparing LigandMPNN input files...")

        # Create dictionaries
        pdb_dict = {str(path): "" for path in design_pdbs}
        fixed_resi_dict = {str(path): design_motif for path in design_pdbs}
        redesigned_resi_dict = {str(path): redesigned_residues for path in design_pdbs}

        # Write JSON files
        self._write_json(output_dir / "pdb_ids.json", pdb_dict)
        self._write_json(output_dir / "fix_residues_multi.json", fixed_resi_dict)
        self._write_json(output_dir / "redesigned_residues_multi.json", redesigned_resi_dict)

        logger.info(f"Created input files for {len(design_pdbs)} designs")

    def _write_json(self, path: Path, data: Dict) -> None:
        """Write dictionary to JSON file

        Args:
            path: Output file path
            data: Dictionary to write
        """
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Wrote {path.name}")

    def _run_ligandmpnn(
        self,
        config: SequenceDesignConfig,
        input_dir: Path,
        output_dir: Path
    ) -> None:
        """Run LigandMPNN sequence design

        Adapted from ProtDesign2's design() function.

        Args:
            config: Sequence design configuration
            input_dir: Directory with JSON input files
            output_dir: Directory for output files
        """
        logger.info("Running LigandMPNN...")

        # Build command options
        opts = [
            f"--model_type {config.model_type}",
            f"--out_folder {output_dir}",
            f"--number_of_batches {config.num_seqs}",
            f"--pdb_path_multi {input_dir / 'pdb_ids.json'}",
            f"--fixed_residues_multi {input_dir / 'fix_residues_multi.json'}",
            f"--redesigned_residues_multi {input_dir / 'redesigned_residues_multi.json'}",
            f"--zero_indexed {1 if config.zero_indexed else 0}",
            f"--pack_side_chains {1 if config.pack_side_chains else 0}",
            f"--pack_with_ligand_context {1 if config.pack_with_ligand_context else 0}",
            f"--number_of_packs_per_design {config.number_of_packs_per_design}",
            f"--repack_everything {1 if config.repack_everything else 0}"
        ]

        # Add optional parameters
        if config.seed is not None:
            opts.append(f"--seed {config.seed}")
        if config.temperature is not None:
            opts.append(f"--temperature {config.temperature}")

        opts_str = " ".join(opts)
        cmd = f"cd {self.ligandmpnn_path} && python3.9 run.py {opts_str}"

        logger.info(f"Executing: {cmd}")
        returncode, stdout, stderr = run_command(cmd, capture_output=False)

        if returncode != 0:
            logger.error(f"LigandMPNN failed with return code {returncode}")
            raise RuntimeError("LigandMPNN execution failed")

        logger.info("LigandMPNN completed successfully")

    def _parse_fasta_results(
        self,
        output_dir: Path,
        name: str,
        design_pdbs: List[Path]
    ) -> List[SequenceDesignResult]:
        """Parse FASTA output files from LigandMPNN

        Args:
            output_dir: LigandMPNN output directory
            name: Experiment name
            design_pdbs: List of input design PDB files

        Returns:
            List of SequenceDesignResult objects
        """
        logger.info("Parsing FASTA results...")

        results = []
        seqs_dir = output_dir / "seqs"

        if not seqs_dir.exists():
            logger.warning(f"Sequences directory not found: {seqs_dir}")
            return results

        # Find all FASTA files
        fasta_files = list(seqs_dir.glob("*.fa"))
        logger.info(f"Found {len(fasta_files)} FASTA files")

        # Parse each FASTA file
        for fasta_file in fasta_files:
            sequences = self._read_fasta(fasta_file)

            # Extract design ID from filename
            # Typically: design_name_0_pdb.fa or similar
            design_id = fasta_file.stem

            # Try to match with input PDB
            matching_pdb = None
            for pdb in design_pdbs:
                if pdb.stem in design_id:
                    matching_pdb = pdb
                    break

            result = SequenceDesignResult(
                design_id=design_id,
                pdb_path=matching_pdb if matching_pdb else Path("unknown"),
                sequences=sequences,
                fasta_path=fasta_file,
                metadata={
                    "num_sequences": len(sequences),
                    "output_dir": str(output_dir)
                }
            )
            results.append(result)

        logger.info(f"Parsed {len(results)} sequence design results")
        return results

    def _read_fasta(self, fasta_path: Path) -> List[str]:
        """Read sequences from FASTA file

        Args:
            fasta_path: Path to FASTA file

        Returns:
            List of sequences (excluding headers)
        """
        sequences = []

        with open(fasta_path, 'r') as f:
            lines = f.readlines()

        # Skip first 2 lines (typically headers/metadata)
        # Then read sequence lines (every other line after headers)
        for i in range(2, len(lines), 2):
            if i + 1 < len(lines):
                sequence = lines[i + 1].strip()
                sequences.append(sequence)

        return sequences

    def merge_fasta_files(
        self,
        output_dir: Path,
        name: str,
        relax_round: int = 0
    ) -> Path:
        """Merge all FASTA files into a single output file

        Adapted from ProtDesign2's postprocessing() function.

        Args:
            output_dir: Directory containing FASTA files
            name: Experiment name
            relax_round: Relax cycle number (for naming)

        Returns:
            Path to merged FASTA file
        """
        logger.info("Merging FASTA files...")

        seqs_dir = output_dir / "seqs"
        fasta_files = list(seqs_dir.glob("*.fa"))

        all_lines = []
        for fasta_file in fasta_files:
            with open(fasta_file, 'r') as f:
                lines = f.readlines()

            # Skip first 2 lines (metadata)
            save_lines = lines[2:]

            # Process headers
            for i in range(0, len(save_lines) - 1, 2):
                header = save_lines[i].split(",")[0]
                if relax_round > 0:
                    save_lines[i] = header + "\n"
                else:
                    # Add naming convention: _nX_c0
                    batch_num = save_lines[i].split(",")[1][4:] if "," in save_lines[i] else "0"
                    save_lines[i] = f"{header}_n{batch_num}_c{relax_round}\n"

            # Ensure last line has newline
            if save_lines:
                save_lines[-1] = save_lines[-1].rstrip() + "\n"

            all_lines.extend(save_lines)

        # Write merged file
        output_path = output_dir / f"{name}_c{relax_round}.fa"
        with open(output_path, 'w') as f:
            f.writelines(all_lines)

        logger.info(f"Merged {len(fasta_files)} files into {output_path.name}")
        return output_path

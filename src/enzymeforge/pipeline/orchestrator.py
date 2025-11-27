"""
Pipeline orchestrator for EnzymeForge

Main entry point for running complete enzyme design pipeline.
Coordinates all components from substrate analysis to final validation.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from enzymeforge.pipeline.config import PipelineConfig
from enzymeforge.substrate.analyzer import SubstrateAnalyzer, Substrate, CatalyticSite
from enzymeforge.substrate.constraint_generator import ConstraintGenerator
from enzymeforge.diffusion.rfdiffusion_runner import RFdiffusionRunner, DiffusionResult
from enzymeforge.validation.structure_validator import StructureValidator
from enzymeforge.sequence.ligandmpnn_runner import LigandMPNNRunner, SequenceDesignResult
from enzymeforge.validation.colabfold_runner import ColabFoldRunner, FoldingResult
from enzymeforge.utils.pdb_utils import get_motifs

logger = logging.getLogger(__name__)


class EnzymePipeline:
    """Complete enzyme design pipeline orchestrator

    Coordinates the full pipeline:
    1. Substrate analysis
    2. Constraint generation
    3. RFdiffusion backbone generation
    4. Quality filtering
    5. LigandMPNN sequence design
    6. ColabFold structure prediction and validation

    Example:
        >>> config = PipelineConfig.from_yaml("config.yaml")
        >>> pipeline = EnzymePipeline(config)
        >>> results = pipeline.run_full_pipeline()
        >>> print(f"Generated {len(results)} validated designs")
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline

        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize component runners
        self.substrate_analyzer = SubstrateAnalyzer()
        self.rfdiffusion_runner = RFdiffusionRunner(
            rfdiffusion_path=config.compute.rfdiffusion_path,
            rfdiffusion_aa_path=config.compute.rfdiffusion_aa_path
        )
        self.structure_validator = StructureValidator()
        self.ligandmpnn_runner = LigandMPNNRunner(
            ligandmpnn_path=config.compute.ligandmpnn_path
        )
        self.colabfold_runner = ColabFoldRunner(
            colabfold_command=config.compute.colabfold_command
        )

        # Pipeline state
        self.substrate: Optional[Substrate] = None
        self.diffusion_results: List[DiffusionResult] = []
        self.sequence_results: List[SequenceDesignResult] = []
        self.folding_results: List[FoldingResult] = []

        # Statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "num_diffusion_designs": 0,
            "num_filtered_after_diffusion": 0,
            "num_sequence_designs": 0,
            "num_filtered_after_sequence": 0,
            "num_final_designs": 0,
            "num_high_quality": 0  # pLDDT > cutoff
        }

        logger.info(f"Initialized EnzymePipeline: {config.name}")
        logger.info(f"Output directory: {self.output_dir}")

    def run_full_pipeline(self) -> List[FoldingResult]:
        """Run complete enzyme design pipeline

        Returns:
            List of FoldingResult objects for final validated designs

        Pipeline stages:
        1. Substrate analysis and constraint generation
        2. RFdiffusion backbone generation
        3. Quality filtering (RMSD, clashes)
        4. LigandMPNN sequence design
        5. Optional second filtering
        6. ColabFold structure prediction
        7. Final metrics and scoring

        Example:
            >>> pipeline = EnzymePipeline(config)
            >>> results = pipeline.run_full_pipeline()
            >>> for result in results:
            ...     print(f"{result.design_id}: pLDDT={result.mean_plddt:.1f}")
        """
        self.stats["start_time"] = datetime.now()
        logger.info("=" * 80)
        logger.info(f"Starting EnzymeForge pipeline: {self.config.name}")
        logger.info("=" * 80)

        try:
            # Stage 1: Substrate analysis
            logger.info("\n[Stage 1/6] Substrate Analysis")
            self._stage1_substrate_analysis()

            # Stage 2: RFdiffusion
            logger.info("\n[Stage 2/6] RFdiffusion Backbone Generation")
            self._stage2_rfdiffusion()

            # Stage 3: Filter diffusion results
            if self.config.filtering.filter_after_diffusion:
                logger.info("\n[Stage 3/6] Quality Filtering (Post-Diffusion)")
                self._stage3_filter_diffusion()
            else:
                logger.info("\n[Stage 3/6] Quality Filtering - SKIPPED")

            # Stage 4: Sequence design
            logger.info("\n[Stage 4/6] LigandMPNN Sequence Design")
            self._stage4_sequence_design()

            # Stage 5: Structure prediction
            logger.info("\n[Stage 5/6] ColabFold Structure Prediction")
            self._stage5_structure_prediction()

            # Stage 6: Final scoring and analysis
            logger.info("\n[Stage 6/6] Final Scoring and Analysis")
            self._stage6_final_scoring()

            # Save final results
            self._save_results()

            self.stats["end_time"] = datetime.now()
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
            logger.info(f"Final designs: {self.stats['num_final_designs']}")
            logger.info(f"High quality (pLDDT >{self.config.filtering.plddt_cutoff}): {self.stats['num_high_quality']}")
            logger.info("=" * 80)

            return self.folding_results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _stage1_substrate_analysis(self) -> None:
        """Stage 1: Analyze substrate and generate constraints"""
        logger.info(f"Loading substrate: {self.config.substrate.name}")

        # Load substrate
        self.substrate = self.substrate_analyzer.load_substrate(
            structure=self.config.substrate.structure,
            format=self.config.substrate.format,
            name=self.config.substrate.name
        )

        # Identify functional groups if SMILES
        if self.config.substrate.format == "smiles":
            groups = self.substrate_analyzer.identify_functional_groups()
            logger.info(f"Identified {len(groups)} functional groups:")
            for group in groups:
                logger.info(f"  - {group['type']}: {group['count']} (mechanism: {group['suggested_mechanism']})")

        # Generate constraints
        catalytic_site = CatalyticSite(
            mechanism=self.config.active_site.mechanism,
            catalytic_residues=self.config.active_site.catalytic_residues
        )

        constraint_gen = ConstraintGenerator(self.substrate, catalytic_site)

        # Save contig string
        contig_file = self.output_dir / f"{self.config.name}_contig.txt"
        with open(contig_file, 'w') as f:
            f.write(self.config.diffusion.contigs)
        logger.info(f"Contig string: {self.config.diffusion.contigs}")

        # Generate constraint file if reference PDB provided
        if self.config.active_site.reference_pdb:
            cst_file = self.output_dir / f"{self.config.name}_constraints.cst"
            constraint_gen.generate_cst_file(str(cst_file))
            logger.info(f"Generated constraint file: {cst_file}")

    def _stage2_rfdiffusion(self) -> None:
        """Stage 2: Generate protein backbones with RFdiffusion"""
        logger.info(f"Generating {self.config.scaffold.num_designs} backbone designs")

        # Choose RFdiffusion variant
        if self.config.compute.use_rfdiffusion_aa and self.config.compute.rfdiffusion_aa_path:
            logger.info("Using RFdiffusion all-atom")
            self.diffusion_results = self.rfdiffusion_runner.run_diffusion_allatom(
                config=self.config.diffusion,
                name=self.config.name,
                output_dir=self.output_dir
            )
        else:
            logger.info("Using standard RFdiffusion")
            # Check if enzyme design mode
            enzyme_design = bool(self.config.diffusion.guide_potentials)
            self.diffusion_results = self.rfdiffusion_runner.run_diffusion(
                config=self.config.diffusion,
                name=self.config.name,
                output_dir=self.output_dir,
                enzyme_design=enzyme_design
            )

        self.stats["num_diffusion_designs"] = len(self.diffusion_results)
        logger.info(f"Generated {len(self.diffusion_results)} backbone designs")

    def _stage3_filter_diffusion(self) -> None:
        """Stage 3: Filter RFdiffusion results by quality"""
        if not self.config.active_site.reference_pdb:
            logger.info("No reference PDB provided, skipping filtering")
            return

        logger.info(f"Filtering {len(self.diffusion_results)} designs")

        # Extract design PDbs
        design_pdbs = [result.pdb_path for result in self.diffusion_results]

        # Get motif information
        design_motif, ref_motif, _ = get_motifs(self.config.diffusion.contigs)

        # Filter designs
        passing_pdbs, metrics_list = self.structure_validator.filter_designs(
            design_pdbs=design_pdbs,
            reference_pdb=self.config.active_site.reference_pdb,
            design_motif=design_motif,
            ref_motif=ref_motif,
            ligand_name=self.config.substrate.ligand_name or "",
            rmsd_cutoff=self.config.filtering.rmsd_cutoff,
            clash_cutoff=self.config.filtering.clash_cutoff
        )

        # Update diffusion results to keep only passing designs
        passing_set = set(passing_pdbs)
        self.diffusion_results = [
            r for r in self.diffusion_results if r.pdb_path in passing_set
        ]

        self.stats["num_filtered_after_diffusion"] = len(self.diffusion_results)
        logger.info(f"Retained {len(self.diffusion_results)} designs after filtering")
        logger.info(f"Filtered out {self.stats['num_diffusion_designs'] - len(self.diffusion_results)} designs")

        if not self.diffusion_results:
            raise RuntimeError("No designs passed quality filtering!")

    def _stage4_sequence_design(self) -> None:
        """Stage 4: Design sequences with LigandMPNN"""
        logger.info(f"Designing sequences for {len(self.diffusion_results)} backbones")

        # Extract design PDbs
        design_pdbs = [result.pdb_path for result in self.diffusion_results]

        # Run sequence design
        self.sequence_results = self.ligandmpnn_runner.run_design(
            design_pdbs=design_pdbs,
            config=self.config.sequence,
            contig_str=self.config.diffusion.contigs,
            reference_pdb=self.config.active_site.reference_pdb or design_pdbs[0],
            ligand_name=self.config.substrate.ligand_name or "",
            output_dir=self.output_dir / self.config.name / "SeqDesign",
            name=self.config.name,
            rmsd_cutoff=self.config.filtering.rmsd_cutoff,
            clash_cutoff=self.config.filtering.clash_cutoff,
            filter_first=self.config.filtering.filter_after_sequence
        )

        self.stats["num_sequence_designs"] = len(self.sequence_results)
        logger.info(f"Generated {len(self.sequence_results)} sequence designs")

        # Count total sequences across all designs
        total_seqs = sum(len(r.sequences) for r in self.sequence_results)
        logger.info(f"Total sequences: {total_seqs}")

    def _stage5_structure_prediction(self) -> None:
        """Stage 5: Predict structures with ColabFold"""
        logger.info(f"Folding {len(self.sequence_results)} sequence designs")

        # Merge all FASTA files from sequence design
        fasta_path = self.ligandmpnn_runner.merge_fasta_files(
            output_dir=self.output_dir / self.config.name / "SeqDesign",
            name=self.config.name,
            relax_round=0  # TODO: support multiple relax rounds
        )

        logger.info(f"Merged FASTA file: {fasta_path}")

        # Run ColabFold
        folding_output_dir = self.output_dir / self.config.name / "Folding"
        self.folding_results = self.colabfold_runner.run_folding(
            fasta_path=fasta_path,
            output_dir=folding_output_dir,
            config=self.config.folding,
            name=self.config.name
        )

        logger.info(f"Predicted structures for {len(self.folding_results)} designs")

    def _stage6_final_scoring(self) -> None:
        """Stage 6: Calculate comprehensive metrics and create scores file"""
        logger.info("Calculating comprehensive metrics")

        # Check if we have reference PDB for detailed scoring
        if not self.config.active_site.reference_pdb:
            logger.warning("No reference PDB - skipping detailed RMSD calculations")
            self.stats["num_final_designs"] = len(self.folding_results)
            return

        # Read sequences from merged FASTA
        fasta_path = self.output_dir / self.config.name / "SeqDesign" / f"{self.config.name}_c0.fa"
        seq_dict = self.colabfold_runner._read_fasta_sequences(fasta_path)

        # Calculate comprehensive scores
        scores_path = self.colabfold_runner.create_scores_file(
            output_dir=self.output_dir / self.config.name / "Folding",
            reference_pdb=self.config.active_site.reference_pdb,
            contig_str=self.config.diffusion.contigs,
            seq_dict=seq_dict,
            diffusion_dir=self.output_dir / self.config.name / "Diffusion"
        )

        logger.info(f"Created scores file: {scores_path}")

        # Load scores and update statistics
        with open(scores_path, 'r') as f:
            scores = json.load(f)

        self.stats["num_final_designs"] = len(scores)

        # Count high-quality designs
        high_quality = sum(
            1 for s in scores.values()
            if s.get("mean-plddt", 0) >= self.config.filtering.plddt_cutoff
        )
        self.stats["num_high_quality"] = high_quality

        logger.info(f"Final designs: {self.stats['num_final_designs']}")
        logger.info(f"High quality (pLDDT >{self.config.filtering.plddt_cutoff}): {high_quality}")

    def _save_results(self) -> None:
        """Save pipeline results and statistics"""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Save statistics
        stats_file = results_dir / f"{self.config.name}_statistics.json"
        stats_dict = {
            **self.stats,
            "start_time": self.stats["start_time"].isoformat() if self.stats["start_time"] else None,
            "end_time": self.stats["end_time"].isoformat() if self.stats["end_time"] else None,
        }

        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=4)

        logger.info(f"Saved statistics: {stats_file}")

        # Save configuration
        config_file = results_dir / f"{self.config.name}_config.yaml"
        self.config.to_yaml(config_file)
        logger.info(f"Saved configuration: {config_file}")

        # Create summary
        summary_file = results_dir / f"{self.config.name}_summary.txt"
        self._write_summary(summary_file)
        logger.info(f"Saved summary: {summary_file}")

    def _write_summary(self, output_path: Path) -> None:
        """Write human-readable summary

        Args:
            output_path: Path to summary file
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"EnzymeForge Pipeline Summary: {self.config.name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Substrate: {self.config.substrate.name}\n")
            f.write(f"  Mechanism: {self.config.active_site.mechanism}\n")
            f.write(f"  Catalytic residues: {', '.join(self.config.active_site.catalytic_residues)}\n")
            f.write(f"  Scaffold size: {self.config.scaffold.min_size}-{self.config.scaffold.max_size} residues\n\n")

            f.write("Pipeline Statistics:\n")
            f.write(f"  RFdiffusion designs: {self.stats['num_diffusion_designs']}\n")
            if self.config.filtering.filter_after_diffusion:
                f.write(f"  After filtering: {self.stats['num_filtered_after_diffusion']}\n")
            f.write(f"  Sequence designs: {self.stats['num_sequence_designs']}\n")
            f.write(f"  Final designs: {self.stats['num_final_designs']}\n")
            f.write(f"  High quality: {self.stats['num_high_quality']}\n\n")

            if self.stats["start_time"] and self.stats["end_time"]:
                duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
                f.write(f"Duration: {duration:.1f}s ({duration/60:.1f} min)\n\n")

            f.write("Output Files:\n")
            f.write(f"  Backbones: {self.output_dir / self.config.name / 'Diffusion'}\n")
            f.write(f"  Sequences: {self.output_dir / self.config.name / 'SeqDesign'}\n")
            f.write(f"  Structures: {self.output_dir / self.config.name / 'Folding'}\n")
            f.write(f"  Scores: {self.output_dir / self.config.name / 'Folding' / 'scores.json'}\n")

    def get_best_designs(self, n: int = 10, metric: str = "mean-plddt") -> List[Dict]:
        """Get top N designs by specified metric

        Args:
            n: Number of designs to return
            metric: Metric to sort by ("mean-plddt", "motif-ca-rmsd", etc.)

        Returns:
            List of design dictionaries sorted by metric

        Example:
            >>> pipeline = EnzymePipeline(config)
            >>> results = pipeline.run_full_pipeline()
            >>> best = pipeline.get_best_designs(n=5, metric="mean-plddt")
            >>> for design in best:
            ...     print(f"{design['id']}: {design['mean-plddt']:.1f}")
        """
        scores_path = self.output_dir / self.config.name / "Folding" / "scores.json"

        if not scores_path.exists():
            logger.warning("No scores file found - run pipeline first")
            return []

        with open(scores_path, 'r') as f:
            scores = json.load(f)

        # Sort by metric (descending for pLDDT, ascending for RMSD)
        reverse = "plddt" in metric.lower()
        sorted_designs = sorted(
            scores.values(),
            key=lambda x: x.get(metric, 0 if reverse else float('inf')),
            reverse=reverse
        )

        return sorted_designs[:n]

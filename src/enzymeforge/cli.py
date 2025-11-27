"""
Command-line interface for EnzymeForge

Provides CLI for running enzyme design pipeline from configuration files.
"""

import argparse
import logging
import sys
from pathlib import Path

from enzymeforge.pipeline.config import PipelineConfig
from enzymeforge.pipeline.orchestrator import EnzymePipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI

    Args:
        verbose: Enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def run_pipeline(config_path: Path, verbose: bool = False) -> int:
    """Run enzyme design pipeline from configuration file

    Args:
        config_path: Path to YAML configuration file
        verbose: Enable verbose logging

    Returns:
        Exit code (0 = success, 1 = error)
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = PipelineConfig.from_yaml(config_path)

        # Create and run pipeline
        logger.info(f"Starting pipeline: {config.name}")
        pipeline = EnzymePipeline(config)
        results = pipeline.run_full_pipeline()

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Generated {len(results)} final designs")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("=" * 80)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=verbose)
        return 1


def analyze_results(output_dir: Path, metric: str = "mean-plddt", top_n: int = 10) -> int:
    """Analyze pipeline results and show top designs

    Args:
        output_dir: Pipeline output directory
        metric: Metric to sort by
        top_n: Number of top designs to show

    Returns:
        Exit code (0 = success, 1 = error)
    """
    import json

    setup_logging(verbose=False)
    logger = logging.getLogger(__name__)

    try:
        # Find scores file
        scores_files = list(output_dir.rglob("scores.json"))

        if not scores_files:
            logger.error(f"No scores.json found in {output_dir}")
            return 1

        scores_path = scores_files[0]
        logger.info(f"Reading scores from: {scores_path}")

        with open(scores_path, 'r') as f:
            scores = json.load(f)

        # Sort by metric
        reverse = "plddt" in metric.lower()
        sorted_designs = sorted(
            scores.values(),
            key=lambda x: x.get(metric, 0 if reverse else float('inf')),
            reverse=reverse
        )

        # Print top designs
        print("\n" + "=" * 80)
        print(f"Top {top_n} designs by {metric}:")
        print("=" * 80)
        print(f"{'Rank':<6} {'Design ID':<20} {metric:<20} {'pLDDT':<10} {'RMSD':<10}")
        print("-" * 80)

        for i, design in enumerate(sorted_designs[:top_n], 1):
            design_id = design.get('id', 'unknown')
            metric_val = design.get(metric, 0)
            plddt = design.get('mean-plddt', 0)
            rmsd = design.get('ca-rmsd', 0)

            print(f"{i:<6} {design_id:<20} {metric_val:<20.2f} {plddt:<10.1f} {rmsd:<10.2f}")

        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def create_example_config(output_path: Path) -> int:
    """Create example configuration file

    Args:
        output_path: Where to save example config

    Returns:
        Exit code (0 = success, 1 = error)
    """
    example_yaml = """# EnzymeForge Pipeline Configuration
# Example: PFAS degradation enzyme design

name: pfas_degrader_example
output_dir: output/pfas_exp1

# Substrate specification
substrate:
  name: PFOA
  structure: "C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(=O)O)F"
  format: smiles
  ligand_name: FOA

# Active site specification
active_site:
  mechanism: hydrolysis
  catalytic_residues:
    - SER
    - HIS
    - ASP
  reference_pdb: data/reference_structures/serine_protease.pdb
  motif_positions:
    - 45
    - 46
    - 47
  ref_chain: A

# Scaffold design parameters
scaffold:
  min_size: 100
  max_size: 150
  num_designs: 20

# RFdiffusion parameters
diffusion:
  contigs: ""  # Auto-generated from scaffold and motif
  iterations: 100
  num_designs: 20
  guide_potentials: null
  guide_scale: 1.0
  noise_scale: 1.0
  deterministic: false
  partial_diffusion: false

# LigandMPNN sequence design parameters
sequence:
  model_type: ligand_mpnn
  num_seqs: 4
  temperature: 0.1
  seed: null
  pack_side_chains: true
  pack_with_ligand_context: true
  number_of_packs_per_design: 1
  repack_everything: false
  zero_indexed: true

# ColabFold parameters
folding:
  msa_mode: single_sequence
  num_models: 2
  num_recycles: 3
  use_gpu: true
  max_msa: null

# Quality filtering parameters
filtering:
  rmsd_cutoff: 2.0
  clash_cutoff: 2.0
  plddt_cutoff: 70.0
  motif_plddt_cutoff: 80.0
  filter_after_diffusion: true
  filter_after_sequence: true

# Computational resources
compute:
  rfdiffusion_path: /path/to/RFdiffusion
  ligandmpnn_path: /path/to/LigandMPNN
  rfdiffusion_aa_path: null
  colabfold_command: colabfold_batch
  use_rfdiffusion_aa: false
  num_workers: 8

# Additional metadata
metadata:
  description: "PFAS degradation enzyme design using serine protease scaffold"
  author: "Your Name"
  date: "2024-01-01"
"""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(example_yaml)

        print(f"Created example configuration: {output_path}")
        print("\nEdit this file with your specific paths and parameters, then run:")
        print(f"  enzymeforge run {output_path}")

        return 0

    except Exception as e:
        print(f"Error creating example config: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="EnzymeForge: AI-powered enzyme design for environmental toxins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline from config file
  enzymeforge run config.yaml

  # Run with verbose logging
  enzymeforge run config.yaml --verbose

  # Analyze results
  enzymeforge analyze output/pfas_exp1

  # Show top designs by specific metric
  enzymeforge analyze output/pfas_exp1 --metric motif-ca-rmsd --top 5

  # Create example configuration
  enzymeforge init example_config.yaml

For more information: https://github.com/yourusername/EnzymeForge
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run enzyme design pipeline'
    )
    run_parser.add_argument(
        'config',
        type=Path,
        help='Path to YAML configuration file'
    )
    run_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze pipeline results'
    )
    analyze_parser.add_argument(
        'output_dir',
        type=Path,
        help='Pipeline output directory'
    )
    analyze_parser.add_argument(
        '--metric',
        default='mean-plddt',
        help='Metric to sort by (default: mean-plddt)'
    )
    analyze_parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top designs to show (default: 10)'
    )

    # Init command
    init_parser = subparsers.add_parser(
        'init',
        help='Create example configuration file'
    )
    init_parser.add_argument(
        'output',
        type=Path,
        nargs='?',
        default=Path('config.yaml'),
        help='Output path for example config (default: config.yaml)'
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'run':
        return run_pipeline(args.config, args.verbose)
    elif args.command == 'analyze':
        return analyze_results(args.output_dir, args.metric, args.top)
    elif args.command == 'init':
        return create_example_config(args.output)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

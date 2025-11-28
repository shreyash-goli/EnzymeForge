#!/usr/bin/env python
"""
Run EnzymeForge pipeline on SLURM cluster

Orchestrates multi-stage pipeline execution:
1. Diffusion (RFdiffusion/RFdiffusion all-atom)
2. Sequence Design (LigandMPNN + optional FastRelax)
3. Folding (ColabFold)

Adapted from ProtDesign2 with improvements.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enzymeforge.cluster import SlurmRunner, SlurmConfig, ContainerType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_cluster_config(config_path: Path) -> dict:
    """Load cluster configuration from YAML file

    Expected format:
    ```yaml
    cluster:
      partition: "gpu"
      email: "user@example.com"
      email_type: "ALL"

    containers:
      rfdiffusion: "/path/to/rfdiffusion.sif"
      rfdiffusion_aa: "/path/to/rfdiffusion_aa.sif"
      seqdesign: "/path/to/ligandmpnn_rosetta.sif"
      folding: "/path/to/colabfold.sif"

    binds:
      - "/data:/data"
      - "/scratch:/scratch"

    stages:
      diffusion:
        time: "01:00:00"
        mem: "4000"
        cpus: 2
        gpus: "a30:1"

      seqdesign:
        time: "03:00:00"
        mem: "8000"
        cpus: 9
        gpus: "a30:1"

      folding:
        time: "01:00:00"
        mem: "4000"
        cpus: 2
        gpus: "a30:1"
    ```

    Args:
        config_path: Path to cluster configuration YAML

    Returns:
        Dictionary with cluster configuration
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required = ["cluster", "containers", "stages"]
    for section in required:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in cluster config")

    return config


def create_stage_config(
    stage_config: dict,
    cluster_config: dict,
    container_path: Path,
    binds: list = None,
) -> SlurmConfig:
    """Create SlurmConfig for a pipeline stage

    Args:
        stage_config: Stage-specific configuration (time, mem, cpus, gpus)
        cluster_config: Global cluster configuration (partition, email)
        container_path: Path to container for this stage
        binds: List of bind mounts

    Returns:
        SlurmConfig object
    """
    return SlurmConfig(
        time=stage_config["time"],
        mem=stage_config["mem"],
        cpus=stage_config["cpus"],
        gpus=stage_config.get("gpus"),
        partition=cluster_config.get("partition"),
        email=cluster_config.get("email"),
        email_type=cluster_config.get("email_type", "ALL"),
        container_type=ContainerType.SINGULARITY,
        container_path=container_path,
        container_binds=binds,
    )


def run_pipeline(
    config_dir: Path,
    output_dir: Path,
    cluster_config_path: Path,
    repo_path: Path,
):
    """Run full pipeline on SLURM cluster

    Args:
        config_dir: Directory containing experiment config files
        output_dir: Directory for SLURM scripts and logs
        cluster_config_path: Path to cluster configuration
        repo_path: Path to EnzymeForge repository
    """
    # Load cluster configuration
    logger.info(f"Loading cluster config from {cluster_config_path}")
    cluster_cfg = load_cluster_config(cluster_config_path)

    # Get experiment config files
    config_files = sorted(config_dir.glob("*.yml")) + sorted(config_dir.glob("*.yaml"))

    if not config_files:
        logger.error(f"No config files found in {config_dir}")
        return

    logger.info(f"Found {len(config_files)} experiment configs")

    # Create SLURM runner
    runner = SlurmRunner()

    # Create output directories
    diffusion_dir = output_dir / "Diffusion"
    seqdesign_dir = output_dir / "SeqDesign"
    folding_dir = output_dir / "Folding"

    for d in [diffusion_dir, seqdesign_dir, folding_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1: Diffusion
    logger.info("Creating diffusion jobs...")
    diffusion_jobs = {}

    for config_file in config_files:
        # Load experiment config to determine container
        with open(config_file) as f:
            exp_config = yaml.safe_load(f)

        exp_name = exp_config["diffusion"]["name"]
        diffusion_type = exp_config["diffusion"].get("type", "base")

        # Select container based on diffusion type
        if diffusion_type == "all-atom":
            container = cluster_cfg["containers"]["rfdiffusion_aa"]
        else:
            container = cluster_cfg["containers"]["rfdiffusion"]

        # Create SLURM config
        slurm_config = create_stage_config(
            stage_config=cluster_cfg["stages"]["diffusion"],
            cluster_config=cluster_cfg["cluster"],
            container_path=Path(container),
            binds=cluster_cfg.get("binds"),
        )

        # Create job
        job_name = f"diff-{exp_name}"
        command = f"python scripts/run_diffusion.py --config {config_file}"

        job = runner.create_job_script(
            name=job_name,
            command=command,
            config=slurm_config,
            output_dir=diffusion_dir,
            working_dir=repo_path,
        )

        diffusion_jobs[exp_name] = job

    # Submit diffusion jobs
    logger.info(f"Submitting {len(diffusion_jobs)} diffusion jobs...")
    diffusion_job_ids = {}

    for exp_name, job in diffusion_jobs.items():
        try:
            job_id = runner.submit_job(job)
            diffusion_job_ids[exp_name] = job_id
            logger.info(f"  {exp_name}: {job_id}")
        except Exception as e:
            logger.error(f"Failed to submit {exp_name}: {e}")

    # Stage 2: Sequence Design (depends on diffusion)
    logger.info("Creating sequence design jobs...")
    seqdesign_jobs = {}

    seqdesign_config = create_stage_config(
        stage_config=cluster_cfg["stages"]["seqdesign"],
        cluster_config=cluster_cfg["cluster"],
        container_path=Path(cluster_cfg["containers"]["seqdesign"]),
        binds=cluster_cfg.get("binds"),
    )

    for config_file in config_files:
        with open(config_file) as f:
            exp_config = yaml.safe_load(f)

        exp_name = exp_config["diffusion"]["name"]

        # Skip if diffusion job failed
        if exp_name not in diffusion_job_ids:
            logger.warning(f"Skipping seqdesign for {exp_name} (diffusion job missing)")
            continue

        job_name = f"seqdesign-{exp_name}"
        command = f"python scripts/run_seqdesign.py --config {config_file}"

        job = runner.create_job_script(
            name=job_name,
            command=command,
            config=seqdesign_config,
            output_dir=seqdesign_dir,
            working_dir=repo_path,
            dependency_ids=[diffusion_job_ids[exp_name]],
        )

        seqdesign_jobs[exp_name] = job

    # Submit sequence design jobs
    logger.info(f"Submitting {len(seqdesign_jobs)} sequence design jobs...")
    seqdesign_job_ids = {}

    for exp_name, job in seqdesign_jobs.items():
        try:
            job_id = runner.submit_job(job)
            seqdesign_job_ids[exp_name] = job_id
            logger.info(f"  {exp_name}: {job_id}")
        except Exception as e:
            logger.error(f"Failed to submit {exp_name}: {e}")

    # Stage 3: Folding (depends on sequence design)
    logger.info("Creating folding jobs...")
    folding_jobs = {}

    folding_config = create_stage_config(
        stage_config=cluster_cfg["stages"]["folding"],
        cluster_config=cluster_cfg["cluster"],
        container_path=Path(cluster_cfg["containers"]["folding"]),
        binds=cluster_cfg.get("binds"),
    )

    for config_file in config_files:
        with open(config_file) as f:
            exp_config = yaml.safe_load(f)

        exp_name = exp_config["diffusion"]["name"]

        # Skip if seqdesign job failed
        if exp_name not in seqdesign_job_ids:
            logger.warning(f"Skipping folding for {exp_name} (seqdesign job missing)")
            continue

        job_name = f"folding-{exp_name}"
        command = f"python scripts/run_folding.py --config {config_file}"

        job = runner.create_job_script(
            name=job_name,
            command=command,
            config=folding_config,
            output_dir=folding_dir,
            working_dir=repo_path,
            dependency_ids=[seqdesign_job_ids[exp_name]],
        )

        folding_jobs[exp_name] = job

    # Submit folding jobs
    logger.info(f"Submitting {len(folding_jobs)} folding jobs...")
    folding_job_ids = {}

    for exp_name, job in folding_jobs.items():
        try:
            job_id = runner.submit_job(job)
            folding_job_ids[exp_name] = job_id
            logger.info(f"  {exp_name}: {job_id}")
        except Exception as e:
            logger.error(f"Failed to submit {exp_name}: {e}")

    # Summary
    logger.info("\n=== Pipeline Submission Complete ===")
    logger.info(f"Diffusion jobs: {len(diffusion_job_ids)}")
    logger.info(f"Sequence design jobs: {len(seqdesign_job_ids)}")
    logger.info(f"Folding jobs: {len(folding_job_ids)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run EnzymeForge pipeline on SLURM cluster"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing experiment config files (.yml/.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for SLURM scripts and logs",
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        required=True,
        help="Path to cluster configuration YAML",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Path to EnzymeForge repository (default: parent of scripts/)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.config_dir.exists():
        logger.error(f"Config directory not found: {args.config_dir}")
        sys.exit(1)

    if not args.cluster_config.exists():
        logger.error(f"Cluster config not found: {args.cluster_config}")
        sys.exit(1)

    # Run pipeline
    try:
        run_pipeline(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            cluster_config_path=args.cluster_config,
            repo_path=args.repo_path,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
SLURM cluster runner for EnzymeForge

Handles SLURM script generation, job submission, and dependency management.
Adapted from ProtDesign2 with improvements.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ContainerType(Enum):
    """Container type for execution"""
    SINGULARITY = "singularity"
    DOCKER = "docker"
    NONE = "none"


class JobStatus(Enum):
    """SLURM job status"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class SlurmConfig:
    """SLURM job configuration

    Attributes:
        time: Wall time limit (e.g., "01:00:00")
        mem: Memory in MB (e.g., "4000")
        cpus: CPUs per task (default: 1)
        gpus: GPU specification (e.g., "a30:1", default: None)
        partition: SLURM partition (default: None)
        email: Email for notifications (default: None)
        email_type: Email notification type (default: "ALL")
        container_type: Container type (default: SINGULARITY)
        container_path: Path to container (default: None)
        container_binds: Container bind mounts (default: None)
        extra_sbatch_args: Additional SBATCH arguments (default: None)
    """
    time: str
    mem: str
    cpus: int = 1
    gpus: Optional[str] = None
    partition: Optional[str] = None
    email: Optional[str] = None
    email_type: str = "ALL"
    container_type: ContainerType = ContainerType.SINGULARITY
    container_path: Optional[Path] = None
    container_binds: Optional[List[str]] = None
    extra_sbatch_args: Optional[Dict[str, str]] = None


@dataclass
class SlurmJob:
    """SLURM job information

    Attributes:
        name: Job name
        job_id: SLURM job ID (None if not submitted)
        config: SLURM configuration
        script_path: Path to SLURM script
        output_path: Path to stdout file
        error_path: Path to stderr file
        command: Command to execute
        working_dir: Working directory (default: None)
        dependency_ids: List of job IDs this job depends on (default: None)
        status: Job status (default: PENDING)
        metadata: Additional metadata (default: empty dict)
    """
    name: str
    job_id: Optional[str]
    config: SlurmConfig
    script_path: Path
    output_path: Path
    error_path: Path
    command: str
    working_dir: Optional[Path] = None
    dependency_ids: Optional[List[str]] = None
    status: JobStatus = JobStatus.PENDING
    metadata: Dict = field(default_factory=dict)


class SlurmRunner:
    """Run jobs on SLURM cluster

    Supports:
    - SLURM script generation
    - Job submission with dependencies
    - Container execution (Singularity/Docker)
    - Multi-stage pipeline orchestration
    """

    def __init__(self, sbatch_command: str = "/usr/bin/sbatch"):
        """Initialize SLURM runner

        Args:
            sbatch_command: Path to sbatch command (default: /usr/bin/sbatch)
        """
        self.sbatch_command = sbatch_command
        self.jobs: Dict[str, SlurmJob] = {}

    def create_job_script(
        self,
        name: str,
        command: str,
        config: SlurmConfig,
        output_dir: Path,
        working_dir: Optional[Path] = None,
        dependency_ids: Optional[List[str]] = None,
    ) -> SlurmJob:
        """Create SLURM job script

        Args:
            name: Job name (used for script filename and job-name)
            command: Command to execute
            config: SLURM configuration
            output_dir: Directory for SLURM scripts and output
            working_dir: Working directory for job execution
            dependency_ids: List of job IDs to depend on

        Returns:
            SlurmJob object with script details
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        script_path = output_dir / f"{name}.slurm"
        output_path = output_dir / f"{name}.out"
        error_path = output_dir / f"{name}.err"

        # Build SLURM script
        lines = [
            "#!/bin/bash\n",
            f"#SBATCH --job-name={name}\n",
            f"#SBATCH --output={output_path}\n",
            f"#SBATCH --error={error_path}\n",
            f"#SBATCH --time={config.time}\n",
            f"#SBATCH --mem={config.mem}\n",
            f"#SBATCH --cpus-per-task={config.cpus}\n",
        ]

        # Add GPU if specified
        if config.gpus:
            lines.append(f"#SBATCH --gres=gpu:{config.gpus}\n")

        # Add partition if specified
        if config.partition:
            lines.append(f"#SBATCH --partition={config.partition}\n")

        # Add email if specified
        if config.email:
            lines.append(f"#SBATCH --mail-user={config.email}\n")
            lines.append(f"#SBATCH --mail-type={config.email_type}\n")

        # Add dependencies if specified
        if dependency_ids:
            dep_str = ":".join(dependency_ids)
            lines.append(f"#SBATCH --dependency=afterok:{dep_str}\n")

        # Add extra SBATCH arguments
        if config.extra_sbatch_args:
            for key, value in config.extra_sbatch_args.items():
                lines.append(f"#SBATCH --{key}={value}\n")

        lines.append("\n")

        # Add working directory if specified
        if working_dir:
            lines.append(f"cd {working_dir}\n")

        # Wrap command in container if specified
        if config.container_type != ContainerType.NONE and config.container_path:
            command = self._wrap_in_container(command, config)

        lines.append(f"{command}\n")

        # Write script
        with open(script_path, 'w') as f:
            f.writelines(lines)

        logger.debug(f"Created SLURM script: {script_path}")

        # Create job object
        job = SlurmJob(
            name=name,
            job_id=None,
            config=config,
            script_path=script_path,
            output_path=output_path,
            error_path=error_path,
            command=command,
            working_dir=working_dir,
            dependency_ids=dependency_ids,
        )

        self.jobs[name] = job
        return job

    def _wrap_in_container(self, command: str, config: SlurmConfig) -> str:
        """Wrap command in container execution

        Args:
            command: Command to wrap
            config: SLURM configuration with container info

        Returns:
            Wrapped command string
        """
        if config.container_type == ContainerType.SINGULARITY:
            # Build singularity exec command
            cmd_parts = ["singularity", "exec"]

            # Add GPU support
            if config.gpus:
                cmd_parts.append("--nv")

            # Add bind mounts
            if config.container_binds:
                bind_str = ",".join(config.container_binds)
                cmd_parts.extend(["-B", bind_str])

            # Add cleanenv flag
            cmd_parts.append("--cleanenv")

            # Add container path
            cmd_parts.append(str(config.container_path))

            # Add command
            cmd_parts.append(command)

            return " ".join(cmd_parts)

        elif config.container_type == ContainerType.DOCKER:
            # Build docker run command
            cmd_parts = ["docker", "run"]

            # Add GPU support
            if config.gpus:
                cmd_parts.append("--gpus all")

            # Add bind mounts
            if config.container_binds:
                for bind in config.container_binds:
                    cmd_parts.extend(["-v", bind])

            # Add container path
            cmd_parts.append(str(config.container_path))

            # Add command
            cmd_parts.append(command)

            return " ".join(cmd_parts)

        return command

    def submit_job(self, job: SlurmJob) -> str:
        """Submit SLURM job

        Args:
            job: SlurmJob to submit

        Returns:
            Job ID from SLURM

        Raises:
            RuntimeError: If submission fails
        """
        cmd = f"{self.sbatch_command} {job.script_path}"

        logger.info(f"Submitting job: {job.name}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
            cwd=job.script_path.parent
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0 or stderr:
            raise RuntimeError(f"Failed to submit job {job.name}: {stderr}")

        # Extract job ID from output
        # Output format: "Submitted batch job 12345"
        job_id = stdout.strip().split()[-1]
        job.job_id = job_id
        job.status = JobStatus.PENDING

        logger.info(f"Job {job.name} submitted with ID {job_id}")

        return job_id

    def submit_all_jobs(self, output_dir: Path) -> Dict[str, str]:
        """Submit all jobs in output directory

        Args:
            output_dir: Directory containing SLURM scripts

        Returns:
            Dictionary mapping job names to job IDs
        """
        script_files = sorted(output_dir.glob("*.slurm"))
        job_ids = {}

        for script_path in script_files:
            name = script_path.stem

            if name not in self.jobs:
                logger.warning(f"No job object for {name}, skipping")
                continue

            job = self.jobs[name]

            try:
                job_id = self.submit_job(job)
                job_ids[name] = job_id
            except RuntimeError as e:
                logger.error(f"Failed to submit {name}: {e}")
                job_ids[name] = None

        return job_ids

    def create_pipeline_jobs(
        self,
        config_files: List[Path],
        stages: List[Dict],
        output_dir: Path,
        repo_path: Path,
    ) -> Dict[str, List[str]]:
        """Create multi-stage pipeline jobs

        Args:
            config_files: List of configuration files
            stages: List of stage configurations, each containing:
                - name: Stage name (e.g., "diffusion", "seqdesign")
                - command: Python command to run (e.g., "python diffuse.py")
                - config: SlurmConfig for this stage
                - job_prefix: Prefix for job names (e.g., "diff")
            output_dir: Base output directory for SLURM scripts
            repo_path: Path to repository (working directory)

        Returns:
            Dictionary mapping stage names to lists of job IDs
        """
        stage_job_ids = {}
        previous_stage_jobs = None

        for stage in stages:
            stage_name = stage["name"]
            stage_dir = output_dir / stage_name.capitalize()
            stage_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Creating jobs for stage: {stage_name}")

            jobs_for_stage = []

            for config_file in config_files:
                # Extract experiment name from config
                import yaml
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)

                # Get name from appropriate section
                if "diffusion" in config_data:
                    exp_name = config_data["diffusion"]["name"]
                else:
                    exp_name = config_file.stem

                job_name = f"{stage['job_prefix']}-{exp_name}"
                command = f"{stage['command']} --config {config_file}"

                # Create job with dependencies from previous stage
                job = self.create_job_script(
                    name=job_name,
                    command=command,
                    config=stage["config"],
                    output_dir=stage_dir,
                    working_dir=repo_path,
                    dependency_ids=previous_stage_jobs.get(exp_name) if previous_stage_jobs else None,
                )

                jobs_for_stage.append((exp_name, job))

            # Submit jobs and track IDs
            job_ids_map = {}
            for exp_name, job in jobs_for_stage:
                job_id = self.submit_job(job)
                job_ids_map[exp_name] = [job_id]

            stage_job_ids[stage_name] = job_ids_map
            previous_stage_jobs = job_ids_map

            logger.info(f"Submitted {len(jobs_for_stage)} jobs for {stage_name}")

        return stage_job_ids

    def check_job_status(self, job_id: str) -> JobStatus:
        """Check status of a SLURM job

        Args:
            job_id: SLURM job ID

        Returns:
            JobStatus enum
        """
        cmd = f"squeue -j {job_id} -h -o %T"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=True
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                # Job not in queue, check if completed or failed
                return JobStatus.COMPLETED

            status_str = stdout.strip()

            # Map SLURM status to JobStatus
            status_map = {
                "PENDING": JobStatus.PENDING,
                "RUNNING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "CANCELLED": JobStatus.CANCELLED,
                "COMPLETING": JobStatus.RUNNING,
            }

            return status_map.get(status_str, JobStatus.PENDING)

        except Exception as e:
            logger.error(f"Failed to check status for job {job_id}: {e}")
            return JobStatus.FAILED

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a SLURM job

        Args:
            job_id: SLURM job ID

        Returns:
            True if successful, False otherwise
        """
        cmd = f"scancel {job_id}"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=True
            )

            _, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to cancel job {job_id}: {stderr}")
                return False

            logger.info(f"Cancelled job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

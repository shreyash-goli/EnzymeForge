"""
Tests for SLURM cluster runner

Basic configuration and script generation tests.
Full integration tests requiring SLURM are separate.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from enzymeforge.cluster.slurm_runner import (
    SlurmRunner,
    SlurmConfig,
    SlurmJob,
    JobStatus,
    ContainerType,
)


class TestSlurmConfig:
    """Test SlurmConfig dataclass"""

    def test_minimal_config(self):
        """Test minimal configuration"""
        config = SlurmConfig(time="01:00:00", mem="4000")

        assert config.time == "01:00:00"
        assert config.mem == "4000"
        assert config.cpus == 1
        assert config.gpus is None
        assert config.partition is None
        assert config.email is None
        assert config.email_type == "ALL"
        assert config.container_type == ContainerType.SINGULARITY

    def test_full_config(self):
        """Test full configuration"""
        config = SlurmConfig(
            time="02:00:00",
            mem="8000",
            cpus=4,
            gpus="a30:1",
            partition="gpu",
            email="user@example.com",
            email_type="END",
            container_type=ContainerType.DOCKER,
            container_path=Path("/path/to/container.sif"),
            container_binds=["/data:/data", "/scratch:/scratch"],
        )

        assert config.time == "02:00:00"
        assert config.mem == "8000"
        assert config.cpus == 4
        assert config.gpus == "a30:1"
        assert config.partition == "gpu"
        assert config.email == "user@example.com"
        assert config.email_type == "END"
        assert config.container_type == ContainerType.DOCKER
        assert config.container_path == Path("/path/to/container.sif")
        assert config.container_binds == ["/data:/data", "/scratch:/scratch"]


class TestSlurmJob:
    """Test SlurmJob dataclass"""

    def test_basic_job(self):
        """Test basic job creation"""
        config = SlurmConfig(time="01:00:00", mem="4000")

        job = SlurmJob(
            name="test_job",
            job_id="12345",
            config=config,
            script_path=Path("/tmp/test_job.slurm"),
            output_path=Path("/tmp/test_job.out"),
            error_path=Path("/tmp/test_job.err"),
            command="python script.py",
        )

        assert job.name == "test_job"
        assert job.job_id == "12345"
        assert job.config == config
        assert job.script_path == Path("/tmp/test_job.slurm")
        assert job.command == "python script.py"
        assert job.working_dir is None
        assert job.dependency_ids is None
        assert job.status == JobStatus.PENDING

    def test_job_with_dependencies(self):
        """Test job with dependencies"""
        config = SlurmConfig(time="01:00:00", mem="4000")

        job = SlurmJob(
            name="dependent_job",
            job_id="12346",
            config=config,
            script_path=Path("/tmp/dependent_job.slurm"),
            output_path=Path("/tmp/dependent_job.out"),
            error_path=Path("/tmp/dependent_job.err"),
            command="python script2.py",
            dependency_ids=["12345"],
        )

        assert job.dependency_ids == ["12345"]


class TestSlurmRunner:
    """Test SlurmRunner"""

    def test_init(self):
        """Test initialization"""
        runner = SlurmRunner()
        assert runner.sbatch_command == "/usr/bin/sbatch"
        assert runner.jobs == {}

    def test_init_custom_sbatch(self):
        """Test initialization with custom sbatch"""
        runner = SlurmRunner(sbatch_command="/custom/path/sbatch")
        assert runner.sbatch_command == "/custom/path/sbatch"

    def test_create_job_script_minimal(self):
        """Test creating minimal job script"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = SlurmRunner()
            config = SlurmConfig(time="01:00:00", mem="4000")

            job = runner.create_job_script(
                name="test_job",
                command="python script.py",
                config=config,
                output_dir=output_dir,
            )

            assert job.name == "test_job"
            assert job.command == "python script.py"
            assert job.script_path.exists()

            # Read script and verify contents
            with open(job.script_path) as f:
                script = f.read()

            assert "#!/bin/bash" in script
            assert "#SBATCH --job-name=test_job" in script
            assert "#SBATCH --time=01:00:00" in script
            assert "#SBATCH --mem=4000" in script
            assert "#SBATCH --cpus-per-task=1" in script
            assert "python script.py" in script

    def test_create_job_script_with_gpu(self):
        """Test creating job script with GPU"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = SlurmRunner()
            config = SlurmConfig(
                time="01:00:00",
                mem="4000",
                gpus="a30:1",
            )

            job = runner.create_job_script(
                name="gpu_job",
                command="python train.py",
                config=config,
                output_dir=output_dir,
            )

            with open(job.script_path) as f:
                script = f.read()

            assert "#SBATCH --gres=gpu:a30:1" in script

    def test_create_job_script_with_dependencies(self):
        """Test creating job script with dependencies"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = SlurmRunner()
            config = SlurmConfig(time="01:00:00", mem="4000")

            job = runner.create_job_script(
                name="dependent_job",
                command="python process.py",
                config=config,
                output_dir=output_dir,
                dependency_ids=["12345", "12346"],
            )

            with open(job.script_path) as f:
                script = f.read()

            assert "#SBATCH --dependency=afterok:12345:12346" in script

    def test_create_job_script_with_email(self):
        """Test creating job script with email"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = SlurmRunner()
            config = SlurmConfig(
                time="01:00:00",
                mem="4000",
                email="user@example.com",
                email_type="END",
            )

            job = runner.create_job_script(
                name="email_job",
                command="python analyze.py",
                config=config,
                output_dir=output_dir,
            )

            with open(job.script_path) as f:
                script = f.read()

            assert "#SBATCH --mail-user=user@example.com" in script
            assert "#SBATCH --mail-type=END" in script

    def test_wrap_in_singularity_container(self):
        """Test wrapping command in Singularity container"""
        runner = SlurmRunner()
        config = SlurmConfig(
            time="01:00:00",
            mem="4000",
            container_type=ContainerType.SINGULARITY,
            container_path=Path("/containers/app.sif"),
            container_binds=["/data:/data", "/scratch:/scratch"],
        )

        wrapped = runner._wrap_in_container("python script.py", config)

        assert "singularity exec" in wrapped
        assert "--cleanenv" in wrapped
        assert "-B /data:/data,/scratch:/scratch" in wrapped
        assert "/containers/app.sif" in wrapped
        assert "python script.py" in wrapped

    def test_wrap_in_singularity_with_gpu(self):
        """Test wrapping command in Singularity with GPU"""
        runner = SlurmRunner()
        config = SlurmConfig(
            time="01:00:00",
            mem="4000",
            gpus="a30:1",
            container_type=ContainerType.SINGULARITY,
            container_path=Path("/containers/app.sif"),
        )

        wrapped = runner._wrap_in_container("python train.py", config)

        assert "singularity exec" in wrapped
        assert "--nv" in wrapped

    def test_wrap_in_docker_container(self):
        """Test wrapping command in Docker container"""
        runner = SlurmRunner()
        config = SlurmConfig(
            time="01:00:00",
            mem="4000",
            container_type=ContainerType.DOCKER,
            container_path=Path("myimage:latest"),
            container_binds=["/data:/data"],
        )

        wrapped = runner._wrap_in_container("python script.py", config)

        assert "docker run" in wrapped
        assert "-v /data:/data" in wrapped
        assert "myimage:latest" in wrapped
        assert "python script.py" in wrapped

    @patch('subprocess.Popen')
    def test_submit_job_success(self, mock_popen):
        """Test successful job submission"""
        # Mock successful submission
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Submitted batch job 12345", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = SlurmRunner()
            config = SlurmConfig(time="01:00:00", mem="4000")

            job = runner.create_job_script(
                name="test_job",
                command="python script.py",
                config=config,
                output_dir=output_dir,
            )

            job_id = runner.submit_job(job)

            assert job_id == "12345"
            assert job.job_id == "12345"
            assert job.status == JobStatus.PENDING

    @patch('subprocess.Popen')
    def test_submit_job_failure(self, mock_popen):
        """Test failed job submission"""
        # Mock failed submission
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "Error: invalid partition")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = SlurmRunner()
            config = SlurmConfig(time="01:00:00", mem="4000")

            job = runner.create_job_script(
                name="test_job",
                command="python script.py",
                config=config,
                output_dir=output_dir,
            )

            with pytest.raises(RuntimeError, match="Failed to submit"):
                runner.submit_job(job)

    @patch('subprocess.Popen')
    def test_check_job_status_running(self, mock_popen):
        """Test checking job status (running)"""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("RUNNING", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        runner = SlurmRunner()
        status = runner.check_job_status("12345")

        assert status == JobStatus.RUNNING

    @patch('subprocess.Popen')
    def test_check_job_status_completed(self, mock_popen):
        """Test checking job status (completed)"""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 1  # Job not in queue
        mock_popen.return_value = mock_process

        runner = SlurmRunner()
        status = runner.check_job_status("12345")

        assert status == JobStatus.COMPLETED

    @patch('subprocess.Popen')
    def test_cancel_job_success(self, mock_popen):
        """Test successful job cancellation"""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        runner = SlurmRunner()
        result = runner.cancel_job("12345")

        assert result is True

    @patch('subprocess.Popen')
    def test_cancel_job_failure(self, mock_popen):
        """Test failed job cancellation"""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "Error: invalid job id")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        runner = SlurmRunner()
        result = runner.cancel_job("12345")

        assert result is False


# Note: Full integration tests requiring SLURM should be in a separate
# test_slurm_integration.py file and run conditionally

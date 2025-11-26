"""
Tests for RFdiffusion runner

Tests configuration dataclasses and basic validation logic.
Integration tests requiring actual RFdiffusion installation are separate.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from enzymeforge.diffusion.rfdiffusion_runner import (
    DiffusionConfig,
    DiffusionResult,
    RFdiffusionRunner
)


class TestDiffusionConfig:
    """Test DiffusionConfig dataclass"""

    def test_basic_config(self):
        """Test basic configuration"""
        config = DiffusionConfig(
            contigs="50-50/A45-47/50-50",
            pdb=Path("input.pdb"),
            substrate="FOA",
            iterations=100,
            num_designs=20
        )

        assert config.contigs == "50-50/A45-47/50-50"
        assert config.pdb == Path("input.pdb")
        assert config.substrate == "FOA"
        assert config.iterations == 100
        assert config.num_designs == 20

    def test_default_values(self):
        """Test default configuration values"""
        config = DiffusionConfig(contigs="100-100")

        assert config.pdb is None
        assert config.substrate is None
        assert config.iterations == 50
        assert config.num_designs == 10
        assert config.guide_potentials is None
        assert config.guide_scale == 1.0
        assert config.noise_scale == 1.0
        assert config.deterministic is False
        assert config.partial_diffusion is False
        assert config.ckpt_override_path is None

    def test_enzyme_design_config(self):
        """Test configuration for enzyme design"""
        config = DiffusionConfig(
            contigs="50-50/A45-47/50-50",
            pdb=Path("input.pdb"),
            substrate="FOA",
            guide_potentials="type:substrate_contacts,weight:1.0",
            guide_scale=1.5
        )

        assert config.guide_potentials is not None
        assert config.guide_scale == 1.5


class TestDiffusionResult:
    """Test DiffusionResult dataclass"""

    def test_basic_result(self):
        """Test basic result creation"""
        result = DiffusionResult(
            design_id="pfas_0",
            pdb_path=Path("/output/pfas_0.pdb"),
            contig_string="50-50/A45-47/50-50"
        )

        assert result.design_id == "pfas_0"
        assert result.pdb_path == Path("/output/pfas_0.pdb")
        assert result.contig_string == "50-50/A45-47/50-50"
        assert result.rmsd is None
        assert result.metadata == {}

    def test_result_with_metadata(self):
        """Test result with metadata"""
        result = DiffusionResult(
            design_id="pfas_0",
            pdb_path=Path("/output/pfas_0.pdb"),
            contig_string="50-50/A45-47/50-50",
            rmsd=1.5,
            metadata={"iteration": 100, "temperature": 298}
        )

        assert result.rmsd == 1.5
        assert result.metadata["iteration"] == 100
        assert result.metadata["temperature"] == 298


class TestRFdiffusionRunnerInit:
    """Test RFdiffusionRunner initialization"""

    @patch('enzymeforge.diffusion.rfdiffusion_runner.check_path_exists')
    def test_init_valid_path(self, mock_check):
        """Test initialization with valid path"""
        mock_check.return_value = True

        runner = RFdiffusionRunner(
            rfdiffusion_path=Path("/path/to/RFdiffusion")
        )

        assert runner.rfdiffusion_path == Path("/path/to/RFdiffusion")
        assert runner.rfdiffusion_aa_path is None
        mock_check.assert_called_once()

    @patch('enzymeforge.diffusion.rfdiffusion_runner.check_path_exists')
    def test_init_with_allatom_path(self, mock_check):
        """Test initialization with all-atom path"""
        mock_check.return_value = True

        runner = RFdiffusionRunner(
            rfdiffusion_path=Path("/path/to/RFdiffusion"),
            rfdiffusion_aa_path=Path("/path/to/RFdiffusionAA")
        )

        assert runner.rfdiffusion_path == Path("/path/to/RFdiffusion")
        assert runner.rfdiffusion_aa_path == Path("/path/to/RFdiffusionAA")

    @patch('enzymeforge.diffusion.rfdiffusion_runner.check_path_exists')
    def test_init_invalid_path(self, mock_check):
        """Test initialization with invalid path"""
        mock_check.return_value = False

        with pytest.raises(ValueError, match="RFdiffusion path not found"):
            RFdiffusionRunner(rfdiffusion_path=Path("/invalid/path"))


class TestRFdiffusionRunnerMethods:
    """Test RFdiffusionRunner methods"""

    @patch('enzymeforge.diffusion.rfdiffusion_runner.check_path_exists')
    def setUp(self, mock_check):
        """Set up runner for tests"""
        mock_check.return_value = True
        return RFdiffusionRunner(
            rfdiffusion_path=Path("/path/to/RFdiffusion"),
            rfdiffusion_aa_path=Path("/path/to/RFdiffusionAA")
        )

    def test_collect_results(self):
        """Test _collect_results method"""
        runner = self.setUp()

        with patch('pathlib.Path.glob') as mock_glob:
            # Mock glob to return fake PDB files
            mock_glob.return_value = [
                Path("/output/pfas_0.pdb"),
                Path("/output/pfas_1.pdb"),
                Path("/output/pfas_2.pdb")
            ]

            results = runner._collect_results(
                output_path=Path("/output"),
                name="pfas",
                contigs="50-50/A45-47/50-50"
            )

            assert len(results) == 3
            assert results[0].design_id == "pfas_0"
            assert results[1].design_id == "pfas_1"
            assert results[2].design_id == "pfas_2"
            assert all(r.contig_string == "50-50/A45-47/50-50" for r in results)

    def test_allatom_requires_aa_path(self):
        """Test that run_diffusion_allatom requires all-atom path"""
        with patch('enzymeforge.diffusion.rfdiffusion_runner.check_path_exists'):
            runner = RFdiffusionRunner(
                rfdiffusion_path=Path("/path/to/RFdiffusion")
                # No all-atom path provided
            )

            config = DiffusionConfig(
                contigs="50-50/A45-47/50-50",
                pdb=Path("input.pdb"),
                substrate="FOA"
            )

            with pytest.raises(ValueError, match="RFdiffusion all-atom path not set"):
                runner.run_diffusion_allatom(
                    config=config,
                    name="test",
                    output_dir=Path("/output")
                )

    def test_allatom_requires_pdb(self):
        """Test that all-atom requires input PDB"""
        runner = self.setUp()

        config = DiffusionConfig(
            contigs="50-50/A45-47/50-50"
            # No PDB provided
        )

        with patch('enzymeforge.diffusion.rfdiffusion_runner.check_path_exists', return_value=True):
            with pytest.raises(ValueError, match="Input PDB required"):
                runner.run_diffusion_allatom(
                    config=config,
                    name="test",
                    output_dir=Path("/output")
                )

    def test_allatom_requires_substrate(self):
        """Test that all-atom requires substrate name"""
        runner = self.setUp()

        config = DiffusionConfig(
            contigs="50-50/A45-47/50-50",
            pdb=Path("input.pdb")
            # No substrate provided
        )

        with patch('enzymeforge.diffusion.rfdiffusion_runner.check_path_exists', return_value=True):
            with pytest.raises(ValueError, match="Substrate name required"):
                runner.run_diffusion_allatom(
                    config=config,
                    name="test",
                    output_dir=Path("/output")
                )


# Note: Integration tests that actually run RFdiffusion would require:
# - RFdiffusion installation
# - GPU availability
# - Sample PDB files
# These should be in a separate test_rfdiffusion_integration.py file

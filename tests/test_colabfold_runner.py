"""
Tests for ColabFold runner

Tests configuration dataclasses and helper methods.
Integration tests requiring actual ColabFold installation are separate.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from enzymeforge.validation.colabfold_runner import (
    FoldingConfig,
    FoldingResult,
    ColabFoldRunner
)


class TestFoldingConfig:
    """Test FoldingConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = FoldingConfig()

        assert config.msa_mode == "single_sequence"
        assert config.num_models == 2
        assert config.num_recycles == 3
        assert config.use_gpu is True
        assert config.max_msa is None

    def test_custom_config(self):
        """Test custom configuration"""
        config = FoldingConfig(
            msa_mode="mmseqs2_uniref_env",
            num_models=5,
            num_recycles=5,
            max_msa=512
        )

        assert config.msa_mode == "mmseqs2_uniref_env"
        assert config.num_models == 5
        assert config.num_recycles == 5
        assert config.max_msa == 512


class TestFoldingResult:
    """Test FoldingResult dataclass"""

    def test_basic_result(self):
        """Test basic result creation"""
        result = FoldingResult(
            design_id="pfas_0_n1_c0",
            sequence="MKSLLFTGHIKLMNPQRSTVWY",
            pdb_path=Path("/output/pfas_0_rank_001.pdb"),
            mean_plddt=85.5,
            mean_motif_plddt=90.2
        )

        assert result.design_id == "pfas_0_n1_c0"
        assert result.sequence == "MKSLLFTGHIKLMNPQRSTVWY"
        assert result.mean_plddt == 85.5
        assert result.mean_motif_plddt == 90.2
        assert result.ca_rmsd is None
        assert result.metadata == {}

    def test_result_with_metrics(self):
        """Test result with full metrics"""
        result = FoldingResult(
            design_id="pfas_0",
            sequence="MKSLLFT",
            pdb_path=Path("/output/pfas_0.pdb"),
            mean_plddt=85.0,
            mean_motif_plddt=90.0,
            ca_rmsd=1.5,
            motif_ca_rmsd=0.8,
            motif_all_atom_rmsd=1.2,
            contig_str="50-50/A45-47/50-50",
            motif_residues=["A51", "A52", "A53"]
        )

        assert result.ca_rmsd == 1.5
        assert result.motif_ca_rmsd == 0.8
        assert result.motif_all_atom_rmsd == 1.2
        assert result.contig_str == "50-50/A45-47/50-50"
        assert len(result.motif_residues) == 3


class TestColabFoldRunnerInit:
    """Test ColabFoldRunner initialization"""

    @patch('enzymeforge.validation.colabfold_runner.check_executable_exists')
    def test_init_default(self, mock_check):
        """Test initialization with default command"""
        mock_check.return_value = True

        runner = ColabFoldRunner()

        assert runner.colabfold_command == "colabfold_batch"
        mock_check.assert_called_once_with("colabfold_batch")

    @patch('enzymeforge.validation.colabfold_runner.check_executable_exists')
    def test_init_custom_command(self, mock_check):
        """Test initialization with custom command"""
        mock_check.return_value = True

        runner = ColabFoldRunner(colabfold_command="custom_colabfold")

        assert runner.colabfold_command == "custom_colabfold"
        mock_check.assert_called_once_with("custom_colabfold")

    @patch('enzymeforge.validation.colabfold_runner.check_executable_exists')
    def test_init_not_found_warning(self, mock_check):
        """Test warning when ColabFold not found"""
        mock_check.return_value = False

        # Should initialize but log warning
        runner = ColabFoldRunner()
        assert runner.colabfold_command == "colabfold_batch"


class TestColabFoldRunnerMethods:
    """Test ColabFoldRunner helper methods"""

    @patch('enzymeforge.validation.colabfold_runner.check_executable_exists')
    def setUp(self, mock_check):
        """Set up runner for tests"""
        mock_check.return_value = True
        return ColabFoldRunner()

    def test_read_fasta_sequences(self):
        """Test FASTA sequence reading"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fa"

            # Write sample FASTA file
            with open(fasta_path, 'w') as f:
                f.write(">pfas_0_n1_c0\n")
                f.write("MKSLLFTGHIKLMNP\n")
                f.write(">pfas_1_n2_c0\n")
                f.write("MKTLLFTGHIKLMNP\n")

            seq_dict = runner._read_fasta_sequences(fasta_path)

            assert len(seq_dict) == 2
            assert seq_dict["pfas_0_n1_c0"] == "MKSLLFTGHIKLMNP"
            assert seq_dict["pfas_1_n2_c0"] == "MKTLLFTGHIKLMNP"

    def test_get_plddt_scores(self):
        """Test pLDDT score extraction"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock JSON file
            json_data = {
                "plddt": [85.0, 90.0, 88.0, 92.0, 87.0, 85.0, 83.0, 89.0, 91.0, 86.0]
            }

            json_path = output_dir / "test_rank_001_model.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f)

            # Test full sequence pLDDT
            mean_plddt, mean_motif_plddt = runner._get_plddt_scores(
                output_dir=output_dir,
                motif_indices=[2, 3, 4]  # Indices for motif
            )

            assert mean_plddt == 87.6  # Mean of all scores
            assert mean_motif_plddt == 89.0  # Mean of indices 2,3,4 (88,92,87)

    def test_get_plddt_scores_no_json(self):
        """Test pLDDT extraction when no JSON file exists"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mean_plddt, mean_motif_plddt = runner._get_plddt_scores(
                output_dir=output_dir,
                motif_indices=[0, 1, 2]
            )

            assert mean_plddt == 0.0
            assert mean_motif_plddt == 0.0

    def test_postprocess_results(self):
        """Test result file organization"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock output files
            files_to_create = [
                "pfas_0_n1_c0_unrelaxed_rank_001_model_1.pdb",
                "pfas_0_n1_c0_scores_rank_001_model_1.json",
                "pfas_1_n2_c0_unrelaxed_rank_001_model_1.pdb",
                "pfas_1_n2_c0.done.txt",  # Should be removed
                "pfas_1_n2_c0.a3m",  # Should be removed
                "log.txt",  # Should be kept in main dir
            ]

            for filename in files_to_create:
                (output_dir / filename).touch()

            runner._postprocess_results(output_dir)

            # Check that .done.txt and .a3m were removed
            assert not (output_dir / "pfas_1_n2_c0.done.txt").exists()
            assert not (output_dir / "pfas_1_n2_c0.a3m").exists()

            # Check that log.txt is still in main directory
            assert (output_dir / "log.txt").exists()

            # Check that result files were moved to subdirectories
            assert (output_dir / "pfas_0_n1_c0").is_dir()
            assert (output_dir / "pfas_0_n1_c0" / "pfas_0_n1_c0_unrelaxed_rank_001_model_1.pdb").exists()
            assert (output_dir / "pfas_1_n2_c0").is_dir()
            assert (output_dir / "pfas_1_n2_c0" / "pfas_1_n2_c0_unrelaxed_rank_001_model_1.pdb").exists()

    def test_parse_results(self):
        """Test result parsing"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create subdirectories with results
            for i in range(2):
                subdir = output_dir / f"pfas_{i}_n1_c0"
                subdir.mkdir()

                # Create mock PDB and JSON
                (subdir / f"pfas_{i}_rank_001.pdb").touch()

                json_data = {"plddt": [85.0] * 10}
                with open(subdir / f"pfas_{i}_rank_001.json", 'w') as f:
                    json.dump(json_data, f)

            seq_dict = {
                "pfas_0_n1_c0": "MKSLLFT",
                "pfas_1_n1_c0": "MKTLLFT"
            }

            results = runner._parse_results(output_dir, seq_dict)

            assert len(results) == 2
            assert all(isinstance(r, FoldingResult) for r in results)
            assert results[0].mean_plddt == 85.0


class TestMetricsCalculation:
    """Test comprehensive metrics calculation"""

    @patch('enzymeforge.validation.colabfold_runner.check_executable_exists')
    @patch('enzymeforge.validation.colabfold_runner.get_ca_rmsd')
    @patch('enzymeforge.validation.colabfold_runner.get_motif_ca_rmsd')
    @patch('enzymeforge.validation.colabfold_runner.get_motif_all_atom_rmsd')
    def test_calculate_metrics(self, mock_all_atom, mock_motif_ca, mock_ca, mock_check):
        """Test comprehensive metrics calculation"""
        mock_check.return_value = True
        mock_ca.return_value = 1.5
        mock_motif_ca.return_value = 0.8
        mock_all_atom.return_value = 1.2

        runner = ColabFoldRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock JSON with pLDDT
            json_data = {
                "plddt": [85.0, 90.0, 88.0, 92.0, 87.0]
            }
            with open(output_dir / "test_rank_001.json", 'w') as f:
                json.dump(json_data, f)

            result = runner.calculate_metrics(
                folded_pdb=Path("/path/to/folded.pdb"),
                reference_pdb=Path("/path/to/ref.pdb"),
                diffusion_pdb=Path("/path/to/diff.pdb"),
                design_motif=["A2", "A3", "A4"],
                ref_motif=["A45", "A46", "A47"],
                sequence="MKSLL",
                contig_str="50-50/A45-47/50-50",
                design_id="pfas_0",
                output_dir=output_dir
            )

            assert result.design_id == "pfas_0"
            assert result.sequence == "MKSLL"
            assert result.mean_plddt == 88.4  # Mean of [85,90,88,92,87]
            assert result.mean_motif_plddt == 89.0  # Mean of indices 2,3,4 = [88,92,87]
            assert result.ca_rmsd == 1.5
            assert result.motif_ca_rmsd == 0.8
            assert result.motif_all_atom_rmsd == 1.2


# Note: Integration tests that actually run ColabFold would require:
# - ColabFold installation
# - Sample FASTA files
# - GPU availability (for speed)
# These should be in a separate test_colabfold_integration.py file

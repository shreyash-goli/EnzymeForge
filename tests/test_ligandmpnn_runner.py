"""
Tests for LigandMPNN runner

Tests configuration dataclasses and helper methods.
Integration tests requiring actual LigandMPNN installation are separate.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from enzymeforge.sequence.ligandmpnn_runner import (
    SequenceDesignConfig,
    SequenceDesignResult,
    LigandMPNNRunner
)


class TestSequenceDesignConfig:
    """Test SequenceDesignConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SequenceDesignConfig()

        assert config.model_type == "ligand_mpnn"
        assert config.num_seqs == 4
        assert config.temperature == 0.1
        assert config.seed is None
        assert config.pack_side_chains is True
        assert config.pack_with_ligand_context is True
        assert config.number_of_packs_per_design == 1
        assert config.repack_everything is False
        assert config.zero_indexed is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = SequenceDesignConfig(
            model_type="protein_mpnn",
            num_seqs=8,
            temperature=0.2,
            seed=42
        )

        assert config.model_type == "protein_mpnn"
        assert config.num_seqs == 8
        assert config.temperature == 0.2
        assert config.seed == 42


class TestSequenceDesignResult:
    """Test SequenceDesignResult dataclass"""

    def test_basic_result(self):
        """Test basic result creation"""
        result = SequenceDesignResult(
            design_id="pfas_0",
            pdb_path=Path("/output/pfas_0.pdb"),
            sequences=["MKSLLFT", "MKTLLFT", "MKNLLFT"]
        )

        assert result.design_id == "pfas_0"
        assert result.pdb_path == Path("/output/pfas_0.pdb")
        assert len(result.sequences) == 3
        assert result.fasta_path is None
        assert result.metadata == {}

    def test_result_with_fasta(self):
        """Test result with FASTA path"""
        result = SequenceDesignResult(
            design_id="pfas_0",
            pdb_path=Path("/output/pfas_0.pdb"),
            sequences=["MKSLLFT"],
            fasta_path=Path("/output/pfas_0.fa")
        )

        assert result.fasta_path == Path("/output/pfas_0.fa")

    def test_result_with_metadata(self):
        """Test result with metadata"""
        result = SequenceDesignResult(
            design_id="pfas_0",
            pdb_path=Path("/output/pfas_0.pdb"),
            sequences=["MKSLLFT"],
            metadata={"num_sequences": 1, "temperature": 0.1}
        )

        assert result.metadata["num_sequences"] == 1
        assert result.metadata["temperature"] == 0.1


class TestLigandMPNNRunnerInit:
    """Test LigandMPNNRunner initialization"""

    @patch('enzymeforge.sequence.ligandmpnn_runner.check_path_exists')
    def test_init_valid_path(self, mock_check):
        """Test initialization with valid path"""
        mock_check.return_value = True

        runner = LigandMPNNRunner(Path("/path/to/LigandMPNN"))

        assert runner.ligandmpnn_path == Path("/path/to/LigandMPNN")
        mock_check.assert_called_once()

    @patch('enzymeforge.sequence.ligandmpnn_runner.check_path_exists')
    def test_init_invalid_path(self, mock_check):
        """Test initialization with invalid path"""
        mock_check.return_value = False

        with pytest.raises(ValueError, match="LigandMPNN path not found"):
            LigandMPNNRunner(Path("/invalid/path"))


class TestLigandMPNNRunnerMethods:
    """Test LigandMPNNRunner helper methods"""

    @patch('enzymeforge.sequence.ligandmpnn_runner.check_path_exists')
    def setUp(self, mock_check):
        """Set up runner for tests"""
        mock_check.return_value = True
        return LigandMPNNRunner(Path("/path/to/LigandMPNN"))

    def test_write_json(self):
        """Test JSON file writing"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            test_data = {"key1": "value1", "key2": [1, 2, 3]}

            runner._write_json(output_path, test_data)

            assert output_path.exists()

            with open(output_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data

    def test_prepare_input_files(self):
        """Test JSON input file preparation"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            design_pdbs = [
                Path("/designs/pfas_0.pdb"),
                Path("/designs/pfas_1.pdb")
            ]
            design_motif = ["A51", "A52", "A53"]
            redesigned = ["A1", "A2", "A50", "A54", "A100"]

            runner._prepare_input_files(
                design_pdbs=design_pdbs,
                design_motif=design_motif,
                redesigned_residues=redesigned,
                output_dir=output_dir
            )

            # Check that all three JSON files were created
            assert (output_dir / "pdb_ids.json").exists()
            assert (output_dir / "fix_residues_multi.json").exists()
            assert (output_dir / "redesigned_residues_multi.json").exists()

            # Check content of pdb_ids.json
            with open(output_dir / "pdb_ids.json", 'r') as f:
                pdb_ids = json.load(f)
            assert len(pdb_ids) == 2
            assert "/designs/pfas_0.pdb" in pdb_ids

            # Check content of fix_residues_multi.json
            with open(output_dir / "fix_residues_multi.json", 'r') as f:
                fixed = json.load(f)
            assert fixed["/designs/pfas_0.pdb"] == design_motif

    def test_read_fasta(self):
        """Test FASTA file reading"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fa"

            # Write sample FASTA file
            with open(fasta_path, 'w') as f:
                f.write(">Header1\n")
                f.write(">Metadata line\n")
                f.write(">Seq1\n")
                f.write("MKSLLFTGHIKLMNP\n")
                f.write(">Seq2\n")
                f.write("MKTLLFTGHIKLMNP\n")

            sequences = runner._read_fasta(fasta_path)

            assert len(sequences) == 2
            assert sequences[0] == "MKSLLFTGHIKLMNP"
            assert sequences[1] == "MKTLLFTGHIKLMNP"

    def test_read_fasta_empty(self):
        """Test reading empty FASTA file"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "empty.fa"

            # Write minimal FASTA file
            with open(fasta_path, 'w') as f:
                f.write(">Header\n")
                f.write(">Metadata\n")

            sequences = runner._read_fasta(fasta_path)
            assert len(sequences) == 0

    def test_merge_fasta_files(self):
        """Test merging FASTA files"""
        runner = self.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            seqs_dir = output_dir / "seqs"
            seqs_dir.mkdir()

            # Create test FASTA files
            for i in range(2):
                fasta_path = seqs_dir / f"test_{i}.fa"
                with open(fasta_path, 'w') as f:
                    f.write(">Header\n")
                    f.write(">Metadata\n")
                    f.write(f">Seq{i}\n")
                    f.write(f"SEQUENCE{i}\n")

            merged_path = runner.merge_fasta_files(
                output_dir=output_dir,
                name="test",
                relax_round=0
            )

            assert merged_path.exists()
            assert merged_path.name == "test_c0.fa"

            # Check content
            with open(merged_path, 'r') as f:
                content = f.read()

            assert "SEQUENCE0" in content
            assert "SEQUENCE1" in content


class TestFiltering:
    """Test design filtering logic"""

    @patch('enzymeforge.sequence.ligandmpnn_runner.check_path_exists')
    @patch('enzymeforge.sequence.ligandmpnn_runner.StructureValidator')
    def test_filter_designs(self, mock_validator_class, mock_check):
        """Test design filtering"""
        mock_check.return_value = True

        # Mock validator
        mock_validator = MagicMock()
        mock_validator.filter_designs.return_value = (
            [Path("design_0.pdb"), Path("design_1.pdb")],  # passing designs
            [{"rmsd": 1.5}, {"rmsd": 1.8}]  # metrics
        )
        mock_validator_class.return_value = mock_validator

        runner = LigandMPNNRunner(Path("/path/to/LigandMPNN"))

        design_pdbs = [
            Path("design_0.pdb"),
            Path("design_1.pdb"),
            Path("design_2.pdb")
        ]

        filtered = runner._filter_designs(
            design_pdbs=design_pdbs,
            reference_pdb=Path("ref.pdb"),
            design_motif=["A51", "A52"],
            ref_motif=["A45", "A46"],
            ligand_name="FOA",
            rmsd_cutoff=2.0,
            clash_cutoff=2.0
        )

        assert len(filtered) == 2
        mock_validator.filter_designs.assert_called_once()


# Note: Integration tests that actually run LigandMPNN would require:
# - LigandMPNN installation
# - Sample PDB files with ligands
# - Proper environment setup
# These should be in a separate test_ligandmpnn_integration.py file

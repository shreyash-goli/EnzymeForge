"""
Tests for Rosetta FastRelax runner

Basic configuration and initialization tests.
Full integration tests requiring PyRosetta are separate.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from enzymeforge.relax.rosetta_runner import (
    RelaxConfig,
    RelaxResult,
    RosettaRunner
)


class TestRelaxConfig:
    """Test RelaxConfig dataclass"""

    def test_default_config(self):
        """Test default configuration"""
        config = RelaxConfig()

        assert config.params_file is None
        assert config.cst_file is None
        assert config.num_workers == 8
        assert config.constrain_to_start is True
        assert config.use_pdb2pqr is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = RelaxConfig(
            params_file=Path("/path/to/ligand.params"),
            cst_file=Path("/path/to/constraints.cst"),
            num_workers=4,
            constrain_to_start=False,
            use_pdb2pqr=False
        )

        assert config.params_file == Path("/path/to/ligand.params")
        assert config.cst_file == Path("/path/to/constraints.cst")
        assert config.num_workers == 4
        assert config.constrain_to_start is False
        assert config.use_pdb2pqr is False


class TestRelaxResult:
    """Test RelaxResult dataclass"""

    def test_basic_result(self):
        """Test basic result creation"""
        result = RelaxResult(
            design_id="pfas_0_n1_c1",
            pdb_path=Path("/output/pfas_0_n1_c1.pdb"),
            energy_before=1000.0,
            energy_after=900.0,
            energy_delta=-100.0
        )

        assert result.design_id == "pfas_0_n1_c1"
        assert result.pdb_path == Path("/output/pfas_0_n1_c1.pdb")
        assert result.energy_before == 1000.0
        assert result.energy_after == 900.0
        assert result.energy_delta == -100.0
        assert result.metadata == {}

    def test_result_with_metadata(self):
        """Test result with metadata"""
        result = RelaxResult(
            design_id="pfas_0",
            pdb_path=Path("/output/pfas_0.pdb"),
            energy_before=1000.0,
            energy_after=950.0,
            energy_delta=-50.0,
            metadata={"cycle": 1, "input_pdb": "/input/pfas_0.pdb"}
        )

        assert result.metadata["cycle"] == 1
        assert result.metadata["input_pdb"] == "/input/pfas_0.pdb"


class TestRosettaRunnerInit:
    """Test RosettaRunner initialization"""

    def test_init_without_dependencies(self):
        """Test initialization without checking dependencies"""
        runner = RosettaRunner(check_dependencies=False)

        assert runner.pyrosetta_available is False
        assert runner.pdb2pqr_available is False

    @patch('enzymeforge.relax.rosetta_runner.check_executable_exists')
    def test_init_with_pdb2pqr(self, mock_check):
        """Test initialization when pdb2pqr is available"""
        mock_check.return_value = True

        with patch.dict('sys.modules', {'pyrosetta': Mock()}):
            runner = RosettaRunner(check_dependencies=True)

            assert runner.pyrosetta_available is True
            assert runner.pdb2pqr_available is True

    @patch('enzymeforge.relax.rosetta_runner.check_executable_exists')
    def test_init_without_pyrosetta(self, mock_check):
        """Test initialization when PyRosetta is not available"""
        mock_check.return_value = False

        runner = RosettaRunner(check_dependencies=True)

        assert runner.pyrosetta_available is False

    def test_run_relax_without_pyrosetta_raises_error(self):
        """Test that running relax without PyRosetta raises error"""
        runner = RosettaRunner(check_dependencies=False)

        with pytest.raises(RuntimeError, match="PyRosetta not available"):
            runner.run_relax_cycle(
                pdb_files=[Path("/tmp/test.pdb")],
                config=RelaxConfig(),
                design_motif=["A51"],
                ref_motif=["A45"],
                ligand_name="FOA",
                output_dir=Path("/tmp/output"),
                cycle_number=1
            )


class TestLigandIndexExtraction:
    """Test ligand index extraction"""

    def test_get_ligand_index(self):
        """Test ligand index extraction from PDB"""
        import tempfile

        runner = RosettaRunner(check_dependencies=False)

        # Create mock PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N\n")
            f.write("HETATM  100  C1  FOA B   1      20.000  20.000  20.000  1.00 30.00           C\n")
            f.write("HETATM  101  C2  FOA B   1      21.000  21.000  21.000  1.00 30.00           C\n")
            pdb_path = Path(f.name)

        try:
            index = runner._get_ligand_index(pdb_path, "FOA")
            assert index == 1
        finally:
            pdb_path.unlink()

    def test_get_ligand_index_not_found(self):
        """Test ligand index extraction when ligand not found"""
        import tempfile

        runner = RosettaRunner(check_dependencies=False)

        # Create mock PDB file without ligand
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N\n")
            pdb_path = Path(f.name)

        try:
            index = runner._get_ligand_index(pdb_path, "FOA")
            assert index == 1  # Default value
        finally:
            pdb_path.unlink()


# Note: Full integration tests requiring PyRosetta should be in a separate
# test_rosetta_integration.py file and run conditionally based on PyRosetta availability

"""
Tests for pipeline orchestrator

Tests EnzymePipeline coordination of all components.
Integration tests requiring actual tools are separate.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from enzymeforge.pipeline.config import (
    PipelineConfig,
    SubstrateConfig,
    ActiveSiteConfig,
    ScaffoldConfig,
    FilteringConfig,
    ComputeConfig
)
from enzymeforge.pipeline.orchestrator import EnzymePipeline
from enzymeforge.diffusion.rfdiffusion_runner import DiffusionConfig, DiffusionResult
from enzymeforge.sequence.ligandmpnn_runner import SequenceDesignConfig, SequenceDesignResult
from enzymeforge.validation.colabfold_runner import FoldingConfig, FoldingResult


# Common decorator to mock all runner initializations
def mock_runners(func):
    """Decorator to mock all runner initializations"""
    return patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')(
        patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')(
            patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')(func)
        )
    )


@pytest.fixture
def minimal_config():
    """Create minimal valid pipeline configuration"""
    return PipelineConfig(
        name="test_pipeline",
        output_dir=Path("/tmp/test_output"),
        substrate=SubstrateConfig(
            name="PFOA",
            structure="CC(F)(F)C(=O)O",
            ligand_name="FOA"
        ),
        active_site=ActiveSiteConfig(
            mechanism="hydrolysis",
            catalytic_residues=["SER", "HIS", "ASP"]
        ),
        scaffold=ScaffoldConfig(min_size=100, max_size=150, num_designs=10),
        diffusion=DiffusionConfig(contigs="50-50"),
        sequence=SequenceDesignConfig(num_seqs=4),
        folding=FoldingConfig(num_models=2),
        filtering=FilteringConfig(rmsd_cutoff=2.0),
        compute=ComputeConfig(
            rfdiffusion_path=Path("/path/to/RFdiffusion"),
            ligandmpnn_path=Path("/path/to/LigandMPNN")
        )
    )


@pytest.fixture
def config_with_reference():
    """Create configuration with reference PDB"""
    return PipelineConfig(
        name="test_with_ref",
        output_dir=Path("/tmp/test_output"),
        substrate=SubstrateConfig(
            name="PFOA",
            structure="CC(F)(F)C(=O)O",
            ligand_name="FOA"
        ),
        active_site=ActiveSiteConfig(
            mechanism="hydrolysis",
            catalytic_residues=["SER", "HIS", "ASP"],
            reference_pdb=Path("/tmp/ref.pdb"),
            motif_positions=[45, 46, 47]
        ),
        scaffold=ScaffoldConfig(min_size=100, num_designs=5),
        diffusion=DiffusionConfig(contigs="48-48/A45-47/49-49"),
        sequence=SequenceDesignConfig(num_seqs=2),
        folding=FoldingConfig(num_models=2),
        filtering=FilteringConfig(rmsd_cutoff=2.0, filter_after_diffusion=True),
        compute=ComputeConfig(
            rfdiffusion_path=Path("/path/to/RFdiffusion"),
            ligandmpnn_path=Path("/path/to/LigandMPNN")
        )
    )


class TestEnzymePipelineInit:
    """Test EnzymePipeline initialization"""

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    def test_init_minimal(self, mock_colabfold, mock_mpnn, mock_rf, minimal_config):
        """Test initialization with minimal config"""
        pipeline = EnzymePipeline(minimal_config)

        assert pipeline.config == minimal_config
        assert pipeline.output_dir == minimal_config.output_dir
        assert pipeline.substrate is None
        assert pipeline.diffusion_results == []
        assert pipeline.sequence_results == []
        assert pipeline.folding_results == []

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    def test_init_creates_output_dir(self, mock_colabfold, mock_mpnn, mock_rf, minimal_config):
        """Test that initialization creates output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = minimal_config
            config.output_dir = Path(tmpdir) / "new_output"

            pipeline = EnzymePipeline(config)

            assert config.output_dir.exists()
            assert config.output_dir.is_dir()


class TestPipelineStages:
    """Test individual pipeline stages"""

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    @patch('enzymeforge.pipeline.orchestrator.SubstrateAnalyzer')
    @patch('enzymeforge.pipeline.orchestrator.ConstraintGenerator')
    def test_stage1_substrate_analysis(
        self,
        mock_constraint_gen_class,
        mock_analyzer_class,
        mock_colabfold,
        mock_mpnn,
        mock_rf,
        minimal_config
    ):
        """Test Stage 1: Substrate analysis"""
        # Mock substrate analyzer
        mock_analyzer = MagicMock()
        mock_substrate = Mock()
        mock_analyzer.load_substrate.return_value = mock_substrate
        mock_analyzer.identify_functional_groups.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        # Mock constraint generator
        mock_constraint_gen = MagicMock()
        mock_constraint_gen_class.return_value = mock_constraint_gen

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(minimal_config)

            pipeline._stage1_substrate_analysis()

            # Verify substrate was loaded
            mock_analyzer.load_substrate.assert_called_once()
            assert pipeline.substrate == mock_substrate

            # Verify contig file was created
            contig_file = Path(tmpdir) / f"{minimal_config.name}_contig.txt"
            assert contig_file.exists()

    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    def test_stage2_rfdiffusion(self, mock_runner_class, mock_mpnn, mock_colabfold, minimal_config):
        """Test Stage 2: RFdiffusion"""
        # Mock RFdiffusion runner
        mock_runner = MagicMock()
        mock_results = [
            DiffusionResult(
                design_id=f"test_{i}",
                pdb_path=Path(f"/tmp/test_{i}.pdb"),
                contig_string="50-50"
            )
            for i in range(3)
        ]
        mock_runner.run_diffusion.return_value = mock_results
        mock_runner_class.return_value = mock_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(minimal_config)

            pipeline._stage2_rfdiffusion()

            # Verify RFdiffusion was called
            mock_runner.run_diffusion.assert_called_once()
            assert len(pipeline.diffusion_results) == 3
            assert pipeline.stats["num_diffusion_designs"] == 3

    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.StructureValidator')
    @patch('enzymeforge.pipeline.orchestrator.get_motifs')
    def test_stage3_filter_diffusion(
        self,
        mock_get_motifs,
        mock_validator_class,
        mock_rf,
        mock_mpnn,
        mock_colabfold,
        config_with_reference
    ):
        """Test Stage 3: Quality filtering"""
        # Mock motif extraction
        mock_get_motifs.return_value = (
            ["A51", "A52", "A53"],  # design_motif
            ["A45", "A46", "A47"],  # ref_motif
            []  # redesigned
        )

        # Mock structure validator
        mock_validator = MagicMock()
        passing_pdbs = [Path("/tmp/test_0.pdb"), Path("/tmp/test_1.pdb")]
        mock_validator.filter_designs.return_value = (passing_pdbs, [{}] * 2)
        mock_validator_class.return_value = mock_validator

        with tempfile.TemporaryDirectory() as tmpdir:
            config_with_reference.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(config_with_reference)

            # Set up mock diffusion results
            pipeline.diffusion_results = [
                DiffusionResult(
                    design_id=f"test_{i}",
                    pdb_path=Path(f"/tmp/test_{i}.pdb"),
                    contig_string="48-48/A45-47/49-49"
                )
                for i in range(3)
            ]
            pipeline.stats["num_diffusion_designs"] = 3

            pipeline._stage3_filter_diffusion()

            # Verify filtering was called
            mock_validator.filter_designs.assert_called_once()

            # Verify only passing designs remain
            assert len(pipeline.diffusion_results) == 2
            assert pipeline.stats["num_filtered_after_diffusion"] == 2

    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    def test_stage4_sequence_design(self, mock_rf, mock_runner_class, mock_colabfold, minimal_config):
        """Test Stage 4: Sequence design"""
        # Mock LigandMPNN runner
        mock_runner = MagicMock()
        mock_results = [
            SequenceDesignResult(
                design_id=f"test_{i}",
                pdb_path=Path(f"/tmp/test_{i}.pdb"),
                sequences=["MKSLLFT", "MKTLLFT"]
            )
            for i in range(2)
        ]
        mock_runner.run_design.return_value = mock_results
        mock_runner_class.return_value = mock_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(minimal_config)

            # Set up mock diffusion results
            pipeline.diffusion_results = [
                DiffusionResult(
                    design_id=f"test_{i}",
                    pdb_path=Path(f"/tmp/test_{i}.pdb"),
                    contig_string="50-50"
                )
                for i in range(2)
            ]

            pipeline._stage4_sequence_design()

            # Verify sequence design was called
            mock_runner.run_design.assert_called_once()
            assert len(pipeline.sequence_results) == 2
            assert pipeline.stats["num_sequence_designs"] == 2

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    def test_stage5_structure_prediction(
        self,
        mock_mpnn_class,
        mock_colabfold_class,
        mock_rf,
        minimal_config
    ):
        """Test Stage 5: Structure prediction"""
        # Mock LigandMPNN runner for FASTA merging
        mock_mpnn = MagicMock()
        fasta_path = Path("/tmp/merged.fa")
        mock_mpnn.merge_fasta_files.return_value = fasta_path
        mock_mpnn_class.return_value = mock_mpnn

        # Mock ColabFold runner
        mock_colabfold = MagicMock()
        mock_results = [
            FoldingResult(
                design_id=f"test_{i}",
                sequence="MKSLLFT",
                pdb_path=Path(f"/tmp/test_{i}_rank_001.pdb"),
                mean_plddt=85.0,
                mean_motif_plddt=90.0
            )
            for i in range(2)
        ]
        mock_colabfold.run_folding.return_value = mock_results
        mock_colabfold_class.return_value = mock_colabfold

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(minimal_config)

            # Set up mock sequence results
            pipeline.sequence_results = [Mock() for _ in range(2)]

            pipeline._stage5_structure_prediction()

            # Verify ColabFold was called
            mock_colabfold.run_folding.assert_called_once()
            assert len(pipeline.folding_results) == 2

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    def test_stage6_final_scoring(self, mock_colabfold_class, mock_mpnn, mock_rf, config_with_reference):
        """Test Stage 6: Final scoring"""
        # Mock ColabFold runner
        mock_colabfold = MagicMock()

        # Mock sequence reading
        mock_colabfold._read_fasta_sequences.return_value = {
            "test_0": "MKSLLFT",
            "test_1": "MKTLLFT"
        }

        # Create mock scores file
        with tempfile.TemporaryDirectory() as tmpdir:
            folding_dir = Path(tmpdir) / config_with_reference.name / "Folding"
            folding_dir.mkdir(parents=True)
            scores_path = folding_dir / "scores.json"

            scores_data = {
                "test_0": {
                    "mean-plddt": 85.0,
                    "motif-ca-rmsd": 1.2
                },
                "test_1": {
                    "mean-plddt": 65.0,  # Below cutoff
                    "motif-ca-rmsd": 1.8
                }
            }

            with open(scores_path, 'w') as f:
                json.dump(scores_data, f)

            mock_colabfold.create_scores_file.return_value = scores_path
            mock_colabfold_class.return_value = mock_colabfold

            config_with_reference.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(config_with_reference)

            pipeline._stage6_final_scoring()

            # Verify scoring was called
            mock_colabfold.create_scores_file.assert_called_once()
            assert pipeline.stats["num_final_designs"] == 2
            assert pipeline.stats["num_high_quality"] == 1  # Only test_0 > 70.0


class TestPipelineResults:
    """Test result handling and analysis"""

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    def test_get_best_designs(self, mock_colabfold, mock_mpnn, mock_rf, minimal_config):
        """Test getting top designs by metric"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock scores file
            folding_dir = Path(tmpdir) / minimal_config.name / "Folding"
            folding_dir.mkdir(parents=True)
            scores_path = folding_dir / "scores.json"

            scores_data = {
                "design_0": {"id": "design_0", "mean-plddt": 90.0, "ca-rmsd": 1.0},
                "design_1": {"id": "design_1", "mean-plddt": 85.0, "ca-rmsd": 1.5},
                "design_2": {"id": "design_2", "mean-plddt": 95.0, "ca-rmsd": 0.8},
            }

            with open(scores_path, 'w') as f:
                json.dump(scores_data, f)

            minimal_config.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(minimal_config)

            # Get best designs by pLDDT (descending)
            best = pipeline.get_best_designs(n=2, metric="mean-plddt")

            assert len(best) == 2
            assert best[0]["id"] == "design_2"  # Highest pLDDT
            assert best[1]["id"] == "design_0"

            # Get best designs by RMSD (ascending)
            best_rmsd = pipeline.get_best_designs(n=2, metric="ca-rmsd")

            assert len(best_rmsd) == 2
            assert best_rmsd[0]["id"] == "design_2"  # Lowest RMSD

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    def test_save_results(self, mock_colabfold, mock_mpnn, mock_rf, minimal_config):
        """Test saving pipeline results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(minimal_config)

            # Set some statistics
            pipeline.stats["num_diffusion_designs"] = 10
            pipeline.stats["num_final_designs"] = 5

            pipeline._save_results()

            # Verify files were created
            results_dir = Path(tmpdir) / "results"
            assert results_dir.exists()

            stats_file = results_dir / f"{minimal_config.name}_statistics.json"
            assert stats_file.exists()

            config_file = results_dir / f"{minimal_config.name}_config.yaml"
            assert config_file.exists()

            summary_file = results_dir / f"{minimal_config.name}_summary.txt"
            assert summary_file.exists()

            # Verify statistics content
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            assert stats["num_diffusion_designs"] == 10
            assert stats["num_final_designs"] == 5

    @patch('enzymeforge.pipeline.orchestrator.RFdiffusionRunner')
    @patch('enzymeforge.pipeline.orchestrator.LigandMPNNRunner')
    @patch('enzymeforge.pipeline.orchestrator.ColabFoldRunner')
    def test_write_summary(self, mock_colabfold, mock_mpnn, mock_rf, minimal_config):
        """Test writing human-readable summary"""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = Path(tmpdir)
            pipeline = EnzymePipeline(minimal_config)

            pipeline.stats["num_diffusion_designs"] = 20
            pipeline.stats["num_final_designs"] = 10
            pipeline.stats["num_high_quality"] = 7

            summary_file = Path(tmpdir) / "summary.txt"
            pipeline._write_summary(summary_file)

            assert summary_file.exists()

            # Verify summary content
            with open(summary_file, 'r') as f:
                content = f.read()

            assert minimal_config.name in content
            assert "RFdiffusion designs: 20" in content
            assert "Final designs: 10" in content
            assert "High quality: 7" in content

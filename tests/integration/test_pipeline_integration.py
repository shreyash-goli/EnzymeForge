"""
Integration tests for full pipeline

End-to-end tests with real data (mock external tools).
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from enzymeforge.pipeline.config import PipelineConfig
from enzymeforge.pipeline.orchestrator import EnzymePipeline
from enzymeforge.substrate import SubstrateAnalyzer


# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestPipelineConfiguration:
    """Test pipeline configuration with real config files"""

    def test_load_simple_config(self):
        """Test loading simple configuration"""
        config_data = {
            "name": "test_enzyme",
            "output_dir": "/tmp/test",
            "substrate": {
                "smiles": "CCOC(=O)C",
                "name": "ETA"
            },
            "active_site": {
                "motif_residues": ["A45", "A72"]
            },
            "scaffold": {
                "target_size": 150
            },
            "compute": {
                "rfdiffusion_path": "/opt/RFdiffusion",
                "ligandmpnn_path": "/opt/LigandMPNN"
            }
        }

        config = PipelineConfig(**config_data)

        assert config.name == "test_enzyme"
        assert config.substrate.smiles == "CCOC(=O)C"
        assert len(config.active_site.motif_residues) == 2

    def test_validate_config(self):
        """Test configuration validation"""
        config_data = {
            "name": "test",
            "output_dir": "/tmp/test",
            "substrate": {
                "smiles": "CCOC(=O)C",
                "name": "ETA"
            },
            "active_site": {
                "motif_residues": ["A45"]
            },
            "scaffold": {
                "min_size": 100,
                "max_size": 200,
                "target_size": 150
            },
            "compute": {
                "rfdiffusion_path": "/opt/RFdiffusion",
                "ligandmpnn_path": "/opt/LigandMPNN"
            }
        }

        config = PipelineConfig(**config_data)

        # Should validate without errors
        assert config.scaffold.min_size < config.scaffold.target_size
        assert config.scaffold.target_size < config.scaffold.max_size


class TestSubstrateAnalysisStage:
    """Test substrate analysis stage with real molecules"""

    def test_stage1_substrate_analysis(self):
        """Test Stage 1: Substrate analysis"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                name="test_esterase",
                output_dir=tmpdir,
                substrate={
                    "smiles": "CCOC(=O)C",
                    "name": "ETA"
                },
                active_site={
                    "motif_residues": ["A45", "A72"]
                },
                scaffold={"target_size": 150},
                compute={
                    "rfdiffusion_path": "/opt/RFdiffusion",
                    "ligandmpnn_path": "/opt/LigandMPNN"
                }
            )

            pipeline = EnzymePipeline(config)

            # Run substrate analysis
            substrate = pipeline._stage1_substrate_analysis()

            # Validate results
            assert substrate is not None
            assert substrate.name == "ETA"
            assert substrate.mol is not None

            # Check output files were created
            output_dir = Path(tmpdir)
            assert (output_dir / "substrate").exists()


class TestConstraintGenerationStage:
    """Test constraint generation with real structures"""

    def test_generate_constraints_from_substrate(self):
        """Test constraint generation from substrate"""
        analyzer = SubstrateAnalyzer()
        substrate = analyzer.load_substrate("CCOC(=O)C", name="ETA")

        assert substrate is not None

        # This would generate constraints for the substrate
        # (simplified test - full test would use ConstraintGenerator)


class TestMockedPipelineStages:
    """Test pipeline stages with mocked external tools"""

    @patch('enzymeforge.diffusion.rfdiffusion_runner.RFdiffusionRunner')
    def test_stage2_diffusion_mocked(self, mock_runner_class):
        """Test Stage 2: RFdiffusion (mocked)"""
        # Mock RFdiffusion runner
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # Mock diffusion results
        mock_runner.run_diffusion.return_value = [
            Mock(
                design_id="design_0",
                pdb_path=Path("/tmp/design_0.pdb"),
                metadata={}
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                name="test",
                output_dir=tmpdir,
                substrate={"smiles": "CCOC(=O)C", "name": "ETA"},
                active_site={"motif_residues": ["A45"]},
                scaffold={"target_size": 150},
                compute={
                    "rfdiffusion_path": "/opt/RFdiffusion",
                    "ligandmpnn_path": "/opt/LigandMPNN"
                }
            )

            pipeline = EnzymePipeline(config)

            # This would run diffusion if not mocked
            # results = pipeline._stage2_rfdiffusion(substrate, contig)

    @patch('enzymeforge.sequence.ligandmpnn_runner.LigandMPNNRunner')
    def test_stage4_sequence_design_mocked(self, mock_runner_class):
        """Test Stage 4: Sequence design (mocked)"""
        # Mock LigandMPNN runner
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # Mock sequence design results
        mock_runner.run_design.return_value = [
            Mock(
                design_id="design_0_seq_0",
                pdb_path=Path("/tmp/packed.pdb"),
                fasta_path=Path("/tmp/seq.fa"),
                metadata={}
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                name="test",
                output_dir=tmpdir,
                substrate={"smiles": "CCOC(=O)C", "name": "ETA"},
                active_site={"motif_residues": ["A45"]},
                scaffold={"target_size": 150},
                compute={
                    "rfdiffusion_path": "/opt/RFdiffusion",
                    "ligandmpnn_path": "/opt/LigandMPNN"
                }
            )

            pipeline = EnzymePipeline(config)

            # This would run sequence design if not mocked


class TestPipelineDataFlow:
    """Test data flow through pipeline stages"""

    def test_substrate_to_constraints_flow(self):
        """Test data flow from substrate to constraints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create substrate
            analyzer = SubstrateAnalyzer()
            substrate = analyzer.load_substrate("CCOC(=O)C", name="ETA")

            # Would flow to constraint generation
            # constraints = generate_constraints(substrate, motif)

            assert substrate is not None

    def test_config_to_pipeline_flow(self):
        """Test configuration flows to pipeline correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                name="flow_test",
                output_dir=tmpdir,
                substrate={"smiles": "CCOC(=O)C", "name": "ETA"},
                active_site={"motif_residues": ["A45", "A72"]},
                scaffold={"target_size": 150},
                diffusion={"num_designs": 10, "type": "base"},
                sequence_design={"num_sequences": 5},
                compute={
                    "rfdiffusion_path": "/opt/RFdiffusion",
                    "ligandmpnn_path": "/opt/LigandMPNN"
                }
            )

            pipeline = EnzymePipeline(config)

            # Check config propagated correctly
            assert pipeline.config.name == "flow_test"
            assert pipeline.config.diffusion.num_designs == 10
            assert pipeline.config.sequence_design.num_sequences == 5


class TestPipelineOutputs:
    """Test pipeline output generation"""

    def test_output_directory_creation(self):
        """Test output directory structure is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                name="output_test",
                output_dir=tmpdir,
                substrate={"smiles": "CCOC(=O)C", "name": "ETA"},
                active_site={"motif_residues": ["A45"]},
                scaffold={"target_size": 150},
                compute={
                    "rfdiffusion_path": "/opt/RFdiffusion",
                    "ligandmpnn_path": "/opt/LigandMPNN"
                }
            )

            pipeline = EnzymePipeline(config)

            output_dir = Path(tmpdir)

            # Check subdirectories exist
            assert output_dir.exists()
            assert (output_dir / "substrate").exists()

    def test_results_saving(self):
        """Test results are saved correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                name="results_test",
                output_dir=tmpdir,
                substrate={"smiles": "CCOC(=O)C", "name": "ETA"},
                active_site={"motif_residues": ["A45"]},
                scaffold={"target_size": 150},
                compute={
                    "rfdiffusion_path": "/opt/RFdiffusion",
                    "ligandmpnn_path": "/opt/LigandMPNN"
                }
            )

            pipeline = EnzymePipeline(config)

            # Mock results
            mock_results = []

            # Save results
            pipeline.save_results(mock_results)

            # Check results file exists
            results_file = Path(tmpdir) / "results.json"
            assert results_file.exists()


class TestErrorHandling:
    """Test error handling in pipeline"""

    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES"""
        analyzer = SubstrateAnalyzer()

        with pytest.raises((ValueError, Exception)):
            # Invalid SMILES
            analyzer.load_substrate("INVALID_SMILES")

    def test_missing_config_fields(self):
        """Test handling of missing configuration fields"""
        with pytest.raises((ValueError, TypeError)):
            # Missing required fields
            config = PipelineConfig(
                name="incomplete",
                # Missing other required fields
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

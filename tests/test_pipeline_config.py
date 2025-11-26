"""
Tests for pipeline configuration

Tests PipelineConfig dataclass and YAML loading/saving.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from enzymeforge.pipeline.config import (
    PipelineConfig,
    SubstrateConfig,
    ActiveSiteConfig,
    ScaffoldConfig,
    FilteringConfig,
    ComputeConfig
)
from enzymeforge.diffusion.rfdiffusion_runner import DiffusionConfig
from enzymeforge.sequence.ligandmpnn_runner import SequenceDesignConfig
from enzymeforge.validation.colabfold_runner import FoldingConfig


class TestSubstrateConfig:
    """Test SubstrateConfig dataclass"""

    def test_default_config(self):
        """Test default substrate configuration"""
        config = SubstrateConfig(
            name="PFOA",
            structure="CC(F)(F)C(=O)O"
        )

        assert config.name == "PFOA"
        assert config.structure == "CC(F)(F)C(=O)O"
        assert config.format == "smiles"
        assert config.ligand_name is None

    def test_custom_config(self):
        """Test custom substrate configuration"""
        config = SubstrateConfig(
            name="FOA",
            structure="data/foa.pdb",
            format="pdb",
            ligand_name="FOA"
        )

        assert config.format == "pdb"
        assert config.ligand_name == "FOA"


class TestActiveSiteConfig:
    """Test ActiveSiteConfig dataclass"""

    def test_default_config(self):
        """Test default active site configuration"""
        config = ActiveSiteConfig(
            mechanism="hydrolysis",
            catalytic_residues=["SER", "HIS", "ASP"]
        )

        assert config.mechanism == "hydrolysis"
        assert len(config.catalytic_residues) == 3
        assert config.reference_pdb is None
        assert config.ref_chain == "A"

    def test_with_reference(self):
        """Test active site with reference PDB"""
        config = ActiveSiteConfig(
            mechanism="oxidation",
            catalytic_residues=["CYS", "HIS"],
            reference_pdb=Path("ref.pdb"),
            motif_positions=[45, 46, 47],
            ref_chain="B"
        )

        assert config.reference_pdb == Path("ref.pdb")
        assert config.motif_positions == [45, 46, 47]
        assert config.ref_chain == "B"


class TestScaffoldConfig:
    """Test ScaffoldConfig dataclass"""

    def test_default_config(self):
        """Test default scaffold configuration"""
        config = ScaffoldConfig()

        assert config.min_size == 100
        assert config.max_size == 150
        assert config.num_designs == 10

    def test_custom_config(self):
        """Test custom scaffold configuration"""
        config = ScaffoldConfig(
            min_size=80,
            max_size=120,
            num_designs=20
        )

        assert config.min_size == 80
        assert config.max_size == 120
        assert config.num_designs == 20


class TestFilteringConfig:
    """Test FilteringConfig dataclass"""

    def test_default_config(self):
        """Test default filtering configuration"""
        config = FilteringConfig()

        assert config.rmsd_cutoff == 2.0
        assert config.clash_cutoff == 2.0
        assert config.plddt_cutoff == 70.0
        assert config.motif_plddt_cutoff == 80.0
        assert config.filter_after_diffusion is True
        assert config.filter_after_sequence is True

    def test_custom_config(self):
        """Test custom filtering configuration"""
        config = FilteringConfig(
            rmsd_cutoff=1.5,
            plddt_cutoff=75.0,
            filter_after_diffusion=False
        )

        assert config.rmsd_cutoff == 1.5
        assert config.plddt_cutoff == 75.0
        assert config.filter_after_diffusion is False


class TestComputeConfig:
    """Test ComputeConfig dataclass"""

    def test_required_fields(self):
        """Test required compute configuration fields"""
        config = ComputeConfig(
            rfdiffusion_path=Path("/path/to/RFdiffusion"),
            ligandmpnn_path=Path("/path/to/LigandMPNN")
        )

        assert config.rfdiffusion_path == Path("/path/to/RFdiffusion")
        assert config.ligandmpnn_path == Path("/path/to/LigandMPNN")
        assert config.colabfold_command == "colabfold_batch"
        assert config.num_workers == 8

    def test_with_rfdiffusion_aa(self):
        """Test compute config with RFdiffusion all-atom"""
        config = ComputeConfig(
            rfdiffusion_path=Path("/path/to/RFdiffusion"),
            ligandmpnn_path=Path("/path/to/LigandMPNN"),
            rfdiffusion_aa_path=Path("/path/to/RFdiffusion_AA"),
            use_rfdiffusion_aa=True
        )

        assert config.rfdiffusion_aa_path == Path("/path/to/RFdiffusion_AA")
        assert config.use_rfdiffusion_aa is True


class TestPipelineConfig:
    """Test PipelineConfig dataclass"""

    def test_minimal_config(self):
        """Test minimal valid pipeline configuration"""
        config = PipelineConfig(
            name="test",
            output_dir=Path("output/test"),
            substrate=SubstrateConfig(name="PFOA", structure="CC(F)(F)C(=O)O"),
            active_site=ActiveSiteConfig(
                mechanism="hydrolysis",
                catalytic_residues=["SER", "HIS", "ASP"]
            ),
            scaffold=ScaffoldConfig(),
            diffusion=DiffusionConfig(contigs="50-50"),
            sequence=SequenceDesignConfig(),
            folding=FoldingConfig(),
            filtering=FilteringConfig(),
            compute=ComputeConfig(
                rfdiffusion_path=Path("/rf"),
                ligandmpnn_path=Path("/mpnn")
            )
        )

        assert config.name == "test"
        assert config.output_dir == Path("output/test")

    def test_validation_min_max_size(self):
        """Test validation of scaffold size"""
        with pytest.raises(ValueError, match="min_size.*must be <= max_size"):
            PipelineConfig(
                name="test",
                output_dir=Path("output/test"),
                substrate=SubstrateConfig(name="PFOA", structure="CC(F)(F)C(=O)O"),
                active_site=ActiveSiteConfig(
                    mechanism="hydrolysis",
                    catalytic_residues=["SER"]
                ),
                scaffold=ScaffoldConfig(min_size=150, max_size=100),  # Invalid
                diffusion=DiffusionConfig(contigs="50-50"),
                sequence=SequenceDesignConfig(),
                folding=FoldingConfig(),
                filtering=FilteringConfig(),
                compute=ComputeConfig(
                    rfdiffusion_path=Path("/rf"),
                    ligandmpnn_path=Path("/mpnn")
                )
            )

    def test_validation_motif_positions(self):
        """Test validation of motif positions with reference PDB"""
        with pytest.raises(ValueError, match="motif_positions required"):
            PipelineConfig(
                name="test",
                output_dir=Path("output/test"),
                substrate=SubstrateConfig(name="PFOA", structure="CC(F)(F)C(=O)O"),
                active_site=ActiveSiteConfig(
                    mechanism="hydrolysis",
                    catalytic_residues=["SER"],
                    reference_pdb=Path("ref.pdb")
                    # Missing motif_positions
                ),
                scaffold=ScaffoldConfig(),
                diffusion=DiffusionConfig(contigs="50-50"),
                sequence=SequenceDesignConfig(),
                folding=FoldingConfig(),
                filtering=FilteringConfig(),
                compute=ComputeConfig(
                    rfdiffusion_path=Path("/rf"),
                    ligandmpnn_path=Path("/mpnn")
                )
            )

    def test_contig_string_generation(self):
        """Test automatic contig string generation"""
        config = PipelineConfig(
            name="test",
            output_dir=Path("output/test"),
            substrate=SubstrateConfig(name="PFOA", structure="CC(F)(F)C(=O)O"),
            active_site=ActiveSiteConfig(
                mechanism="hydrolysis",
                catalytic_residues=["SER", "HIS", "ASP"],
                reference_pdb=Path("ref.pdb"),
                motif_positions=[45, 46, 47]
            ),
            scaffold=ScaffoldConfig(min_size=100),
            diffusion=DiffusionConfig(contigs=""),  # Empty, should be auto-generated
            sequence=SequenceDesignConfig(),
            folding=FoldingConfig(),
            filtering=FilteringConfig(),
            compute=ComputeConfig(
                rfdiffusion_path=Path("/rf"),
                ligandmpnn_path=Path("/mpnn")
            )
        )

        # Should auto-generate contig string
        assert config.diffusion.contigs != ""
        assert "A45-47" in config.diffusion.contigs

    def test_yaml_roundtrip(self):
        """Test saving and loading configuration from YAML"""
        original_config = PipelineConfig(
            name="test_yaml",
            output_dir=Path("output/test"),
            substrate=SubstrateConfig(
                name="PFOA",
                structure="CC(F)(F)C(=O)O",
                format="smiles",
                ligand_name="FOA"
            ),
            active_site=ActiveSiteConfig(
                mechanism="hydrolysis",
                catalytic_residues=["SER", "HIS", "ASP"],
                reference_pdb=Path("ref.pdb"),
                motif_positions=[45, 46, 47]
            ),
            scaffold=ScaffoldConfig(min_size=100, max_size=150, num_designs=10),
            diffusion=DiffusionConfig(
                contigs="48-48/A45-47/49-49",
                iterations=100,
                num_designs=10
            ),
            sequence=SequenceDesignConfig(num_seqs=4),
            folding=FoldingConfig(num_models=2),
            filtering=FilteringConfig(rmsd_cutoff=2.0),
            compute=ComputeConfig(
                rfdiffusion_path=Path("/rf"),
                ligandmpnn_path=Path("/mpnn")
            ),
            metadata={"author": "test"}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"

            # Save to YAML
            original_config.to_yaml(yaml_path)
            assert yaml_path.exists()

            # Load from YAML
            loaded_config = PipelineConfig.from_yaml(yaml_path)

            # Verify key fields match
            assert loaded_config.name == original_config.name
            assert loaded_config.substrate.name == original_config.substrate.name
            assert loaded_config.active_site.mechanism == original_config.active_site.mechanism
            assert loaded_config.scaffold.min_size == original_config.scaffold.min_size
            assert loaded_config.metadata == original_config.metadata

    def test_yaml_format(self):
        """Test that generated YAML is human-readable"""
        config = PipelineConfig(
            name="test",
            output_dir=Path("output/test"),
            substrate=SubstrateConfig(name="PFOA", structure="CC(F)(F)C(=O)O"),
            active_site=ActiveSiteConfig(
                mechanism="hydrolysis",
                catalytic_residues=["SER"]
            ),
            scaffold=ScaffoldConfig(),
            diffusion=DiffusionConfig(contigs="50-50"),
            sequence=SequenceDesignConfig(),
            folding=FoldingConfig(),
            filtering=FilteringConfig(),
            compute=ComputeConfig(
                rfdiffusion_path=Path("/rf"),
                ligandmpnn_path=Path("/mpnn")
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            # Read and verify YAML content
            with open(yaml_path, 'r') as f:
                content = f.read()

            # Should contain readable sections
            assert "name: test" in content
            assert "substrate:" in content
            assert "active_site:" in content
            assert "mechanism: hydrolysis" in content

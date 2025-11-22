"""
Tests for constraint generator module
"""

import pytest
import tempfile
from pathlib import Path

from enzymeforge.substrate import Substrate, CatalyticSite, ConstraintGenerator


class TestConstraintGenerator:
    """Test ConstraintGenerator functionality"""

    @pytest.fixture
    def sample_substrate(self):
        """Create sample substrate for testing"""
        return Substrate(
            name="PFOA",
            structure="C(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            format="smiles"
        )

    @pytest.fixture
    def sample_active_site(self):
        """Create sample active site for testing"""
        return CatalyticSite(
            mechanism="hydrolysis",
            catalytic_residues=["SER", "HIS", "ASP"],
            geometry_constraints={}
        )

    def test_initialization(self, sample_substrate, sample_active_site):
        """Test ConstraintGenerator initialization"""
        generator = ConstraintGenerator(sample_substrate, sample_active_site)

        assert generator.substrate == sample_substrate
        assert generator.active_site == sample_active_site

    def test_generate_contig_string_with_motif(self, sample_substrate, sample_active_site):
        """Test contig string generation with motif"""
        generator = ConstraintGenerator(sample_substrate, sample_active_site)

        contig = generator.generate_contig_string(
            scaffold_size=(100, 150),
            motif_positions=[45, 46, 47]
        )

        # Should have format: "N-term/motif/C-term"
        assert "/" in contig
        assert "A45-47" in contig

        # Parse the parts
        parts = contig.split("/")
        assert len(parts) == 3

    def test_generate_contig_string_no_motif(self, sample_substrate, sample_active_site):
        """Test contig string generation without motif"""
        generator = ConstraintGenerator(sample_substrate, sample_active_site)

        contig = generator.generate_contig_string(
            scaffold_size=(100, 150),
            motif_positions=[]
        )

        # Should just be de novo design
        assert "-" in contig
        assert "100-150" == contig

    def test_generate_cst_file(self, sample_substrate, sample_active_site):
        """Test constraint file generation"""
        generator = ConstraintGenerator(sample_substrate, sample_active_site)

        with tempfile.TemporaryDirectory() as tmpdir:
            cst_path = Path(tmpdir) / "test.cst"

            output_path = generator.generate_cst_file(str(cst_path))

            assert Path(output_path).exists()

            # Read and verify content
            with open(output_path) as f:
                content = f.read()

            assert "AtomPair" in content
            assert "HARMONIC" in content
            # Should have constraints for 3 catalytic residues
            lines = content.strip().split('\n')
            assert len(lines) > 0

    def test_generate_cst_file_custom_distance(self, sample_substrate, sample_active_site):
        """Test constraint file with custom distance cutoff"""
        generator = ConstraintGenerator(sample_substrate, sample_active_site)

        with tempfile.TemporaryDirectory() as tmpdir:
            cst_path = Path(tmpdir) / "test.cst"

            output_path = generator.generate_cst_file(
                str(cst_path),
                distance_cutoff=5.0
            )

            with open(output_path) as f:
                content = f.read()

            # Check that distance cutoff is used
            assert "5.0" in content

    def test_generate_guide_potential(self, sample_substrate, sample_active_site):
        """Test guide potential generation"""
        generator = ConstraintGenerator(sample_substrate, sample_active_site)

        potential = generator.generate_guide_potential()

        assert "substrate_contacts" in potential
        assert "weight" in potential
        assert "substrate_chain" in potential

    def test_generate_motif_definition(self, sample_substrate, sample_active_site):
        """Test motif definition generation"""
        generator = ConstraintGenerator(sample_substrate, sample_active_site)

        motif_def = generator.generate_motif_definition([45, 46, 47])

        assert "positions" in motif_def
        assert "residue_types" in motif_def
        assert "chain" in motif_def

        assert motif_def["positions"] == [45, 46, 47]
        assert motif_def["residue_types"] == ["SER", "HIS", "ASP"]
        assert motif_def["chain"] == "A"

"""
Integration tests for constraint generation

Tests constraint file generation with real structures.
"""

import pytest
import tempfile
from pathlib import Path

from enzymeforge.substrate.constraint_generator import ConstraintGenerator


# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestConstraintGeneration:
    """Integration tests for constraint file generation"""

    def test_generate_contig_string_basic(self):
        """Test contig string generation with basic motif"""
        generator = ConstraintGenerator()

        contig = generator.generate_contig_string(
            motif_residues=["A45", "A72", "A105"],
            scaffold_length=150
        )

        assert contig is not None
        assert "A45-45" in contig
        assert "A72-72" in contig
        assert "A105-105" in contig

        # Should have gaps between motif residues
        assert "/0" in contig or "/" in contig

    def test_generate_contig_with_reference(self):
        """Test contig generation with reference structure"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        contig = generator.generate_contig_string(
            motif_residues=["A45", "A72", "A105"],
            scaffold_length=150
        )

        assert contig is not None
        assert "45" in contig
        assert "72" in contig
        assert "105" in contig

    def test_generate_cst_file(self):
        """Test constraint file generation"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_constraints.cst"

            generator.generate_cst_file(
                motif_residues=["A45", "A72", "A105"],
                ligand_name="LIG",
                output_file=output_file
            )

            # Check file was created
            assert output_file.exists()

            # Check file contents
            with open(output_file) as f:
                content = f.read()

            # Should contain AtomPair constraints
            assert "AtomPair" in content or "HARMONIC" in content

    def test_generate_cst_file_custom_distances(self):
        """Test constraint file with custom distance parameters"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "custom_constraints.cst"

            generator.generate_cst_file(
                motif_residues=["A45", "A72"],
                ligand_name="LIG",
                output_file=output_file,
                distance_cutoff=5.0  # Custom distance
            )

            assert output_file.exists()

    def test_generate_guide_potential_config(self):
        """Test guide potential configuration generation"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "guide_potential.yml"

            guide_config = generator.generate_guide_potential(
                substrate_name="LIG",
                catalytic_residues=["A45", "A72"],
                output_file=output_file
            )

            # Check returned config
            assert guide_config is not None
            assert isinstance(guide_config, dict)

            # Check file was created
            assert output_file.exists()

    def test_motif_definition(self):
        """Test motif definition generation for RFdiffusion"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "motif.pdb"

            generator.generate_motif_definition(
                motif_residues=["A45", "A72", "A105"],
                output_file=output_file
            )

            # Check file was created
            assert output_file.exists()

            # Check it's a valid PDB
            with open(output_file) as f:
                content = f.read()

            assert "ATOM" in content or "HETATM" in content


class TestConstraintValidation:
    """Test constraint validation and error handling"""

    def test_invalid_residue_format(self):
        """Test error handling for invalid residue format"""
        generator = ConstraintGenerator()

        with pytest.raises((ValueError, IndexError)):
            # Invalid format (missing chain)
            generator.generate_contig_string(
                motif_residues=["45"],  # Should be "A45"
                scaffold_length=150
            )

    def test_empty_motif_residues(self):
        """Test handling of empty motif"""
        generator = ConstraintGenerator()

        contig = generator.generate_contig_string(
            motif_residues=[],
            scaffold_length=150
        )

        # Should still generate a contig for scaffold only
        assert contig is not None
        assert "150" in contig or "100-200" in contig


class TestConstraintFormats:
    """Test different constraint file formats"""

    def test_rosetta_cst_format(self):
        """Test Rosetta constraint file format"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "rosetta.cst"

            generator.generate_cst_file(
                motif_residues=["A45", "A72"],
                ligand_name="LIG",
                output_file=output_file
            )

            # Read and validate format
            with open(output_file) as f:
                lines = f.readlines()

            # Should have constraint lines
            assert len(lines) > 0

            # Check Rosetta format (basic validation)
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    # Should have multiple fields
                    parts = line.split()
                    assert len(parts) >= 3


class TestRealWorldScenarios:
    """Test constraint generation for real-world scenarios"""

    def test_esterase_constraints(self):
        """Test constraint generation for esterase design"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        # Catalytic triad
        motif_residues = ["A45", "A72", "A105"]  # Ser-His-Asn

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate contig
            contig = generator.generate_contig_string(
                motif_residues=motif_residues,
                scaffold_length=150
            )
            assert contig is not None

            # Generate constraints
            cst_file = Path(tmpdir) / "esterase.cst"
            generator.generate_cst_file(
                motif_residues=motif_residues,
                ligand_name="LIG",
                output_file=cst_file
            )
            assert cst_file.exists()

            # Generate guide potential
            guide_file = Path(tmpdir) / "guide.yml"
            generator.generate_guide_potential(
                substrate_name="LIG",
                catalytic_residues=motif_residues,
                output_file=guide_file
            )
            assert guide_file.exists()

    def test_lactonase_constraints(self):
        """Test constraint generation for lactonase design"""
        ref_pdb = TEST_DATA_DIR / "reference_enzyme.pdb"

        if not ref_pdb.exists():
            pytest.skip("Reference PDB not found")

        generator = ConstraintGenerator(reference_pdb=str(ref_pdb))

        # Lactonase active site
        motif_residues = ["A45", "A72", "A105"]

        with tempfile.TemporaryDirectory() as tmpdir:
            contig = generator.generate_contig_string(
                motif_residues=motif_residues,
                scaffold_length=200  # Larger scaffold
            )

            assert contig is not None
            assert "200" in contig or "150-250" in contig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

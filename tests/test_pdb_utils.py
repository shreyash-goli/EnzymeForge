"""
Tests for PDB utilities

Tests adapted PDB manipulation functions from ProtDesign2
"""

import pytest
from pathlib import Path

from enzymeforge.utils.pdb_utils import (
    get_motifs,
    atom_sort_key,
)


class TestGetMotifs:
    """Test contig string parsing"""

    def test_simple_motif(self):
        """Test basic motif parsing"""
        design_motif, ref_motif, redesigned = get_motifs("50-50/A45-47/50-50")

        # Motif should be at positions 51-53 in design
        assert design_motif == ["A51", "A52", "A53"]
        # Reference motif from A45-47
        assert ref_motif == ["A45", "A46", "A47"]
        # Total length = 50 + 3 + 50 = 103
        assert len(design_motif) + len(redesigned) == 103

    def test_single_residue_motif(self):
        """Test single-residue motif"""
        design_motif, ref_motif, redesigned = get_motifs("10-10/A25-25/10-10")

        assert design_motif == ["A11"]
        assert ref_motif == ["A25"]
        assert len(design_motif) + len(redesigned) == 21

    def test_no_motif(self):
        """Test de novo design with no motif"""
        design_motif, ref_motif, redesigned = get_motifs("100-100")

        assert design_motif == []
        assert ref_motif == []
        assert len(redesigned) == 100

    def test_multiple_motifs(self):
        """Test multiple motif blocks"""
        design_motif, ref_motif, redesigned = get_motifs("20-20/A10-12/20-20/B30-31/20-20")

        # First motif at 21-23, second at 44-45
        assert "A21" in design_motif
        assert "A22" in design_motif
        assert "A23" in design_motif
        assert "A44" in design_motif
        assert "A45" in design_motif

        assert "A10" in ref_motif
        assert "A11" in ref_motif
        assert "A12" in ref_motif
        assert "B30" in ref_motif
        assert "B31" in ref_motif

        # Total = 20 + 3 + 20 + 2 + 20 = 65
        assert len(design_motif) + len(redesigned) == 65

    def test_chain_break_ignored(self):
        """Test that chain breaks (0) are ignored"""
        design_motif, ref_motif, redesigned = get_motifs("50-50/A45-47/0/50-50")

        # Should be same as without the 0
        assert design_motif == ["A51", "A52", "A53"]
        assert ref_motif == ["A45", "A46", "A47"]

    def test_redesigned_residues_correct(self):
        """Test that redesigned residues don't include motif"""
        design_motif, ref_motif, redesigned = get_motifs("5-5/A10-11/5-5")

        # Positions 1-5 are N-term, 6-7 are motif, 8-12 are C-term
        assert "A1" in redesigned
        assert "A5" in redesigned
        assert "A6" not in redesigned  # This is motif
        assert "A7" not in redesigned  # This is motif
        assert "A8" in redesigned
        assert "A12" in redesigned


class TestAtomSortKey:
    """Test atom sorting function"""

    def test_atom_sort_key(self):
        """Test that atom_sort_key returns atom id"""
        # Mock atom object
        class MockAtom:
            def __init__(self, atom_id):
                self.id = atom_id

        ca_atom = MockAtom("CA")
        n_atom = MockAtom("N")

        assert atom_sort_key(ca_atom) == "CA"
        assert atom_sort_key(n_atom) == "N"


# Note: Tests requiring actual PDB files would go in integration tests
# For now, we're testing the pure logic functions that don't need files

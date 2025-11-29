"""
Integration tests for substrate analysis module

Tests with real substrate structures and chemistry.
"""

import pytest
from pathlib import Path

from enzymeforge.substrate import SubstrateAnalyzer, Substrate


# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestSubstrateAnalysisIntegration:
    """Integration tests for substrate analysis with real molecules"""

    def test_ethyl_acetate_loading(self):
        """Test loading ethyl acetate from SMILES"""
        analyzer = SubstrateAnalyzer()

        # Load from SMILES
        substrate = analyzer.load_substrate("CCOC(=O)C", name="ethyl_acetate")

        assert substrate is not None
        assert substrate.name == "ethyl_acetate"
        assert substrate.structure == "CCOC(=O)C"
        assert substrate.format == "smiles"
        assert analyzer.rdkit_mol is not None

    def test_functional_group_identification(self):
        """Test functional group identification"""
        analyzer = SubstrateAnalyzer()

        # Load ester
        analyzer.load_substrate("CCOC(=O)C")

        # Identify functional groups
        groups = analyzer.identify_functional_groups()

        assert len(groups) > 0
        group_types = [g["type"] for g in groups]
        assert "ester" in group_types

    def test_carboxylic_acid_detection(self):
        """Test carboxylic acid detection"""
        analyzer = SubstrateAnalyzer()

        # Acetic acid
        analyzer.load_substrate("CC(=O)O")
        groups = analyzer.identify_functional_groups()

        group_types = [g["type"] for g in groups]
        assert "carboxylic_acid" in group_types

    def test_lactone_detection(self):
        """Test lactone detection"""
        analyzer = SubstrateAnalyzer()

        # Gamma-butyrolactone
        analyzer.load_substrate("C1CCOC(=O)C1")
        groups = analyzer.identify_functional_groups()

        group_types = [g["type"] for g in groups]
        # Lactones are special esters
        assert "lactone" in group_types or "ester" in group_types

    def test_catalytic_mechanism_suggestion(self):
        """Test catalytic mechanism suggestion"""
        analyzer = SubstrateAnalyzer()

        # Test hydrolysis mechanism
        residues = analyzer.suggest_catalytic_residues("hydrolysis")

        assert residues is not None
        assert isinstance(residues, list)
        assert len(residues) > 0
        # Should suggest typical hydrolase residues
        assert any(r in residues for r in ["SER", "THR", "CYS"])

    def test_pdb_file_loading(self):
        """Test loading substrate from PDB file"""
        pdb_file = TEST_DATA_DIR / "ethyl_acetate.pdb"

        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")

        analyzer = SubstrateAnalyzer()

        # Load from PDB
        substrate = analyzer.load_substrate(str(pdb_file), format="pdb")

        assert substrate is not None
        assert substrate.format == "pdb"
        assert substrate.structure == str(pdb_file)

    def test_multiple_substrates(self):
        """Test analyzing multiple substrates"""
        test_cases = [
            ("CCOC(=O)C", "ester"),  # Ethyl acetate
            ("CC(=O)O", "carboxylic_acid"),  # Acetic acid
        ]

        for smiles, expected_group in test_cases:
            analyzer = SubstrateAnalyzer()
            analyzer.load_substrate(smiles)
            groups = analyzer.identify_functional_groups()

            group_types = [g["type"] for g in groups]
            assert expected_group in group_types, \
                f"Expected {expected_group} in {group_types} for {smiles}"


class TestCatalyticMechanisms:
    """Test catalytic mechanism suggestions"""

    def test_hydrolysis_mechanism(self):
        """Test hydrolysis mechanism suggestion"""
        analyzer = SubstrateAnalyzer()
        residues = analyzer.suggest_catalytic_residues("hydrolysis")

        assert residues is not None
        assert isinstance(residues, list)
        assert len(residues) > 0
        # Should suggest serine/threonine for hydrolysis
        assert "SER" in residues or "THR" in residues

    def test_lactonase_mechanism(self):
        """Test lactonase mechanism suggestion"""
        analyzer = SubstrateAnalyzer()
        residues = analyzer.suggest_catalytic_residues("lactonase")

        assert residues is not None
        assert isinstance(residues, list)
        assert len(residues) > 0
        # Lactonase typically has metal-binding histidines
        assert "HIS" in residues

    def test_dehalogenation_mechanism(self):
        """Test dehalogenation mechanism suggestion"""
        analyzer = SubstrateAnalyzer()
        residues = analyzer.suggest_catalytic_residues("dehalogenation")

        assert residues is not None
        assert isinstance(residues, list)
        assert len(residues) > 0


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES"""
        analyzer = SubstrateAnalyzer()

        with pytest.raises((ValueError, Exception)):
            analyzer.load_substrate("INVALID_SMILES")

    def test_unknown_mechanism(self):
        """Test unknown mechanism handling"""
        analyzer = SubstrateAnalyzer()
        residues = analyzer.suggest_catalytic_residues("unknown_mechanism")

        # Should return default serine protease triad
        assert residues is not None
        assert isinstance(residues, list)
        # Default should be serine protease triad
        assert "SER" in residues and "HIS" in residues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

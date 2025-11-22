"""
Tests for substrate analysis module
"""

import pytest
from enzymeforge.substrate import SubstrateAnalyzer, Substrate, CatalyticSite


class TestSubstrateAnalyzer:
    """Test SubstrateAnalyzer functionality"""

    def test_load_smiles_substrate(self):
        """Test loading substrate from SMILES"""
        analyzer = SubstrateAnalyzer()

        # PFOA SMILES (perfluorooctanoic acid)
        pfoa_smiles = "C(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"

        substrate = analyzer.load_substrate(pfoa_smiles, format="smiles", name="PFOA")

        assert substrate.name == "PFOA"
        assert substrate.format == "smiles"
        assert substrate.structure == pfoa_smiles
        assert analyzer.rdkit_mol is not None

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES"""
        analyzer = SubstrateAnalyzer()

        with pytest.raises(ValueError, match="Invalid SMILES"):
            analyzer.load_substrate("INVALID_SMILES", format="smiles")

    def test_identify_functional_groups_carboxylic_acid(self):
        """Test functional group identification for carboxylic acid"""
        analyzer = SubstrateAnalyzer()

        # Simple carboxylic acid: acetic acid
        acetic_acid = "CC(=O)O"
        substrate = analyzer.load_substrate(acetic_acid, format="smiles")

        groups = analyzer.identify_functional_groups()

        assert len(groups) > 0
        # Should detect carboxylic acid
        assert any(g["type"] == "carboxylic_acid" for g in groups)

    def test_identify_functional_groups_ester(self):
        """Test functional group identification for ester"""
        analyzer = SubstrateAnalyzer()

        # Ethyl acetate (ester)
        ethyl_acetate = "CC(=O)OCC"
        substrate = analyzer.load_substrate(ethyl_acetate, format="smiles")

        groups = analyzer.identify_functional_groups()

        assert len(groups) > 0
        assert any(g["type"] == "ester" for g in groups)
        # Check that suggested mechanism is hydrolysis
        ester_group = [g for g in groups if g["type"] == "ester"][0]
        assert ester_group["suggested_mechanism"] == "hydrolysis"

    def test_identify_functional_groups_pfas(self):
        """Test functional group identification for PFAS"""
        analyzer = SubstrateAnalyzer()

        # PFOA
        pfoa_smiles = "C(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
        substrate = analyzer.load_substrate(pfoa_smiles, format="smiles")

        groups = analyzer.identify_functional_groups()

        # Should detect both carboxylic acid and halogens
        group_types = [g["type"] for g in groups]
        assert "carboxylic_acid" in group_types
        assert "halogen" in group_types

    def test_suggest_catalytic_residues_hydrolysis(self):
        """Test catalytic residue suggestion for hydrolysis"""
        analyzer = SubstrateAnalyzer()

        residues = analyzer.suggest_catalytic_residues("hydrolysis")

        assert residues == ["SER", "HIS", "ASP"]

    def test_suggest_catalytic_residues_lactonase(self):
        """Test catalytic residue suggestion for lactonase (quorum quenching)"""
        analyzer = SubstrateAnalyzer()

        residues = analyzer.suggest_catalytic_residues("lactonase")

        assert residues == ["HIS", "HIS", "ASP"]

    def test_suggest_catalytic_residues_dehalogenation(self):
        """Test catalytic residue suggestion for dehalogenation"""
        analyzer = SubstrateAnalyzer()

        residues = analyzer.suggest_catalytic_residues("dehalogenation")

        assert residues == ["ASP", "ARG", "TRP"]

    def test_suggest_catalytic_residues_unknown_mechanism(self):
        """Test handling of unknown mechanism"""
        analyzer = SubstrateAnalyzer()

        # Should default to serine protease triad
        residues = analyzer.suggest_catalytic_residues("unknown_mechanism")

        assert residues == ["SER", "HIS", "ASP"]


class TestSubstrateDataclass:
    """Test Substrate dataclass"""

    def test_substrate_creation(self):
        """Test creating Substrate object"""
        substrate = Substrate(
            name="test_substrate",
            structure="CCO",
            format="smiles"
        )

        assert substrate.name == "test_substrate"
        assert substrate.structure == "CCO"
        assert substrate.format == "smiles"
        assert substrate.reactive_groups == []
        assert substrate.binding_residues is None


class TestCatalyticSite:
    """Test CatalyticSite dataclass"""

    def test_catalytic_site_creation(self):
        """Test creating CatalyticSite object"""
        site = CatalyticSite(
            mechanism="hydrolysis",
            catalytic_residues=["SER", "HIS", "ASP"],
            geometry_constraints={"distance": 4.0}
        )

        assert site.mechanism == "hydrolysis"
        assert len(site.catalytic_residues) == 3
        assert site.geometry_constraints["distance"] == 4.0

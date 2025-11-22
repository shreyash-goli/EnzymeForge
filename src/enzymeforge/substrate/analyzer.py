"""
Substrate analysis module for EnzymeForge

Analyzes substrate molecules and suggests catalytic strategies
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class Substrate:
    """Represents target substrate molecule"""
    name: str
    structure: str  # PDB file path or SMILES string
    format: str  # "pdb", "smiles", "mol2"
    reactive_groups: List[str] = field(default_factory=list)
    binding_residues: Optional[List[str]] = None

    def to_pdb(self) -> str:
        """Convert substrate to PDB format

        Returns:
            Path to PDB file
        """
        if self.format == "pdb":
            return self.structure
        # Will implement conversion in next commit
        raise NotImplementedError("SMILES to PDB conversion not yet implemented")

    def identify_reactive_sites(self) -> List[Dict]:
        """Identify potential catalytic sites

        Returns:
            List of reactive site dictionaries
        """
        # Will implement in functional group identification commit
        return []


@dataclass
class CatalyticSite:
    """Represents desired active site"""
    mechanism: str  # "hydrolysis", "oxidation", etc.
    catalytic_residues: List[str]  # ["SER", "HIS", "ASP"]
    geometry_constraints: Dict = field(default_factory=dict)  # distance/angle constraints


@dataclass
class DesignConfig:
    """Complete design specification"""
    substrate: Substrate
    active_site: CatalyticSite
    scaffold_size: tuple  # (min, max) residues
    num_designs: int
    optimization_params: Dict = field(default_factory=dict)


class SubstrateAnalyzer:
    """Analyze substrate and suggest catalytic strategies"""

    def __init__(self):
        self.rdkit_mol = None
        self.pdb_structure = None

    def load_substrate(
        self,
        structure: str,
        format: str = "smiles",
        name: str = "substrate"
    ) -> Substrate:
        """Load substrate from file or string

        Args:
            structure: SMILES string or file path
            format: "smiles", "pdb", "mol2"
            name: Name for the substrate

        Returns:
            Substrate object
        """
        if format == "smiles":
            return self._smiles_to_substrate(structure, name)
        elif format == "pdb":
            return self._pdb_to_substrate(structure, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _smiles_to_substrate(self, smiles: str, name: str) -> Substrate:
        """Convert SMILES to Substrate object

        Args:
            smiles: SMILES string
            name: Substrate name

        Returns:
            Substrate object
        """
        try:
            from rdkit import Chem
            self.rdkit_mol = Chem.MolFromSmiles(smiles)

            if self.rdkit_mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")

            return Substrate(
                name=name,
                structure=smiles,
                format="smiles",
                reactive_groups=[]
            )
        except ImportError:
            raise ImportError("RDKit is required for SMILES parsing. Install with: pip install rdkit")

    def _pdb_to_substrate(self, pdb_path: str, name: str) -> Substrate:
        """Convert PDB file to Substrate object

        Args:
            pdb_path: Path to PDB file
            name: Substrate name

        Returns:
            Substrate object
        """
        try:
            from Bio.PDB import PDBParser

            if not Path(pdb_path).exists():
                raise FileNotFoundError(f"PDB file not found: {pdb_path}")

            parser = PDBParser(QUIET=True)
            self.pdb_structure = parser.get_structure(name, pdb_path)

            return Substrate(
                name=name,
                structure=pdb_path,
                format="pdb",
                reactive_groups=[]
            )
        except ImportError:
            raise ImportError("BioPython is required for PDB parsing. Install with: pip install biopython")

    def identify_functional_groups(self) -> List[Dict]:
        """Identify reactive functional groups

        Returns:
            List of functional groups with properties
        """
        if self.rdkit_mol is None:
            return []

        try:
            from rdkit.Chem import Fragments
        except ImportError:
            raise ImportError("RDKit is required for functional group identification")

        groups = []

        # Check for common reactive groups
        # Esters
        if Fragments.fr_ester(self.rdkit_mol) > 0:
            groups.append({
                "type": "ester",
                "count": Fragments.fr_ester(self.rdkit_mol),
                "suggested_mechanism": "hydrolysis"
            })

        # Carboxylic acids
        if Fragments.fr_COO(self.rdkit_mol) > 0:
            groups.append({
                "type": "carboxylic_acid",
                "count": Fragments.fr_COO(self.rdkit_mol),
                "suggested_mechanism": "hydrolysis"
            })

        # Amides
        if Fragments.fr_amide(self.rdkit_mol) > 0:
            groups.append({
                "type": "amide",
                "count": Fragments.fr_amide(self.rdkit_mol),
                "suggested_mechanism": "hydrolysis"
            })

        # Lactones (for quorum sensing molecules)
        if Fragments.fr_lactone(self.rdkit_mol) > 0:
            groups.append({
                "type": "lactone",
                "count": Fragments.fr_lactone(self.rdkit_mol),
                "suggested_mechanism": "lactonase"
            })

        # Alcohols
        if Fragments.fr_Al_OH(self.rdkit_mol) > 0:
            groups.append({
                "type": "alcohol",
                "count": Fragments.fr_Al_OH(self.rdkit_mol),
                "suggested_mechanism": "oxidation"
            })

        # Halogens (for PFAS and other halogenated compounds)
        if Fragments.fr_halogen(self.rdkit_mol) > 0:
            groups.append({
                "type": "halogen",
                "count": Fragments.fr_halogen(self.rdkit_mol),
                "suggested_mechanism": "dehalogenation"
            })

        # Nitro groups
        if Fragments.fr_nitro(self.rdkit_mol) > 0:
            groups.append({
                "type": "nitro",
                "count": Fragments.fr_nitro(self.rdkit_mol),
                "suggested_mechanism": "reduction"
            })

        return groups

    def suggest_catalytic_residues(self, mechanism: str) -> List[str]:
        """Suggest catalytic residues for mechanism

        Args:
            mechanism: "hydrolysis", "oxidation", etc.

        Returns:
            List of suggested residue types
        """
        catalytic_triads = {
            "hydrolysis": ["SER", "HIS", "ASP"],  # Serine protease
            "oxidation": ["CYS", "HIS"],  # Peroxidase
            "reduction": ["CYS", "CYS"],  # Disulfide reductase
            "lactonase": ["HIS", "HIS", "ASP"],  # Metallo-beta-lactamase fold (quorum quenching)
            "dehalogenation": ["ASP", "ARG", "TRP"],  # Haloalkane dehalogenase
        }

        residues = catalytic_triads.get(mechanism, [])

        if not residues:
            # Default to serine protease triad if mechanism not recognized
            print(f"Warning: Unknown mechanism '{mechanism}', using default serine protease triad")
            residues = ["SER", "HIS", "ASP"]

        return residues

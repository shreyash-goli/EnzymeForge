#!/usr/bin/env python
"""
Generate test substrate structures for integration testing

Creates 3D structures from SMILES for various substrates.
Requires RDKit.
"""

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

# Test substrates
SUBSTRATES = {
    "ethyl_acetate": {
        "smiles": "CCOC(=O)C",
        "name": "ETA",
        "description": "Simple ester for basic testing"
    },
    "benzoic_acid": {
        "smiles": "C1=CC=C(C=C1)C(=O)O",
        "name": "BZA",
        "description": "Aromatic carboxylic acid"
    },
    "lactone": {
        "smiles": "C1CCOC(=O)C1",
        "name": "GBL",
        "description": "Gamma-butyrolactone - simple lactone"
    },
    "ahl_c4": {
        "smiles": "CCCC(=O)N[C@@H]1CCOC1=O",
        "name": "C4HSL",
        "description": "N-butanoyl-L-homoserine lactone (short AHL)"
    },
}


def generate_substrate_pdb(smiles: str, output_path: Path, name: str = "LIG"):
    """Generate 3D structure from SMILES and save as PDB

    Args:
        smiles: SMILES string
        output_path: Path to save PDB file
        name: Residue name (3-letter code)
    """
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # Write PDB file
    writer = Chem.PDBWriter(str(output_path))
    writer.write(mol)
    writer.close()

    # Post-process: Fix residue name
    with open(output_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        for line in lines:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                # Replace UNL with custom name
                line = line.replace('UNL', name)
            f.write(line)

    print(f"  Generated {output_path.name}")


def main():
    """Generate all test substrates"""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    print("Generating test substrate structures...")

    for substrate_id, info in SUBSTRATES.items():
        output_path = output_dir / f"{substrate_id}.pdb"

        try:
            generate_substrate_pdb(
                smiles=info["smiles"],
                output_path=output_path,
                name=info["name"]
            )
        except Exception as e:
            print(f"  ERROR: Failed to generate {substrate_id}: {e}")

    print("\nDone! Generated substrate files:")
    for pdb_file in sorted(output_dir.glob("*.pdb")):
        print(f"  - {pdb_file.name}")


if __name__ == "__main__":
    main()

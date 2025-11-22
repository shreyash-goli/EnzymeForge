"""
Constraint generator for RFdiffusion

Generates geometric constraints for enzyme design
"""

from typing import List
from pathlib import Path

from enzymeforge.substrate.analyzer import Substrate, CatalyticSite


class ConstraintGenerator:
    """Generate geometric constraints for RFdiffusion"""

    def __init__(self, substrate: Substrate, active_site: CatalyticSite):
        """Initialize constraint generator

        Args:
            substrate: Target substrate molecule
            active_site: Desired catalytic site
        """
        self.substrate = substrate
        self.active_site = active_site

    def generate_contig_string(
        self,
        scaffold_size: tuple,
        motif_positions: List[int]
    ) -> str:
        """Generate RFdiffusion contig string

        Args:
            scaffold_size: (min, max) total residues
            motif_positions: Which positions are fixed catalytic residues

        Returns:
            Contig string like "50-50/A130-136/20-20"

        Example:
            >>> gen = ConstraintGenerator(substrate, active_site)
            >>> gen.generate_contig_string((100, 150), [45, 46, 47])
            "48-48/A45-47/49-49"
        """
        min_size, max_size = scaffold_size

        if not motif_positions:
            # No motif, simple de novo design
            return f"{min_size}-{max_size}"

        # Calculate de novo regions around motif
        motif_start = min(motif_positions)
        motif_end = max(motif_positions)
        motif_length = len(motif_positions)

        # Place motif in middle of protein
        n_term_length = (min_size - motif_length) // 2
        c_term_length = min_size - n_term_length - motif_length

        # Build contig string
        contig = f"{n_term_length}-{n_term_length}/"
        contig += f"A{motif_start}-{motif_end}/"
        contig += f"{c_term_length}-{c_term_length}"

        return contig

    def generate_cst_file(
        self,
        output_path: str,
        distance_cutoff: float = 4.0
    ) -> str:
        """Generate Rosetta constraint file

        Args:
            output_path: Where to save .cst file
            distance_cutoff: Max distance for catalytic interactions (Å)

        Returns:
            Path to generated file

        Example constraint file format:
            AtomPair CA 1A CA 1B HARMONIC 4.0 1.0
            AtomPair CA 2A CA 1B HARMONIC 4.0 1.0
        """
        constraints = []

        # Generate distance constraints between catalytic residues and substrate
        for i, res in enumerate(self.active_site.catalytic_residues):
            # Constraint between catalytic residue and substrate
            # Chain A = protein, Chain B = substrate
            constraint = (
                f"AtomPair CA {i+1}A CA 1B "
                f"HARMONIC {distance_cutoff} 1.0"
            )
            constraints.append(constraint)

        # Add inter-residue constraints for catalytic residues
        n_catalytic = len(self.active_site.catalytic_residues)
        if n_catalytic >= 2:
            for i in range(n_catalytic - 1):
                for j in range(i + 1, n_catalytic):
                    # Constraint between catalytic residues
                    constraint = (
                        f"AtomPair CA {i+1}A CA {j+1}A "
                        f"HARMONIC 6.0 1.0"  # Allow ~6Å between catalytic residues
                    )
                    constraints.append(constraint)

        # Write to file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            f.write('\n'.join(constraints))

        return str(output)

    def generate_guide_potential(self) -> str:
        """Generate custom guide potential string for RFdiffusion

        Returns:
            Guide potential specification

        Example:
            "type:substrate_contacts,weight:1.0,substrate_chain:B"
        """
        # Example: substrate binding pocket potential
        potential = (
            f"type:substrate_contacts,"
            f"weight:1.0,"
            f"substrate_chain:B"
        )

        return potential

    def generate_motif_definition(self, motif_positions: List[int]) -> dict:
        """Generate motif definition for RFdiffusion

        Args:
            motif_positions: Residue positions for motif

        Returns:
            Dictionary with motif information
        """
        return {
            "positions": motif_positions,
            "residue_types": self.active_site.catalytic_residues,
            "chain": "A",
        }

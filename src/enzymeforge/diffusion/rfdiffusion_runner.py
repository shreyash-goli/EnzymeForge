"""
RFdiffusion runner for EnzymeForge

Handles RFdiffusion execution via Google Colab
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffusionResult:
    """Output from RFdiffusion"""
    design_id: str
    pdb_path: str
    contig_string: str
    sequence: str


@dataclass
class DiffusionConfig:
    """RFdiffusion parameters"""
    contigs: str
    pdb: Optional[str]
    iterations: int
    num_designs: int
    guide_potentials: Optional[str] = None
    guide_scale: float = 1.0

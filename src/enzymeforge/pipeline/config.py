"""
Pipeline configuration for EnzymeForge

Unified configuration for complete enzyme design pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List

from enzymeforge.diffusion.rfdiffusion_runner import DiffusionConfig
from enzymeforge.sequence.ligandmpnn_runner import SequenceDesignConfig
from enzymeforge.validation.colabfold_runner import FoldingConfig


@dataclass
class SubstrateConfig:
    """Substrate specification

    Attributes:
        name: Substrate name (e.g., "PFOA", "AHL")
        structure: PDB file path or SMILES string
        format: "pdb", "smiles", or "mol2"
        ligand_name: Three-letter ligand code in PDB (e.g., "FOA")
    """
    name: str
    structure: str
    format: str = "smiles"
    ligand_name: Optional[str] = None


@dataclass
class ActiveSiteConfig:
    """Active site specification

    Attributes:
        mechanism: Catalytic mechanism (e.g., "hydrolysis", "oxidation")
        catalytic_residues: Residue types for active site (e.g., ["SER", "HIS", "ASP"])
        reference_pdb: Optional PDB with catalytic motif
        motif_positions: Residue positions in reference PDB (e.g., [45, 46, 47])
        ref_chain: Chain ID in reference PDB (default: "A")
    """
    mechanism: str
    catalytic_residues: List[str]
    reference_pdb: Optional[Path] = None
    motif_positions: Optional[List[int]] = None
    ref_chain: str = "A"


@dataclass
class ScaffoldConfig:
    """Scaffold design parameters

    Attributes:
        min_size: Minimum protein length (residues)
        max_size: Maximum protein length (residues)
        num_designs: Number of backbone designs to generate
    """
    min_size: int = 100
    max_size: int = 150
    num_designs: int = 10


@dataclass
class FilteringConfig:
    """Quality filtering parameters

    Attributes:
        rmsd_cutoff: Maximum RMSD for motif (Å)
        clash_cutoff: Maximum clash score
        plddt_cutoff: Minimum pLDDT score
        motif_plddt_cutoff: Minimum motif pLDDT score
        filter_after_diffusion: Filter after RFdiffusion (default: True)
        filter_after_sequence: Filter after sequence design (default: True)
    """
    rmsd_cutoff: float = 2.0
    clash_cutoff: float = 2.0
    plddt_cutoff: float = 70.0
    motif_plddt_cutoff: float = 80.0
    filter_after_diffusion: bool = True
    filter_after_sequence: bool = True


@dataclass
class ComputeConfig:
    """Computational resource configuration

    Attributes:
        rfdiffusion_path: Path to RFdiffusion installation
        rfdiffusion_aa_path: Path to RFdiffusion all-atom (optional)
        ligandmpnn_path: Path to LigandMPNN installation
        colabfold_command: ColabFold command (default: "colabfold_batch")
        use_rfdiffusion_aa: Use all-atom RFdiffusion (default: False)
        num_workers: Parallel workers for processing (default: 8)
    """
    rfdiffusion_path: Path
    ligandmpnn_path: Path
    rfdiffusion_aa_path: Optional[Path] = None
    colabfold_command: str = "colabfold_batch"
    use_rfdiffusion_aa: bool = False
    num_workers: int = 8


@dataclass
class PipelineConfig:
    """Complete pipeline configuration

    Combines all configuration components for full enzyme design pipeline.

    Attributes:
        name: Experiment name
        output_dir: Base output directory
        substrate: Substrate configuration
        active_site: Active site configuration
        scaffold: Scaffold design parameters
        diffusion: RFdiffusion parameters
        sequence: LigandMPNN parameters
        folding: ColabFold parameters
        filtering: Quality filtering parameters
        compute: Computational resources
        metadata: Additional metadata

    Example:
        >>> config = PipelineConfig(
        ...     name="pfas_degrader",
        ...     output_dir=Path("output/pfas_exp1"),
        ...     substrate=SubstrateConfig(
        ...         name="PFOA",
        ...         structure="C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(=O)O)F",
        ...         format="smiles",
        ...         ligand_name="FOA"
        ...     ),
        ...     active_site=ActiveSiteConfig(
        ...         mechanism="hydrolysis",
        ...         catalytic_residues=["SER", "HIS", "ASP"],
        ...         reference_pdb=Path("refs/serine_protease.pdb"),
        ...         motif_positions=[45, 46, 47]
        ...     ),
        ...     scaffold=ScaffoldConfig(
        ...         min_size=100,
        ...         max_size=150,
        ...         num_designs=20
        ...     ),
        ...     diffusion=DiffusionConfig(
        ...         contigs="50-50/A45-47/50-50",
        ...         iterations=100
        ...     ),
        ...     sequence=SequenceDesignConfig(
        ...         num_seqs=4,
        ...         temperature=0.1
        ...     ),
        ...     folding=FoldingConfig(
        ...         num_models=2,
        ...         num_recycles=3
        ...     ),
        ...     filtering=FilteringConfig(
        ...         rmsd_cutoff=2.0,
        ...         plddt_cutoff=70.0
        ...     ),
        ...     compute=ComputeConfig(
        ...         rfdiffusion_path=Path("/path/to/RFdiffusion"),
        ...         ligandmpnn_path=Path("/path/to/LigandMPNN")
        ...     )
        ... )
    """
    name: str
    output_dir: Path
    substrate: SubstrateConfig
    active_site: ActiveSiteConfig
    scaffold: ScaffoldConfig
    diffusion: DiffusionConfig
    sequence: SequenceDesignConfig
    folding: FoldingConfig
    filtering: FilteringConfig
    compute: ComputeConfig
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration"""
        # Ensure output directory is Path
        self.output_dir = Path(self.output_dir)

        # Validate that if reference_pdb is provided, motif_positions are too
        if self.active_site.reference_pdb and not self.active_site.motif_positions:
            raise ValueError("motif_positions required when reference_pdb is provided")

        # Validate scaffold size
        if self.scaffold.min_size > self.scaffold.max_size:
            raise ValueError(f"min_size ({self.scaffold.min_size}) must be <= max_size ({self.scaffold.max_size})")

        # Generate contig string for diffusion if not set
        if not self.diffusion.contigs:
            self.diffusion.contigs = self._generate_contig_string()

        # Set reference PDB in diffusion config if not already set
        if self.active_site.reference_pdb and not self.diffusion.pdb:
            self.diffusion.pdb = self.active_site.reference_pdb

        # Set substrate in diffusion config
        if self.substrate.ligand_name and not self.diffusion.substrate:
            self.diffusion.substrate = self.substrate.ligand_name

    def _generate_contig_string(self) -> str:
        """Generate RFdiffusion contig string from config

        Returns:
            Contig string like "50-50/A45-47/50-50"
        """
        if not self.active_site.motif_positions:
            # No motif, simple de novo design
            return f"{self.scaffold.min_size}-{self.scaffold.max_size}"

        # Calculate de novo regions around motif
        motif_start = min(self.active_site.motif_positions)
        motif_end = max(self.active_site.motif_positions)
        motif_length = len(self.active_site.motif_positions)

        # Place motif in middle of protein
        n_term_length = (self.scaffold.min_size - motif_length) // 2
        c_term_length = self.scaffold.min_size - n_term_length - motif_length

        # Build contig string
        chain = self.active_site.ref_chain
        contig = f"{n_term_length}-{n_term_length}/"
        contig += f"{chain}{motif_start}-{motif_end}/"
        contig += f"{c_term_length}-{c_term_length}"

        return contig

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PipelineConfig instance

        Note:
            YAML format should match the dataclass structure.
        """
        import yaml

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Helper to convert string paths to Path objects
        def convert_to_path(d, path_keys):
            for key in path_keys:
                if key in d and d[key] is not None:
                    d[key] = Path(d[key])
            return d

        # Convert nested dicts to dataclasses
        substrate = SubstrateConfig(**config_dict.get("substrate", {}))

        active_site_dict = config_dict.get("active_site", {})
        convert_to_path(active_site_dict, ["reference_pdb"])
        active_site = ActiveSiteConfig(**active_site_dict)

        scaffold = ScaffoldConfig(**config_dict.get("scaffold", {}))

        diffusion_dict = config_dict.get("diffusion", {})
        convert_to_path(diffusion_dict, ["pdb", "ckpt_override_path"])
        diffusion = DiffusionConfig(**diffusion_dict)

        sequence = SequenceDesignConfig(**config_dict.get("sequence", {}))
        folding = FoldingConfig(**config_dict.get("folding", {}))
        filtering = FilteringConfig(**config_dict.get("filtering", {}))

        compute_dict = config_dict.get("compute", {})
        convert_to_path(compute_dict, ["rfdiffusion_path", "ligandmpnn_path", "rfdiffusion_aa_path"])
        compute = ComputeConfig(**compute_dict)

        return cls(
            name=config_dict["name"],
            output_dir=Path(config_dict["output_dir"]),
            substrate=substrate,
            active_site=active_site,
            scaffold=scaffold,
            diffusion=diffusion,
            sequence=sequence,
            folding=folding,
            filtering=filtering,
            compute=compute,
            metadata=config_dict.get("metadata", {})
        )

    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file

        Args:
            output_path: Where to save YAML file
        """
        import yaml
        from dataclasses import asdict

        config_dict = asdict(self)

        # Convert Path objects to strings for YAML serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj

        config_dict = convert_paths(config_dict)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

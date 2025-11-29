# EnzymeForge

AI-powered platform for designing novel enzymes that degrade environmental toxins through guided de novo protein design. Integrates RFdiffusion, LigandMPNN, and AlphaFold2 with chemistry-aware substrate analysis to create custom enzymes for PFAS degradation, microplastic breakdown, and bacterial biofilm disruption.

## What I Built

### 1. Substrate Analysis & Constraint Generation
**Chemistry-Aware Active Site Design** ([substrate/](src/enzymeforge/substrate/))

- **SubstrateAnalyzer**: RDKit-based molecular structure analysis
  - Functional group detection (esters, amides, lactones, halides)
  - Catalytic mechanism matching (hydrolysis, oxidation, dehalogenation)
  - Residue suggestion based on reaction type (Ser/His/Asp triads for hydrolases)

- **ConstraintGenerator**: Converts chemistry requirements → RFdiffusion constraints
  - **Contig strings**: Define protein architecture (e.g., `"A10-40/0 A45-45/0 A50-100"`)
  - **CST files**: Spatial constraints for active site geometry
  - **Hotspot residues**: Fixed catalytic residues during design

**Key Learning**: Bridging chemistry and structure - translating substrate reactivity into geometric constraints that guide protein folding.

### 2. Backbone Generation with RFdiffusion
**Diffusion Model Integration** ([diffusion/](src/enzymeforge/diffusion/))

- **RFdiffusionRunner**: Subprocess orchestration for backbone generation
  - Handles both protein-only and all-atom (with ligand) diffusion
  - Enzyme-specific checkpoints: `ActiveSite` mode for catalytic proteins
  - Guide potentials: Custom energy terms for substrate contacts, geometry

- **DiffusionConfig**: Pydantic-based configuration
  - Type-safe parameter validation
  - Optional features: partial diffusion, symmetry, secondary structure control

**Guide Potentials Implemented**:
```python
- substrate_contacts: Encourage ligand-protein interactions (weight: 5.0)
- catalytic_geometry: Enforce active site distances (weight: 10.0, target: 3.5Å)
- olig_contacts: Multi-chain interfaces (for dimers/trimers)
```

**Key Learning**: Diffusion models aren't just image generators - RFdiffusion operates in 3D rotation/translation space, conditioning on partial structures (motifs) to generate complete protein backbones.

### 3. Sequence Design with LigandMPNN
**Message Passing Neural Network for Protein Design** ([sequence/](src/enzymeforge/sequence/))

- **LigandMPNNRunner**: Sequence optimization with ligand awareness
  - Fixed vs. redesignable regions based on contig parsing
  - Temperature sampling for sequence diversity
  - Ligand-aware scoring: considers substrate when selecting amino acids

- **FastRelax Integration**: Iterative design-relax cycles
  - **Cycle 1**: LigandMPNN designs sequences → Rosetta FastRelax optimizes
  - **Cycle 2**: Relaxed structures → LigandMPNN redesigns → final relax
  - Improves geometry and removes clashes

**Key Learning**: Graph neural networks can learn protein sequence-structure relationships from massive datasets, enabling de novo design without evolutionary templates.

### 4. Structure Prediction & Validation
**AlphaFold2 via ColabFold** ([validation/](src/enzymeforge/validation/))

- **ColabFoldRunner**: Validates designed sequences fold correctly
  - Batch MSA generation (MMseqs2)
  - AlphaFold2 inference with custom recycling
  - pLDDT and pTM metrics for quality filtering

- **StructureValidator**: Quality control pipeline
  - pLDDT cutoffs (typically >75 for high confidence)
  - Active site preservation checks
  - Clash detection post-prediction

**Key Learning**: AlphaFold2 is a folding oracle - if a designed sequence doesn't fold to the intended structure, it's likely non-viable in vitro.

### 5. HPC Cluster Orchestration
**SLURM Job Management** ([cluster/](src/enzymeforge/cluster/))

- **SlurmRunner**: Multi-stage pipeline automation
  - Dependency graphs: folding waits for sequence design, which waits for diffusion
  - Resource allocation: GPU partitions, memory limits, time constraints
  - Batch processing: 100s of designs in parallel

- **Job Templates**: Jinja2-based SLURM script generation
  - Stage-specific resource profiles (diffusion needs GPUs, MSA is CPU-intensive)
  - Automatic output directory management
  - Email notifications and checkpointing

**Key Learning**: Scientific computing at scale requires orchestration layers - raw tools are powerful but need workflow management for production use.

### 6. End-to-End Pipeline
**Configuration-Driven Orchestration** ([pipeline/](src/enzymeforge/pipeline/))

- **Orchestrator**: Coordinates all stages with error handling
  ```
  Substrate → Constraints → RFdiffusion → LigandMPNN → FastRelax → ColabFold → Report
  ```
- **YAMLConfig**: Human-readable experiment definition
- **ResultsTracking**: Structured output with versioning

## How Components Interact

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  1. Substrate Analysis (RDKit + Chemistry Rules)            │
│     Input: SMILES string (e.g., PFOA perfluorinated acid)   │
│     Output: Functional groups, suggested residues           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Constraint Generation                                    │
│     Input: Substrate + mechanism + motif residues           │
│     Output: contig.txt, constraints.cst                     │
│     - Contig: "A10-40/0 A45-45/0 A50-100" (fixed A45)      │
│     - CST: Geometric constraints (distances, angles)        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  3. RFdiffusion (Backbone Generation)                       │
│     Input: contig + cst + guide potentials                  │
│     Process: Denoise latent → 3D backbone coordinates       │
│     Output: 100 PDB backbones with active site motif        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  4a. LigandMPNN (Sequence Design)                           │
│      Input: Backbone PDB + ligand params + contig           │
│      Process: GNN predicts amino acids for each position    │
│      Output: 10 sequences per backbone (1000 total)         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (optional)
┌─────────────────────────────────────────────────────────────┐
│  4b. Rosetta FastRelax (Iterative Refinement)               │
│      Cycle 1: LigandMPNN → relax backbones                  │
│      Cycle 2: Redesign on relaxed → final relax             │
│      Output: Clash-free, energy-minimized designs           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  5. ColabFold (Structure Prediction)                        │
│     Input: FASTA sequences (1000 designs)                   │
│     Process: MSA → AlphaFold2 → pLDDT/pTM scoring          │
│     Output: Predicted structures + confidence metrics       │
│     Filter: Keep designs with pLDDT > 75, active site OK   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Results & Reporting                                      │
│     - Top designs ranked by pLDDDT + active site geometry   │
│     - PDB files + FASTA sequences ready for synthesis       │
│     - Metrics: success rate, diversity, computational cost  │
└─────────────────────────────────────────────────────────────┘
```

### Key Integration Points

1. **Contig Parsing**: Shared between RFdiffusion (scaffold generation) and LigandMPNN (fixed residue regions)
   - Format: `"chain_id start-end / chain_break"`
   - Enables communication about which residues are catalytic (fixed) vs designable

2. **Params Files**: Ligand chemical topology for Rosetta/LigandMPNN
   - Generated from SMILES using RDKit → Rosetta's `molfile_to_params.py`
   - Defines atom types, bonds, charges for non-standard molecules

3. **PDB Threading**: Each stage passes structures to the next
   - RFdiffusion outputs backbone-only (Cα atoms) → LigandMPNN threads sequences
   - LigandMPNN outputs full-atom models → FastRelax optimizes
   - Final designs → ColabFold validates

4. **Configuration Inheritance**: YAML config propagates through pipeline
   - Single source of truth for experiment parameters
   - Pydantic validation catches errors early

## What I Learned

### Protein Design Fundamentals

1. **De Novo vs. Directed Evolution**:
   - Traditional: Start with natural enzyme, mutate, screen
   - De novo: Generate completely new protein from scratch (no homology required)
   - EnzymeForge does de novo - can design enzymes for substrates with no natural degraders

2. **Inverse Folding**:
   - Normal problem: sequence → structure (AlphaFold2)
   - Inverse: structure → sequence (LigandMPNN, ProteinMPNN)
   - Key insight: Many sequences fold to same structure (sequence space >> structure space)

3. **Diffusion Models in 3D**:
   - RFdiffusion learns p(structure) from PDB database
   - Forward: Add noise to coordinates → destroy structure
   - Reverse: Denoise → generate valid protein
   - Conditioning: Fix active site motif, generate surrounding scaffold

4. **Catalytic Constraints**:
   - Active sites require precise geometry (3-4Å distances for H-bonding/nucleophilic attack)
   - Constraints are "soft" guides, not hard rules (diffusion can violate if energetically favorable)
   - Oxyanion holes, Ser-His-Asp triads: recurring motifs in hydrolase chemistry

### Machine Learning for Biology

1. **Graph Neural Networks (GNNs)**:
   - Proteins are graphs: nodes (residues), edges (proximity/bonds)
   - Message passing: Each node aggregates info from neighbors
   - LigandMPNN: 3 encoder layers + 3 decoder layers, learns from 300k+ protein structures

2. **Attention Mechanisms in AlphaFold2**:
   - Evoformer: Pair representation (residue-residue relationships)
   - MSA co-evolution signals: Coupled mutations reveal structural contacts
   - Recycling: Iteratively refine structure prediction (like gradient descent for structures)

3. **Scoring Functions**:
   - pLDDT: Per-residue confidence (0-100), >90 is excellent
   - pTM: Template modeling score, measures global fold correctness
   - Energy-based: Rosetta scoring (van der Waals, electrostatics, solvation)

### Software Engineering for Research

1. **Configuration as Code**:
   - YAML configs make experiments reproducible
   - Pydantic validation prevents invalid parameter combinations
   - Version control for configs → full experimental provenance

2. **Container Orchestration**:
   - Docker: Development and local testing
   - Singularity: HPC clusters (no root required, GPU passthrough)
   - Multi-stage builds: Minimize image size (RFdiffusion ~8GB, LigandMPNN ~2GB)

3. **Testing Scientific Software**:
   - Unit tests: Mock subprocess calls, test logic in isolation
   - Integration tests: Real data, small-scale runs (10 designs, not 100)
   - Fixtures: Pre-computed outputs for validation

4. **Subprocess Management**:
   - Python `subprocess.run()` for external tools (RFdiffusion is PyTorch, not a library)
   - Error handling: Capture stderr, parse for specific failure modes
   - Timeouts: Prevent hanging jobs on clusters

### Computational Chemistry

1. **SMILES Notation**: 1D string representation of molecules
   - `CCOC(=O)C` = ethyl acetate (CH3-CO-O-CH2-CH3)
   - RDKit parses to 2D/3D structures

2. **Force Fields**: Rosetta uses knowledge-based potentials
   - Trained on experimental structures (X-ray, NMR)
   - Balances physics (electrostatics) with statistics (observed frequencies)

3. **Coordinate Systems**:
   - Cartesian (x, y, z): Standard PDB format
   - Internal (φ, ψ, χ angles): Backbone dihedrals
   - RFdiffusion operates on frames (rotation matrices + translations per residue)

### Domain-Specific Challenges

1. **PFAS "Forever Chemicals"**:
   - Perfluorinated compounds extremely stable (C-F bond is strongest in organic chemistry)
   - No known natural enzymes degrade PFAS effectively
   - Design challenge: Create defluorinase from scratch

2. **Quorum Sensing**:
   - Bacteria use AHL (acyl-homoserine lactone) signaling for biofilm formation
   - Lactonases hydrolyze lactone ring → disrupt communication
   - Specificity challenge: Target AHL without affecting host molecules

3. **Microplastics**:
   - PETase evolved recently for polyethylene terephthalate
   - Substrate size challenge: Polymer chains are huge compared to typical small molecules
   - Surface binding + active site needs

## Tech Stack

### Core Tools (External)
- **RFdiffusion**: Diffusion model for protein backbones (PyTorch, ~50k lines C++/Python)
- **LigandMPNN**: GNN for sequence design (PyTorch, ~5k lines Python)
- **ColabFold**: Fast AlphaFold2 inference (JAX, MMseqs2 for MSA)
- **Rosetta**: Energy minimization and FastRelax (C++, 2M+ lines)

### Chemistry & Structure
- **RDKit**: Molecular structure manipulation, SMILES parsing
- **Biopython**: PDB parsing, sequence handling
- **NumPy**: Coordinate transformations

### Pipeline & Orchestration
- **Pydantic**: Configuration validation with type safety
- **PyYAML**: Config file parsing
- **Jinja2**: SLURM script templating

### Development
- **pytest**: 154 tests (119 unit + 35 integration)
- **Docker/Singularity**: Containerization for reproducibility
- **Black, Ruff, MyPy**: Code formatting and type checking

### Infrastructure
- **SLURM**: HPC job scheduling
- **Git**: Version control with large file handling (LFS for model weights)

## Use Cases

- **PFAS Degradation**: Design defluorinases for perfluorooctanoic acid (PFOA)
- **Microplastic Breakdown**: PETase variants for different polymer substrates
- **Quorum Sensing Disruption**: Lactonases targeting AHL signaling molecules
- **Biofilm Dispersal**: Enzymes for bacterial community disruption

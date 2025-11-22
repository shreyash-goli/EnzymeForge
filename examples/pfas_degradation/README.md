# PFAS Degradation Enzyme Design

This example demonstrates how to use EnzymeForge to design a novel enzyme for degrading PFAS ("forever chemicals").

## Background

PFAS (per- and polyfluoroalkyl substances) are persistent environmental pollutants with extremely strong C-F bonds that resist biological degradation. This example designs an enzyme to catalyze C-F bond breaking in PFOA (perfluorooctanoic acid).

## Target Substrate

**PFOA (Perfluorooctanoic acid)**
- SMILES: `C(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F`
- Challenge: 13 C-F bonds, highly stable
- Target: Defluorination via hydrolytic cleavage

## Catalytic Strategy

Based on haloalkane dehalogenase mechanism:
- **Catalytic residues:** ASP, ARG, TRP
- **Mechanism:** Nucleophilic substitution at C-F bond
- **Products:** Fluoride ion + hydroxylated product

## Running the Design

```bash
# From project root
python -m enzymeforge.pipeline --config examples/pfas_degradation/config.yaml
```

This will:
1. Analyze PFOA structure and identify reactive sites
2. Generate RFdiffusion constraints for fluorine-binding pocket
3. Run backbone generation (manual Colab step)
4. Design sequences with LigandMPNN (manual Colab step)
5. Validate with AlphaFold2 (manual Colab step)
6. Generate report with top designs

## Expected Outputs

```
results/pfas_example/
├── constraints.cst           # Rosetta constraints
├── diffusion/                # RFdiffusion backbones
├── sequences/                # LigandMPNN sequences
├── folding/                  # AlphaFold2 structures
└── report.md                 # Summary of top designs
```

## Design Goals

- High stability (for environmental applications)
- Strong fluorine binding pocket
- Catalytic geometry optimized for C-F activation
- Predicted activity metrics (pLDDT > 70)

## Experimental Validation (Future)

Suggested wet-lab experiments:
1. Gene synthesis (codon-optimized for E. coli)
2. Expression and purification
3. Activity assay: PFOA degradation + fluoride ion release
4. Structural characterization (X-ray crystallography)

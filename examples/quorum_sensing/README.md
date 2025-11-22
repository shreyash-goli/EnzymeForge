# Quorum Sensing Disruptor Enzyme Design

Design a lactonase enzyme that degrades bacterial quorum sensing molecules to prevent biofilm formation and antibiotic resistance.

## Background

Bacteria use quorum sensing (QS) to coordinate group behaviors like biofilm formation and virulence factor production. Disrupting QS by degrading autoinducer molecules (like AHLs) is a promising anti-virulence strategy that doesn't kill bacteria (reducing selection pressure for resistance).

## Target Substrate

**N-acyl homoserine lactone (AHL)**
- Common bacterial QS signal
- Contains lactone ring (target for enzymatic cleavage)
- Mechanism: Lactonase activity → ring opening → inactive molecule

## Catalytic Strategy

Based on metallo-beta-lactamase fold:
- **Catalytic residues:** HIS, HIS, ASP (+ Zn²⁺ cofactor)
- **Mechanism:** Nucleophilic attack on lactone carbonyl
- **Products:** Ring-opened inactive AHL

## Applications

- **Biofilm prevention:** Disrupt bacterial communication → prevent biofilm
- **Anti-virulence therapy:** Reduce pathogenicity without killing bacteria
- **Antibiotic adjuvant:** Combine with antibiotics for enhanced efficacy

## Running the Design

```bash
python -m enzymeforge.pipeline --config examples/quorum_sensing/config.yaml
```

## Design Goals

- High specificity for AHL substrates
- Stable at body temperature (37°C)
- Compatible with E. coli expression
- Minimal off-target effects

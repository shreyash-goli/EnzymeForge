# EnzymeForge

AI-powered platform for designing novel enzymes that degrade environmental toxins through guided de novo protein design.

## Overview

EnzymeForge combines state-of-the-art protein design tools (RFdiffusion, LigandMPNN, AlphaFold2) with chemistry-aware substrate analysis to create custom enzymes for:

- **PFAS degradation** - Breaking down "forever chemicals"
- **Microplastic degradation** - PETase variants for different polymers
- **Quorum sensing disruption** - Enzymes targeting bacterial signaling
- **Biofilm dispersal** - Targeting resistant bacterial communities

## Key Features

- 🧬 **De novo enzyme design** - No existing enzyme required
- 🔬 **Mechanism-aware** - Guides design with catalytic chemistry principles
- 🎯 **Target-specific** - Optimized for environmental toxins
- 🚀 **Colab-friendly** - Runs on free Google Colab GPUs
- 📊 **End-to-end pipeline** - From substrate → designed enzyme

## Quick Start

```bash
# Install
git clone https://github.com/yourusername/enzymeforge.git
cd enzymeforge
pip install -e .

# Run example
python -m enzymeforge.pipeline --config examples/pfas_degradation/config.yaml
```

## Architecture

```
User Input → Substrate Analysis → RFdiffusion → Sequence Design → Validation → Report
              ↓                      ↓              ↓                ↓
           Constraints          Backbones      Sequences         Structures
```

## Project Status

🚧 **MVP in Development** - 8-week timeline to functional PFAS degradation enzyme design

## Citation

If you use EnzymeForge in your research, please cite:

```bibtex
@software{enzymeforge2024,
  title = {EnzymeForge: AI-Powered Enzyme Design for Environmental Toxin Degradation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/enzymeforge}
}
```

## License

MIT License - see LICENSE file

## Acknowledgments

Built on top of:
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) - Baker Lab, UW
- [LigandMPNN](https://github.com/dauparas/LigandMPNN) - Dauparas et al.
- [ColabFold](https://github.com/sokrypton/ColabFold) - Mirdita et al.

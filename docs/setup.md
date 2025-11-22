# EnzymeForge Setup Guide

## Prerequisites

- Python 3.9 or later
- pip or conda
- Google account (for Colab)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/enzymeforge.git
cd enzymeforge
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n enzymeforge python=3.9
conda activate enzymeforge
```

### 3. Install Dependencies

```bash
# Install in development mode
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Run tests
pytest

# Check imports
python -c "from enzymeforge import SubstrateAnalyzer; print('✓ EnzymeForge installed successfully')"
```

## Google Colab Setup

EnzymeForge uses Google Colab for running GPU-intensive tasks (RFdiffusion, LigandMPNN, ColabFold).

### 1. Copy Notebooks to Google Drive

1. Open Google Colab: https://colab.research.google.com
2. Upload notebooks from `enzymeforge/notebooks/`:
   - `colab_rfdiffusion.ipynb`
   - `colab_ligandmpnn.ipynb`
   - `colab_colabfold.ipynb`
3. Save to your Google Drive

### 2. Set Up Google Drive Folder

Create a folder structure in Google Drive:
```
MyDrive/
└── enzymeforge/
    ├── inputs/
    ├── outputs/
    └── notebooks/
```

## Dependencies

### Core Dependencies
- **numpy** - Numerical operations
- **pyyaml** - Configuration parsing
- **biopython** - PDB file manipulation

### Chemistry Dependencies
- **rdkit** - Molecular structure analysis and SMILES parsing

### Optional Dependencies
- **pytest** - Testing
- **mypy** - Type checking
- **black** - Code formatting

## Troubleshooting

### RDKit Installation Issues

If RDKit fails to install via pip:

```bash
# Try conda installation
conda install -c conda-forge rdkit

# Or use mamba (faster)
mamba install -c conda-forge rdkit
```

### BioPython Issues

```bash
# Ensure BioPython is recent version
pip install --upgrade biopython
```

### Import Errors

If you get import errors:

```bash
# Ensure you're in the enzymeforge directory
cd /path/to/enzymeforge

# Install in editable mode
pip install -e .
```

## Next Steps

See [tutorial.md](tutorial.md) for usage instructions.

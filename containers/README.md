# EnzymeForge Containers

Docker and Singularity containers for running EnzymeForge pipelines.

## Available Containers

### 1. RFdiffusion (`rfdiffusion.sif` / `enzymeforge/rfdiffusion`)

Protein backbone generation with RFdiffusion.

**Includes:**
- RFdiffusion (base model)
- PyTorch 1.12.1 + CUDA 11.6
- DGL 1.0.2
- ColabDesign utilities

**Use for:**
- Stage 1: Backbone generation
- Diffusion with guide potentials

### 2. LigandMPNN + Rosetta (`ligandmpnn.sif` / `enzymeforge/ligandmpnn`)

Sequence design with LigandMPNN and structure refinement with Rosetta FastRelax.

**Includes:**
- LigandMPNN for sequence design
- PyRosetta for FastRelax
- pdb2pqr for structure protonation
- PyTorch 2.2.1

**Use for:**
- Stage 2: Sequence design
- Iterative FastRelax + design cycles

### 3. Development Environment (`enzymeforge/dev`)

Full development environment with all dependencies.

**Includes:**
- All EnzymeForge dependencies
- Development tools (pytest, black, ruff, mypy)
- Jupyter notebooks
- RDKit for chemistry

**Use for:**
- Local development
- Testing
- Interactive analysis

## Quick Start

### Docker

Build all containers:
```bash
cd containers/
./build_all.sh docker
```

Run development container:
```bash
docker run -it --gpus all -v $(pwd):/workspace enzymeforge/dev:latest
```

Run specific stage:
```bash
# RFdiffusion
docker run --gpus all -v $(pwd):/workspace \
  enzymeforge/rfdiffusion:latest \
  python scripts/run_diffusion.py --config config.yml

# LigandMPNN
docker run --gpus all -v $(pwd):/workspace \
  enzymeforge/ligandmpnn:latest \
  python scripts/run_seqdesign.py --config config.yml
```

### Singularity

Build all containers:
```bash
cd containers/
./build_all.sh singularity
```

Run on HPC cluster:
```bash
# RFdiffusion
singularity exec --nv \
  -B /data:/data \
  rfdiffusion.sif \
  python scripts/run_diffusion.py --config config.yml

# LigandMPNN
singularity exec --nv \
  -B /data:/data,/scratch:/scratch \
  ligandmpnn.sif \
  python scripts/run_seqdesign.py --config config.yml
```

## Building Containers

### Docker

Build individual containers:
```bash
# RFdiffusion
docker build -f Dockerfile.rfdiffusion -t enzymeforge/rfdiffusion:latest .

# LigandMPNN
docker build -f Dockerfile.ligandmpnn -t enzymeforge/ligandmpnn:latest .

# Development
docker build -f Dockerfile.dev -t enzymeforge/dev:latest .
```

### Singularity

From Docker images (recommended):
```bash
singularity build rfdiffusion.sif docker-daemon://enzymeforge/rfdiffusion:latest
singularity build ligandmpnn.sif docker-daemon://enzymeforge/ligandmpnn:latest
```

From definition files:
```bash
sudo singularity build rfdiffusion.sif rfdiffusion.def
sudo singularity build ligandmpnn.sif ligandmpnn.def
```

## Usage Examples

### Local Development with Docker

```bash
# Start interactive session
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/data:/data \
  enzymeforge/dev:latest

# Inside container
cd /workspace
python scripts/run_full_pipeline.py --config examples/pfas_design.yml
```

### HPC Cluster with Singularity

```bash
# Interactive session
singularity shell --nv \
  -B /data:/data,/scratch:/scratch \
  rfdiffusion.sif

# Batch job (in SLURM script)
singularity exec --nv \
  -B /data:/data \
  --cleanenv \
  rfdiffusion.sif \
  python scripts/run_diffusion.py --config $CONFIG_FILE
```

### Multi-Stage Pipeline

Using SLURM (see [cluster documentation](../src/enzymeforge/cluster/README.md)):

```bash
python scripts/run_cluster.py \
  --config-dir experiments/ \
  --output-dir slurm_jobs/ \
  --cluster-config cluster_config.yml
```

This automatically:
1. Submits diffusion jobs with `rfdiffusion.sif`
2. Submits sequence design jobs with `ligandmpnn.sif` (depends on diffusion)
3. Submits folding jobs with ColabFold container (depends on seqdesign)

## Container Specifications

### RFdiffusion

```dockerfile
Base: nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
Python: 3.9
PyTorch: 1.12.1+cu116
DGL: 1.0.2+cu116
Size: ~8 GB
GPU: Required
```

### LigandMPNN + Rosetta

```dockerfile
Base: nvcr.io/nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
Python: 3.9
PyTorch: 2.2.1
PyRosetta: Latest
Size: ~10 GB
GPU: Optional (recommended for large batches)
```

### Development

```dockerfile
Base: nvcr.io/nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
Python: 3.9
PyTorch: 2.2.1
EnzymeForge: Installed from source
Size: ~12 GB
GPU: Optional
```

## Bind Mounts

Common bind mounts for Singularity:

```bash
# Data directories
-B /data:/data
-B /scratch:/scratch

# Model weights
-B /path/to/models:/models

# ColabFold cache
-B /path/to/colabfold_cache:/cache

# Multiple mounts
-B /data:/data,/scratch:/scratch,/models:/models
```

## GPU Support

### Docker

Enable GPU access:
```bash
docker run --gpus all ...
```

Specify GPU devices:
```bash
docker run --gpus '"device=0,1"' ...
```

### Singularity

Enable NVIDIA GPU:
```bash
singularity exec --nv ...
```

Check GPU availability:
```bash
singularity exec --nv rfdiffusion.sif nvidia-smi
```

## Environment Variables

### RFdiffusion

- `DGLBACKEND=pytorch`: DGL backend
- `PYTHONUNBUFFERED=1`: Unbuffered Python output

### LigandMPNN

- `PYTHONUNBUFFERED=1`: Unbuffered Python output

### Custom Variables

Pass environment variables:
```bash
# Docker
docker run -e MY_VAR=value ...

# Singularity
singularity exec --env MY_VAR=value ...
```

## Troubleshooting

### Docker: Permission Denied

Run with user permissions:
```bash
docker run --user $(id -u):$(id -g) ...
```

### Singularity: Library Not Found

Ensure proper CUDA paths in container:
```bash
singularity exec --nv \
  --env LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  container.sif command
```

### PyRosetta License

PyRosetta requires accepting the license agreement. The containers include automatic installation, but users must comply with the [PyRosetta license](https://www.pyrosetta.org/downloads).

For manual installation:
```bash
singularity exec container.sif \
  python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### Container Size

Reduce container size:
- Use multi-stage builds
- Remove build dependencies after installation
- Use `--squash` flag for Docker builds
- Clean apt cache and pip cache

## Integration with SLURM

See [cluster documentation](../src/enzymeforge/cluster/README.md) for:
- Automatic container selection based on stage
- Bind mount configuration
- GPU resource allocation
- Job dependency management

Example cluster config:
```yaml
containers:
  rfdiffusion: "/containers/rfdiffusion.sif"
  ligandmpnn: "/containers/ligandmpnn.sif"
  folding: "/containers/colabfold.sif"

binds:
  - "/data:/data"
  - "/scratch:/scratch"
```

## Best Practices

1. **Use Singularity on HPC**: Better integration with batch schedulers
2. **Use Docker for local dev**: Easier to build and test
3. **Pin versions**: Ensure reproducibility with specific versions
4. **Mount read-only**: Use `:ro` for input data to prevent modifications
5. **Use named volumes**: Persist data across container restarts
6. **Clean up**: Remove old containers and images regularly

## See Also

- [SLURM cluster support](../src/enzymeforge/cluster/README.md)
- [Example configurations](../examples/)
- [RFdiffusion documentation](https://github.com/RosettaCommons/RFdiffusion)
- [LigandMPNN documentation](https://github.com/dauparas/LigandMPNN)

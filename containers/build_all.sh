#!/bin/bash
# Build all EnzymeForge containers
#
# Usage:
#   ./build_all.sh [docker|singularity]
#
# Examples:
#   ./build_all.sh docker       # Build Docker images
#   ./build_all.sh singularity  # Build Singularity images
#   ./build_all.sh              # Build both

set -e

CONTAINER_TYPE="${1:-both}"

echo "=== Building EnzymeForge Containers ==="
echo "Container type: $CONTAINER_TYPE"
echo ""

# Docker builds
if [ "$CONTAINER_TYPE" = "docker" ] || [ "$CONTAINER_TYPE" = "both" ]; then
    echo "=== Building Docker Images ==="

    echo "[1/3] Building RFdiffusion..."
    docker build -f Dockerfile.rfdiffusion -t enzymeforge/rfdiffusion:latest .

    echo "[2/3] Building LigandMPNN + Rosetta..."
    docker build -f Dockerfile.ligandmpnn -t enzymeforge/ligandmpnn:latest .

    echo "[3/3] Building Development Environment..."
    docker build -f Dockerfile.dev -t enzymeforge/dev:latest .

    echo ""
    echo "Docker images built successfully!"
    echo ""
    docker images | grep enzymeforge
    echo ""
fi

# Singularity builds
if [ "$CONTAINER_TYPE" = "singularity" ] || [ "$CONTAINER_TYPE" = "both" ]; then
    echo "=== Building Singularity Images ==="

    # Check if singularity is available
    if ! command -v singularity &> /dev/null; then
        echo "ERROR: Singularity not found. Please install Singularity first."
        echo "See: https://github.com/sylabs/singularity/releases"
        exit 1
    fi

    echo "[1/2] Building RFdiffusion SIF..."
    if [ "$CONTAINER_TYPE" = "both" ]; then
        # Build from Docker image
        singularity build rfdiffusion.sif docker-daemon://enzymeforge/rfdiffusion:latest
    else
        # Build from definition file
        sudo singularity build rfdiffusion.sif rfdiffusion.def
    fi

    echo "[2/2] Building LigandMPNN + Rosetta SIF..."
    if [ "$CONTAINER_TYPE" = "both" ]; then
        # Build from Docker image
        singularity build ligandmpnn.sif docker-daemon://enzymeforge/ligandmpnn:latest
    else
        # Build from definition file
        sudo singularity build ligandmpnn.sif ligandmpnn.def
    fi

    echo ""
    echo "Singularity images built successfully!"
    echo ""
    ls -lh *.sif
    echo ""
fi

echo "=== Build Complete ==="
echo ""
echo "Docker images:"
echo "  - enzymeforge/rfdiffusion:latest"
echo "  - enzymeforge/ligandmpnn:latest"
echo "  - enzymeforge/dev:latest"
echo ""
echo "Singularity images:"
echo "  - rfdiffusion.sif"
echo "  - ligandmpnn.sif"
echo ""
echo "Usage examples:"
echo ""
echo "  Docker:"
echo "    docker run -it --gpus all -v \$(pwd):/workspace enzymeforge/dev:latest"
echo ""
echo "  Singularity:"
echo "    singularity exec --nv -B /data:/data rfdiffusion.sif python script.py"
echo ""

#!/bin/bash
# Installation script for mpi4jax on Levante GPU nodes with OpenMPI

set -e  # Exit on error

echo "=========================================="
echo "Installing mpi4jax for Levante GPU"
echo "=========================================="

# Activate the Python environment
echo "[1/5] Activating Python environment..."
source /work/mh0033/m300883/Project_week_global/MEssE/build_gpu/messe_env/py_env/bin/activate

# Load OpenMPI module matching NVHPC
echo "[2/5] Loading OpenMPI module..."
module load openmpi/4.1.5-nvhpc-24.7

# Verify MPI is correct
echo "[3/5] Verifying OpenMPI setup..."
echo "Using MPI compiler:"
which mpicc
mpicc -v 2>&1 | head -1

# Install mpi4jax with no cache
echo "[4/5] Installing mpi4jax with OpenMPI-compatible build..."
pip install --no-cache-dir --force-reinstall \
    --no-binary :all: \
    mpi4jax==0.9.0

# Verify installation
echo "[5/5] Verifying installation..."
python -c "import mpi4jax; print('✓ mpi4jax imported successfully')"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="

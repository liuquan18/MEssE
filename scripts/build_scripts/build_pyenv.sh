#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
#set -x
IFS=$'\n\t'

require() { command -v "$1" >/dev/null || { echo "Missing: $1"; exit 1; }; }
require git

# require one argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <directory_where_you_want_to_setup_your_env>" >&2
    exit 1
fi

INPUT="$1"
# check directory exists
if [[ ! -d "$INPUT" ]]; then
    echo "Error: directory does not exist: $INPUT" >&2
    exit 1
fi

# prepare environment
BASE_DIR="$(readlink -f "$INPUT")/messe_env"
echo "Preparing messe_env-levante environment here: $BASE_DIR"

PY_ENV_DIR=$BASE_DIR/py_env

mkdir -p $PY_ENV_DIR

# python module on levante
PY_BIN="/sw/spack-levante/miniforge3-24.11.3-0-Linux-x86_64-2zhbdz/bin/python"
$PY_BIN -m venv $PY_ENV_DIR
source $PY_ENV_DIR/bin/activate

pip install --upgrade pip
pip install setuptools wheel numpy pandas cython pyyaml isodate matplotlib netcdf4 xarray cartopy
# Note: mpi4py should be loaded via spack (spack load py-mpi4py) instead of pip

# mpi-version (for reference if needed)
export MPI_ROOT=/sw/spack-levante/openmpi-4.1.2-mnmady
export CC="${MPI_ROOT}/bin/mpicc"
export CXX="${MPI_ROOT}/bin/mpicxx"
export FC="${MPI_ROOT}/bin/mpif90"
export MPI_LAUNCH="${MPI_ROOT}/bin/mpiexec"

echo "Setup complete."
echo "The python environment is here: $PY_ENV_DIR"
echo "To activate: source $PY_ENV_DIR/bin/activate"
echo "Note: Load mpi4py with 'spack load py-mpi4py' when needed"
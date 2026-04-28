#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
#set -x
IFS=$'\n\t'

require() { command -v "$1" >/dev/null || { echo "Missing: $1"; exit 1; }; }
require git
require make

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

SRC_DIR=$BASE_DIR/src_code
BUILD_DIR=$BASE_DIR/build_dir
PY_ENV_DIR=$BASE_DIR/py_env

mkdir -p $SRC_DIR $BUILD_DIR $PY_ENV_DIR

# python module on levante
PY_BIN="/sw/spack-levante/miniforge3-24.11.3-2-Linux-x86_64-rf4err/bin/python"
$PY_BIN -m venv $PY_ENV_DIR
source $PY_ENV_DIR/bin/activate

pip install --upgrade pip
pip install setuptools wheel pumpy pandas cython pyyaml isodate matplotlib netcdf4 xarray cartopy cupy-cuda12x flax mpi4py
pip install --upgrade "jax[cuda13]"

# Install earth2grid with its CUDA extension into a fresh Python venv.
#
# Proven working combination on Levante:
#   - torch 2.4.0+cu121
#   - CUDA 12.2 toolkit (spack): /sw/spack-levante/cuda-12.2.0-2ttufp
#   - GCC 12.3 (spack, C++20 + max supported by CUDA 12.2): /sw/spack-levante/gcc-12.3.0-ab6j4u
#   - TORCH_CUDA_ARCH_LIST=8.0 (A100 GPU)

pip install --upgrade pip setuptools wheel

# PyTorch 2.4.0 built with CUDA 12.1 — required for earth2grid CUDA extension ABI
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install cartopy cupy-cuda12x flax

# Build earth2grid CUDA extension
# GCC 12.3: supports C++20, and is the maximum GCC version allowed by CUDA 12.2
# CUDA 12.2: compatible with torch 2.4.0+cu121 at extension build time
export CUDA_HOME=/sw/spack-levante/cuda-12.2.0-2ttufp
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/sw/spack-levante/gcc-12.3.0-ab6j4u/bin/gcc
export CXX=/sw/spack-levante/gcc-12.3.0-ab6j4u/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0"

pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz


# clone the repositories
ICON_DIR_NAME=icon-model

git_checkout_or_update() {
    local repo_url="$1"
    local branch="$2"
    local target_dir="$3"

    if [[ -d "$target_dir/.git" ]]; then
        echo "Using existing checkout in $target_dir"
        git -C "$target_dir" fetch --all --prune
        git -C "$target_dir" checkout "$branch"
        git -C "$target_dir" pull --rebase
    else
        echo "Cloning fresh checkout into $target_dir"
        git clone -b "$branch" --recurse-submodules -j8 "$repo_url" "$target_dir"
    fi
}

git_checkout_or_update git@gitlab.dkrz.de:icon/icon-model.git release-2025.10-public $SRC_DIR/$ICON_DIR_NAME

# out-of-source builds for icon
ICON_BUILD_DIR=$BUILD_DIR/$ICON_DIR_NAME
mkdir -p $ICON_BUILD_DIR
pushd $ICON_BUILD_DIR
$SRC_DIR/$ICON_DIR_NAME/config/dkrz/levante.gpu.nvhpc-24.7 --enable-comin --disable-jsbach --disable-quincy --disable-rte-rrtmgp --enable-bundled-python=comin  --disable-silent-rules

make -j $(nproc)
popd

pushd $ICON_BUILD_DIR
./make_runscripts --all
popd

echo "Setup complete."
echo "The python environment is here: $PY_ENV_DIR"

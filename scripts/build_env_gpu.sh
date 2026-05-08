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
pip install setuptools wheel pumpy pandas cython numpy pyyaml isodate matplotlib netcdf4 xarray torch cartopy cupy-cuda12x flax healpy scipy
pip install --upgrade "jax[cuda13]"


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

git_checkout_or_update https://gitlab.dkrz.de/icon/icon-model.git release-2025.10-public $SRC_DIR/$ICON_DIR_NAME

# out-of-source builds for icon
ICON_BUILD_DIR=$BUILD_DIR/$ICON_DIR_NAME
mkdir -p $ICON_BUILD_DIR
pushd $ICON_BUILD_DIR
$SRC_DIR/$ICON_DIR_NAME/config/dkrz/levante.gpu.nvhpc-24.7 --enable-comin --enable-yac --disable-quincy --disable-rte-rrtmgp --enable-bundled-python=comin  --disable-silent-rules
# Export ICON's YAC symbols globally so the Python yac.so shares the same lookup table
# --export-dynamic-symbol requires binutils>=2.35; use --dynamic-list for compatibility with RHEL8 ld (~2.30)
printf '{ yac_*; };\n' > yac_export.list
sed -i "s|^LDFLAGS=.*|& -Wl,--dynamic-list=$(pwd)/yac_export.list|" icon.mk

make -j $(nproc)
./make_runscripts --all

popd


# ===== Build YAC Python bindings =====
# ICON bundles YAC and YAXT in externals/ — reuse the same source.
YAC_SRC_DIR=$SRC_DIR/$ICON_DIR_NAME/externals/yac
YAXT_SRC_DIR=$SRC_DIR/$ICON_DIR_NAME/externals/yaxt
YAXT_BUILD_DIR=$BUILD_DIR/yaxt_python
YAXT_INST_DIR=$BASE_DIR/yaxt_inst
YAC_PY_BUILD_DIR=$BUILD_DIR/yac_python

# Use the exact same MPI and library paths as the ICON GPU build (nvhpc-24.7)
# so that the YAC Python bindings share the same MPI ABI as ICON at runtime.
NVHPC_ROOT=/sw/spack-levante/nvhpc-24.7-py26uc/Linux_x86_64/24.7
YAC_MPI_ROOT=${NVHPC_ROOT}/comm_libs/12.5/openmpi4/openmpi-4.1.5
YAC_CC=${YAC_MPI_ROOT}/bin/mpicc
YAC_FC=${YAC_MPI_ROOT}/bin/mpifort
FYAML_ROOT=/sw/spack-levante/libfyaml-0.7.12-fvbhgo
NETCDF_ROOT=/sw/spack-levante/netcdf-c-main-rpye7o

# mpi4py must be built from source linking nvhpc's libmpi.so (same ABI as ICON/YAC).
MPICC_WRAP_DIR=$(mktemp -d)
OMPI_COMPILE=$(${YAC_CC} --showme:compile)
OMPI_LINK=$(${YAC_CC} --showme:link)
cat > ${MPICC_WRAP_DIR}/mpicc << EOF
#!/bin/bash
exec gcc ${OMPI_COMPILE} ${OMPI_LINK} "\$@"
EOF
chmod +x ${MPICC_WRAP_DIR}/mpicc
MPICC=${MPICC_WRAP_DIR}/mpicc pip install --no-binary :all: --no-cache-dir mpi4py
rm -rf ${MPICC_WRAP_DIR}


# Build YAXT (YAC depends on it)
mkdir -p $YAXT_BUILD_DIR
pushd $YAXT_BUILD_DIR
test -f $YAXT_SRC_DIR/configure || (cd $YAXT_SRC_DIR && ./autogen.sh)
$YAXT_SRC_DIR/configure \
    CC=${YAC_CC} \
    FC=${YAC_FC} \
    --disable-mpi-checks \
    --disable-silent-rules \
    --prefix=$YAXT_INST_DIR
make -j $(nproc)
make install
popd

# Build YAC with Python bindings, install into the venv
# --prefix=$PY_ENV_DIR places yac.so in venv site-packages → import yac works
mkdir -p $YAC_PY_BUILD_DIR
pushd $YAC_PY_BUILD_DIR
test -f $YAC_SRC_DIR/configure || (cd $YAC_SRC_DIR && ./autogen.sh)
$YAC_SRC_DIR/configure \
    CC=${YAC_CC} \
    FC=${YAC_FC} \
    --disable-mpi-checks \
    --disable-silent-rules \
    --enable-python-bindings \
    --with-fyaml-root=$FYAML_ROOT \
    --with-netcdf-root=$NETCDF_ROOT \
    --with-yaxt-root=$YAXT_INST_DIR \
    --prefix=$PY_ENV_DIR
make -j $(nproc)
make install
popd


echo "Setup complete."
echo "The python environment is here: $PY_ENV_DIR"

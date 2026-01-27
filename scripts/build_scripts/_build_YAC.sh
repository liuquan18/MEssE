#!/bin/bash
module load git
# Build YAC
BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located
# Create a work directory and setup the environment
mkdir -p $BASE_DIR/build/YAC

source $BASE_DIR/environment/activate_levante_env

# Environment - use the project's Python virtual environment
source $BASE_DIR/environment/python/py_venv/bin/activate

# Upgrade pip and install dependencies
$BASE_DIR/environment/python/py_venv/bin/pip install --upgrade pip
$BASE_DIR/environment/python/py_venv/bin/pip install wheel cython pyyaml isodate numpy matplotlib mpi4py netcdf4 xarray cartopy

# YAXT
pushd $BASE_DIR/build/YAC
git clone --depth=1 -b "release-0.11.1" https://gitlab.dkrz.de/dkrz-sw/yaxt.git "yaxt"
popd

# Configure, build and install YAXT
pushd $BASE_DIR/build/YAC/yaxt
yaxt_install_dir=`pwd -P`/install
mkdir -p build
cd build
../configure -q --enable-silent-rules --prefix=${yaxt_install_dir}
make -j10
make install
popd

# Run the test suite of YAXT (optional)
pushd $BASE_DIR/build/YAC/yaxt/build
make check
popd

# YAC
pushd $BASE_DIR/build/YAC
git clone --depth=1 -b "release-3.2.0" https://gitlab.dkrz.de/dkrz-sw/yac.git "yac"
popd

# Configure, build and Install YAC
pushd $BASE_DIR/build/YAC/yac
yac_install_dir=`pwd -P`/install
mkdir -p build
cd build
../configure --prefix=${yac_install_dir} \
  CC=mpicc FC=mpif90 CFLAGS="-O0 -g" FCFLAGS="-O0 -g" \
    --with-yaxt-root="${yaxt_install_dir}" \
    MKL_CFLAGS="`pkg-config --cflags mkl-static-lp64-seq`" \
    MKL_CLIBS="`pkg-config --libs mkl-static-lp64-seq`" \
    -q --enable-silent-rules --enable-python-bindings
make -j10
make install
popd

# Install YAC also in python venv
$BASE_DIR/environment/python/py_venv/bin/pip install $BASE_DIR/build/YAC/yac/build/python

# Run the test suite of YAC (optional)
pushd $BASE_DIR/build/YAC/yac/build
make check
popd

# Make
# Build the toy atmosphere and ocean model
# make

# # Run the models separately
# mpirun -n 4 toy_atm.x
# mpirun -n 5 toy_ocn.x

# # Run them simultaneously
# mpirun -n 4 toy_atm.x : -n 5 toy_ocn.x

# # Question: Why does it fail?
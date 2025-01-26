#!/bin/bash

# Build YAC

# Create a work directory and setup the environment
mkdir -p ../../build/YAC
source activate_levante_env

# Environment
python -m venv ${workdir}/.venv
# no need to activate the env. PATH variable is already set in env
pip install --upgrade pip
pip install wheel cython pyyaml isodate numpy matplotlib mpi4py netcdf4 xarray cartopy

# YAXT
pushd ../../build/YAC
git clone --depth=1 -b "release-0.11.1" https://gitlab.dkrz.de/dkrz-sw/yaxt.git "yaxt"
popd

# Configure, build and install YAXT
pushd ../../build/YAC/yaxt
yaxt_install_dir=`pwd -P`/install
mkdir -p build
cd build
../configure -q --enable-silent-rules --prefix=${yaxt_install_dir}
make -j10
make install
popd

# Run the test suite of YAXT (optional)
pushd ../../build/YAC/yaxt/build
make check
popd

# YAC
pushd ../../build/YAC
git clone --depth=1 -b "release-3.2.0" https://gitlab.dkrz.de/dkrz-sw/yac.git "yac"
popd

# Configure, build and Install YAC
pushd ../../build/YAC/yac
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
pip install ../../build/YAC/yac/build/python

# Run the test suite of YAC (optional)
pushd ../../build/YAC/yac/build
make check
popd

# Make
# Build the toy atmosphere and ocean model
make

# Run the models separately
mpirun -n 4 toy_atm.x
mpirun -n 5 toy_ocn.x

# Run them simultaneously
mpirun -n 4 toy_atm.x : -n 5 toy_ocn.x

# Question: Why does it fail?
#!/bin/bash
module load git
# Build YAC
BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located
# Create a work directory and setup the environment
mkdir -p $BASE_DIR/build/YAC

source $BASE_DIR/environment/activate_levante_env

# Set MKL pkg-config path (required for YAC configure)
export PKG_CONFIG_PATH="${MKLROOT}/lib/pkgconfig:${PKG_CONFIG_PATH}"

# Environment
python -m venv ${workdir}/.venv
# no need to activate the env. PATH variable is already set in env
pip install --upgrade pip
pip install wheel cython pyyaml isodate numpy matplotlib mpi4py netcdf4 xarray cartopy

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
pip install $BASE_DIR/build/YAC/yac/build/python

# Run the test suite of YAC (optional)
# pushd $BASE_DIR/build/YAC/yac/build
# make check
# popd

echo ""
echo "========================================="
echo "YAC and YAXT build completed successfully!"
echo "========================================="
echo ""
echo "YAXT installed at: $BASE_DIR/build/YAC/yaxt/install"
echo "YAC installed at: $BASE_DIR/build/YAC/yac/install"
echo ""
echo "Example programs are available in:"
echo "  $BASE_DIR/build/YAC/yac/build/examples/toy_atm_ocn/"
echo ""
echo "To run toy examples:"
echo "  cd $BASE_DIR/build/YAC/yac/build/examples/toy_atm_ocn"
echo "  mpirun -n 4 ./toy_reg2d_atm.x : -n 5 ./toy_reg2d_ocn.x"
echo ""
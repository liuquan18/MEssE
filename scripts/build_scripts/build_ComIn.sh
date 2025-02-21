#!/bin/bash
export ICON_BUILD_DIR=../../build/gcc  # relative path to ICON build directory

source ../../environment/spack_initialize.sh
cd $ICON_BUILD_DIR

spack load py-mpi4py  # issue: should we load the python environment here?
cd externals/comin/build && cmake -DCOMIN_ENABLE_EXAMPLES=ON  -DCOMIN_ENABLE_PYTHON_ADAPTER=ON .

cd $ICON_BUILD_DIR/externals/comin/build && make
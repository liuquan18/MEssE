#!/bin/bash
BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located
export ICON_BUILD_DIR=$BASE_DIR/build/gcc  # relative path to ICON build directory

source $BASE_DIR/environment/spack_initialize.sh
cd $ICON_BUILD_DIR

spack load py-mpi4py  # issue: should we load the python environment here?
cd externals/comin/build && cmake -DCOMIN_ENABLE_EXAMPLES=ON  -DCOMIN_ENABLE_PYTHON_ADAPTER=ON .

cd $ICON_BUILD_DIR/externals/comin/build && make
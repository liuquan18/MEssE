#!/bin/bash
module load git

BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located
# export ICON_BUILD_DIR=$BASE_DIR/build/messe_env/build_dir/icon-model  # relative path to ICON build directory
export ICON_BUILD_DIR=$BASE_DIR/build/gcc  # relative path to ICON build directory

spack load py-mpi4py  # issue: should we load the python environment here?

source $BASE_DIR/build/messe_env/py_env/bin/activate
cd $ICON_BUILD_DIR


cd externals/comin/build && cmake -DCOMIN_ENABLE_EXAMPLES=ON  -DCOMIN_ENABLE_PYTHON_ADAPTER=ON .

cd $ICON_BUILD_DIR/externals/comin/build && make
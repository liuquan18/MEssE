#!/bin/bash

source ../../environment/spack_initialize.sh
cd $ICONDIR/build

spack load py-mpi4py
cd externals/comin/build && cmake -DCOMIN_ENABLE_EXAMPLES=ON  -DCOMIN_ENABLE_PYTHON_ADAPTER=ON .

cd $ICONDIR/build/externals/comin/build && make
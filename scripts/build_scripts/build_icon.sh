#!/bin/bash
# perfom out-of-source build
module load git

export ICONDIR=../../ICON/icon-model  # relative path to ICON source code

mkdir -p ../../build/gcc && cd $_ 

$ICONDIR/config/dkrz/levante.gcc -q --enable-comin

# make -j4 2>&1 > compile.log
make -j8

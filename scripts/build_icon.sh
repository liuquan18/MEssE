#!/bin/bash
# perfom in source build

export ICONDIR=../ICON/icon-model  # relative path to ICON source code

mkdir -p ../build/gcc && cd ../build/gcc

$ICONDIR/config/dkrz/levante.gcc -q --enable-comin

make -j4 2>&1 > compile.log

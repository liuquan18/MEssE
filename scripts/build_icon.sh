#!/bin/bash

export ICONDIR=../ICON/icon-model  # relative path to ICON source code


mkdir -p ../build
cd ../build
../config/dkrz/levante.gcc -q --enable-comin

make -j4 2>&1 > compile.log
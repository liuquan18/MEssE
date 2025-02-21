#!/bin/bash
# perfom out-of-source build
BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located

module load git

if [ ! -d "../../ICON/icon-model" ]; then
    git clone git@gitlab.dkrz.de:icon/icon-model.git ../../ICON/icon-model
fi
export ICONDIR=$BASE_DIR/ICON/icon-model  # relative path to ICON source code

mkdir -p $BASE_DIR/build/gcc && cd $_ 

$ICONDIR/config/dkrz/levante.gcc -q --enable-comin

# make -j4 2>&1 > compile.log
make -j8

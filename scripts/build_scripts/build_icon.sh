#!/bin/bash
module load git
# perfom out-of-source build
BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located

if [ ! -d "$BASE_DIR/ICON/icon-model" ]; then
    git clone git@gitlab.dkrz.de:icon/icon-model.git $BASE_DIR/ICON/icon-model
    cd $BASE_DIR/ICON/icon-model
    git submodule update --init --recursive
fi
export ICONDIR=$BASE_DIR/ICON/icon-model  # relative path to ICON source code

# Ensure submodules are initialized (in case the directory already existed)
cd $ICONDIR
git submodule update --init --recursive

mkdir -p $BASE_DIR/build/gcc && cd $_ 

$ICONDIR/config/dkrz/levante.gcc -q --enable-comin --enable-mixed-precision --enable-openmp --enable-bundled-python='mtime','yac','comin'

# make -j4 2>&1 > compile.log
make -j8

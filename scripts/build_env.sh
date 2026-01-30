#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
#set -x
IFS=$'\n\t'

require() { command -v "$1" >/dev/null || { echo "Missing: $1"; exit 1; }; }
require git
require make

# require one argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <directory_where_you_want_to_setup_your_env>" >&2
    exit 1
fi

INPUT="$1"
# check directory exists
if [[ ! -d "$INPUT" ]]; then
    echo "Error: directory does not exist: $INPUT" >&2
    exit 1
fi

# prepare environment
BASE_DIR="$(readlink -f "$INPUT")/messe_env"
echo "Preparing messe_env-levante environment here: $BASE_DIR"

SRC_DIR=$BASE_DIR/src_code
BUILD_DIR=$BASE_DIR/build_dir
PY_ENV_DIR=$BASE_DIR/py_env

mkdir -p $SRC_DIR $BUILD_DIR $PY_ENV_DIR

# python module on levante
PY_BIN="/sw/spack-levante/miniforge3-24.11.3-0-Linux-x86_64-2zhbdz/bin/python"
$PY_BIN -m venv $PY_ENV_DIR
source $PY_ENV_DIR/bin/activate

pip install --upgrade pip
pip install setuptools wheel pumpy pandas cython pyyaml isodate matplotlib mpi4py netcdf4 xarray torch cartopy

# clone the repositories
ICON_DIR_NAME=icon-model

git_checkout_or_update() {
    local repo_url="$1"
    local branch="$2"
    local target_dir="$3"

    if [[ -d "$target_dir/.git" ]]; then
        echo "Using existing checkout in $target_dir"
        git -C "$target_dir" fetch --all --prune
        git -C "$target_dir" checkout "$branch"
        git -C "$target_dir" pull --rebase
    else
        echo "Cloning fresh checkout into $target_dir"
        git clone -b "$branch" --recurse-submodules -j8 "$repo_url" "$target_dir"
    fi
}

git_checkout_or_update git@gitlab.dkrz.de:icon/icon-model.git release-2025.10-public $SRC_DIR/$ICON_DIR_NAME

# out-of-source builds for icon
ICON_BUILD_DIR=$BUILD_DIR/$ICON_DIR_NAME
mkdir -p $ICON_BUILD_DIR
pushd $ICON_BUILD_DIR
$SRC_DIR/$ICON_DIR_NAME/config/dkrz/levante.intel-2021.5.0 ICON_BUNDLED_CFLAGS='-fPIC -O2' -q --enable-comin --enable-mixed-precision --enable-openmp --enable-bundled-python='mtime','yac','comin'
make -j $(nproc)
popd


echo "Setup complete."
echo "The python environment is here: $PY_ENV_DIR"

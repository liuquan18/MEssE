#!/bin/bash
module load git
BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located

spack load /fwv # this loads a proper python installation
spack load /mnm # tells mpi4py which MPI to use

python -m venv $BASE_DIR/environment/python/py_venv

# activate the environment
source $BASE_DIR/environment/python/py_venv/bin/activate
pip install pandas numpy mpi4py xarray torch matplotlib
pip install -e $BASE_DIR/scripts/plugin # make the functions in /src available in the venv
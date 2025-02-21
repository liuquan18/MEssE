#!/bin/bash
BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located

spack load /fwv # this loads a proper python installation
python -m venv $BASE_DIR/environment/python/py_venv
../../environment/python/py_venv/bin/activate
pip install pandas

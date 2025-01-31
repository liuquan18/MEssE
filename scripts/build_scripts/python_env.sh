#!/bin/bash

spack load /fwv # this loads a proper python installation
python -m venv ../../environment/python/py_venv
../../environment/python/py_venv/bin/activate
pip install pandas

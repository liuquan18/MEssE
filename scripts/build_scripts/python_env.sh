#!/bin/bash

spack load /fwv # this loads a proper python installation
python -m venv ../../environment/python/AICON
../../environment/python/AICON/bin/activate
pip install pandas

#!/bin/bash

for COMP in atm ocn common spmd
do
  sed -e '/! TODO/,/! END TODO/ {/! TODO/b; d}' < reference/toy_$COMP.F90 > toy_$COMP.F90
done

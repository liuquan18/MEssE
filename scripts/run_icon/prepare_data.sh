#!/bin/bash
module load git

BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located

# Base directory for ICON sources and binaries
export ICONDIR=$BASE_DIR/build/gcc

# Absolute path to directory with plenty of space
export EXPDIR=$BASE_DIR/experiment
if [ ! -d $EXPDIR ]; then
    mkdir -p $EXPDIR
fi

# Preparing necessary data in the output directory (data from workshop)
cp /home/m/m300883/comin-training-exercises/exercise/prepared/icon_master.namelist $EXPDIR/
cp /home/m/m300883/comin-training-exercises/exercise/prepared/NAMELIST_ICON $EXPDIR/
cp /home/m/m300883/comin-training-exercises/exercise/prepared/icon-lam.sbatch $EXPDIR/


# Directory with input grids and external data
export GRIDDIR=/pool/data/ICON/ICON_training/exercise_lam/grids
# Directory with initial data
export DATADIR=/pool/data/ICON/ICON_training/exercise_lam/data_lam

cd ${EXPDIR}

# Link data needed for radiation
ln -sf ${ICONDIR}/externals/ecrad/data ecrad_data

# Grid files: link to output directory
ln -sf ${GRIDDIR}/*.nc .
# Data files
ln -sf ${DATADIR}/* .

# Dictionaries for the mapping: DWD GRIB2 names <-> ICON internal names
ln -sf ${ICONDIR}/run/ana_varnames_map_file.txt .
ln -sf ${GRIDDIR}/../exercise_lam/map_file.latbc .

# For output: Dictionary for the mapping: names specified in the output nml <-> ICON internal names
ln -sf ${ICONDIR}/run/dict.output.dwd dict.output.dwd

#!/bin/bash

# Base directory for ICON sources and binaries
export SCRATCHDIR=/scratch/${USER::1}/$USER
# export ICONDIR=${SCRATCHDIR}/icon
export ICONDIR=/work/mh0033/m300883/ComIn/icon-model

# Absolute path to directory with plenty of space
export EXPDIR=${SCRATCHDIR}/icon_exercise_comin
if [ ! -d $EXPDIR ]; then
    mkdir -p $EXPDIR
fi

# Preparing necessary data in the output directory
export SCRIPTDIR=$HOME/comin-training-exercises/exercise
cp $SCRIPTDIR/prepared/icon_master.namelist $EXPDIR/
cp $SCRIPTDIR/prepared/NAMELIST_ICON $EXPDIR/
cp $SCRIPTDIR/prepared/icon-lam.sbatch $EXPDIR/

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

# Adding a new output namelist
cat >> $EXPDIR/NAMELIST_ICON << EOF
! output_nml: specifies an output stream --------------------------------------
&output_nml
 filetype                    = 4                     ! netcdf
 dom                         = 1
 output_bounds               = 0., 10000000., 3600.  ! start, end, increment
 steps_per_file              = 5
 mode                        = 1
 include_last                = .FALSE.
 steps_per_file_inclfirst    = .FALSE.
 output_filename             = 'NWP_LAM'
 filename_format             = '<output_filename>_DOM<physdom>_<datetime2>'
 output_grid                 = .FALSE.
 remap                       = 1                     ! 1: remap to lat-lon grid
 reg_lon_def                 = 0.8,0.1,17.2
 reg_lat_def                 = 43.9,0.1,57.7
 ml_varlist                  = "RHI_MAX", "QI_MAX"

/
EOF

# Adding comin_nml
export SCRIPTDIR=$HOME/comin-training-exercises/exercise/scripts
cat >> $EXPDIR/NAMELIST_ICON << EOF
&comin_nml
   plugin_list(1)%name           = "comin_plugin"
   plugin_list(1)%plugin_library = "$ICONDIR/build/externals/comin/build/plugins/python_adapter/libpython_adapter.so"
   plugin_list(1)%options        = "$SCRIPTDIR/comin_plugin.py"
/
EOF

# The ICON batch job
cd $EXPDIR
pwd
echo $ICONDIR

export ICONDIR=$ICONDIR
export EXPDIR=$EXPDIR

cd $EXPDIR && sbatch --account $SLURM_JOB_ACCOUNT icon-lam.sbatch
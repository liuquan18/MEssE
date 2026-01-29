#!/bin/bash
module load git

BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located

export ICONDIR=$BASE_DIR/build/gcc
export EXPDIR=$BASE_DIR/experiment
export PYENVS=$BASE_DIR/build/messe_env/py_env

# Ensure we have a fresh NAMELIST_ICON by copying the prepared version
cp /work/mh0033/m300883/comin-training-exercises/exercise/prepared/NAMELIST_ICON $EXPDIR/

# Adding a new output namelist
cat >> $EXPDIR/NAMELIST_ICON << EOF
! output_nml: specifies an output stream --------------------------------------
&output_nml
 filetype                    = 4                     ! netcdf
 dom                         = -1
 output_bounds               = 0., 10000000., 3600.  ! start, end, increment
 steps_per_file              = 5
 mode                        = 1
 include_last                = .FALSE.
 steps_per_file_inclfirst    = .FALSE.
 output_filename             = 'NWP_LAM'
 filename_format             = '<output_filename>_DOM<physdom>_<datetime2>'
 output_grid                 = .TRUE.
 remap                       = 0                     ! 0: native ICON grid, 1: remap to lat-lon grid
 ml_varlist                  = "RHI_MAX", "QI_MAX"    ! list of variables to be written to the output file

/
EOF

# Adding comin_nml
export SCRIPTDIR=$BASE_DIR/scripts/plugin/scripts

cat >> $EXPDIR/NAMELIST_ICON << EOF
&comin_nml
   plugin_list(1)%name           = "comin_plugin"
   plugin_list(1)%plugin_library = "$ICONDIR/externals/comin/build/plugins/python_adapter/libpython_adapter.so"
   plugin_list(1)%options        = "$SCRIPTDIR/comin_plugin.py"
/
/
EOF


# The ICON batch job
cd $EXPDIR && sbatch --account mh0033 icon-lam.sbatch
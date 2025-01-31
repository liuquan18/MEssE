#!/bin/bash
export ICONDIR=../../build/gcc
export EXPDIR=../../experiment

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
 ml_varlist                  = "RHI_MAX", "QI_MAX"    ! list of variables to be written to the output file

/
EOF

# Adding comin_nml
export SCRIPTDIR=../plugin/scripts

cat >> $EXPDIR/NAMELIST_ICON << EOF
&comin_nml
   plugin_list(1)%name           = "comin_plugin"
   plugin_list(1)%plugin_library = "$ICONDIR/build/externals/comin/build/plugins/python_adapter/libpython_adapter.so"
   plugin_list(1)%options        = "$SCRIPTDIR/comin_plugin.py"
/
EOF

# The ICON batch job

cd $EXPDIR && sbatch --account $SLURM_JOB_ACCOUNT icon-lam.sbatch
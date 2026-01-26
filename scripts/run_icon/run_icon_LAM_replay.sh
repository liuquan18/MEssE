#!/bin/bash
module load git

BASE_DIR="$(git rev-parse --show-toplevel)" # where the repository is located

export ICONDIR=$BASE_DIR/build/gcc
export EXPDIR=$BASE_DIR/experiment

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
&replay_tool_nml
  replay_data_path = "path/to/the/replay_data/"
  msg_level        = 42
/
&comin_nml
  plugin_list(1)%name           = "var_replay_plugin"
  plugin_list(1)%plugin_library = "$COMIN_DIR/build/replay_tool/libcomin_var_replay_plugin.so"
  plugin_list(2)%name           = "simple_python_plugin"
  plugin_list(2)%plugin_library = "libpython_adapter.so"
  plugin_list(2)%options        = "$PLUGINDIR/simple_python_plugin.py"
/
EOF

# The ICON batch job

cd $EXPDIR && sbatch --account $SLURM_JOB_ACCOUNT icon-lam.sbatch
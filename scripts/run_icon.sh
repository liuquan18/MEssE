#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
#set -x
IFS=$'\n\t'

# require three arguments
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <ICON_BUILD_DIR> <COMIN_PLUGIN_SCRIPT> <LEVANTE_ACCOUNT>" >&2
    echo "  ICON_BUILD_DIR:       Path to ICON build directory" >&2
    echo "  COMIN_PLUGIN_SCRIPT:  Path to the ComIn plugin Python script" >&2
    echo "  LEVANTE_ACCOUNT:      Levante account for sbatch submission" >&2
    exit 1
fi

ICON_BUILD_DIR="$1"
COMIN_PLUGIN_SCRIPT="$2"
LEVANTE_ACCOUNT="$3"

# check ICON_BUILD_DIR exists
if [[ ! -d "$ICON_BUILD_DIR" ]]; then
    echo "Error: ICON_BUILD_DIR does not exist: $ICON_BUILD_DIR" >&2
    exit 1
fi

# check COMIN_PLUGIN_SCRIPT exists
if [[ ! -f "$COMIN_PLUGIN_SCRIPT" ]]; then
    echo "Error: COMIN_PLUGIN_SCRIPT does not exist: $COMIN_PLUGIN_SCRIPT" >&2
    exit 1
fi

pushd ${ICON_BUILD_DIR}
./make_runscripts esm_bb_ruby0

ICON_RUN_SCRIPT=${ICON_BUILD_DIR}/run/exp.esm_bb_ruby0.run

# Add comin_nml to the atmo_namelist block in the generated run script
sed -i "/cat > \${atmo_namelist} << EOF/,/^EOF$/{/^EOF$/i\\
&comin_nml\\
   plugin_list(1)%name           = \"comin_plugin\"\\
   plugin_list(1)%plugin_library = \"${ICON_BUILD_DIR}/externals/comin/build/plugins/python_adapter/libpython_adapter.so\"\\
   plugin_list(1)%options        = \"$COMIN_PLUGIN_SCRIPT\"\\
/
}" ${ICON_RUN_SCRIPT}

# Fix hardcoded paths in the generated run script
# Replace old messe_env path with project-relative path
OLD_PATH="/work/mh1498/m301257/messe_env"
NEW_PATH="/work/mh1498/m301257/project/MEssE-1/messe_env"
sed -i "s|${OLD_PATH}|${NEW_PATH}|g" ${ICON_RUN_SCRIPT}

# Create outputs directory if it doesn't exist
OUTPUTS_DIR="/work/mh1498/m301257/project/MEssE-1/outputs"
mkdir -p "$OUTPUTS_DIR"

# Adjust experiment runtime to 24 hours (default generated script used 6 hours)
if [[ -f "${ICON_RUN_SCRIPT}" ]]; then
    sed -i 's|end_date=${end_date:="1979-01-01T06:00:00Z"}|end_date=${end_date:="1979-01-02T00:00:00Z"}|' ${ICON_RUN_SCRIPT}
    sed -i 's|restart_interval="PT6H"|restart_interval="PT24H"|' ${ICON_RUN_SCRIPT} || true
fi

# Submit job with output redirection to outputs directory
sbatch --account=$LEVANTE_ACCOUNT \
       --output="${OUTPUTS_DIR}/icon_job_%j.out" \
       --error="${OUTPUTS_DIR}/icon_job_%j.err" \
       $ICON_RUN_SCRIPT

popd

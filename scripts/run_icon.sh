#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
#set -x
IFS=$'\n\t'

# require three to six arguments
if [[ $# -lt 3 ]] || [[ $# -gt 6 ]]; then
    echo "Usage: $0 <ICON_BUILD_DIR> <COMIN_PLUGIN_SCRIPT> <LEVANTE_ACCOUNT> [START_DATE] [END_DATE] [NUM_NODES]" >&2
    echo "  ICON_BUILD_DIR:       Path to ICON build directory" >&2
    echo "  COMIN_PLUGIN_SCRIPT:  Path to the ComIn plugin Python script" >&2
    echo "  LEVANTE_ACCOUNT:      Levante account for sbatch submission" >&2
    echo "  START_DATE:           Optional start date (default: 1979-01-01T00:00:00Z)" >&2
    echo "  END_DATE:             Optional end date (default: 1979-01-01T06:00:00Z)" >&2
    echo "  NUM_NODES:            Optional number of nodes (default: 1)" >&2
    exit 1
fi

ICON_BUILD_DIR="$1"
COMIN_PLUGIN_SCRIPT="$2"
LEVANTE_ACCOUNT="$3"
START_DATE="${4:-1979-01-01T00:00:00Z}"
END_DATE="${5:-1979-01-01T06:00:00Z}"
NUM_NODES="${6:-1}"

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

# Modify start_date in the run script
sed -i "s|start_date=\${start_date:=.*}|start_date=\${start_date:=\"${START_DATE}\"}|" ${ICON_RUN_SCRIPT}

# Modify end_date in the run script
sed -i "s|end_date=\${end_date:=.*}|end_date=\${end_date:=\"${END_DATE}\"}|" ${ICON_RUN_SCRIPT}

# Modify number of nodes in SBATCH header
sed -i "s|^#SBATCH --nodes=.*|#SBATCH --nodes=${NUM_NODES}|" ${ICON_RUN_SCRIPT}

# Set restart_interval to avoid auto-restart (use a long interval like 100 years)
sed -i 's|restart_interval="PT6H"|restart_interval="P100Y"|' ${ICON_RUN_SCRIPT}

# Add comin_nml to the atmo_namelist block in the generated run script
sed -i "/cat > \${atmo_namelist} << EOF/,/^EOF$/{/^EOF$/i\\
&comin_nml\\
   plugin_list(1)%name           = \"comin_plugin\"\\
   plugin_list(1)%plugin_library = \"${ICON_BUILD_DIR}/externals/comin/build/plugins/python_adapter/libpython_adapter.so\"\\
   plugin_list(1)%options        = \"$COMIN_PLUGIN_SCRIPT\"\\
/
}" ${ICON_RUN_SCRIPT}

sbatch --account=$LEVANTE_ACCOUNT $ICON_RUN_SCRIPT

popd

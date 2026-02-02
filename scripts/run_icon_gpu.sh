#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
#set -x
IFS=$'\n\t'

# GPU-enabled version of run_icon.sh
# Automatically requests GPU partition and A100 GPU resources

# require three to seven arguments
if [[ $# -lt 3 ]] || [[ $# -gt 7 ]]; then
    echo "Usage: $0 <ICON_BUILD_DIR> <COMIN_PLUGIN_SCRIPT> <LEVANTE_ACCOUNT> [START_DATE] [END_DATE] [NUM_NODES] [TIME_LIMIT]" >&2
    echo "  ICON_BUILD_DIR:       Path to ICON build directory" >&2
    echo "  COMIN_PLUGIN_SCRIPT:  Path to the ComIn plugin Python script" >&2
    echo "  LEVANTE_ACCOUNT:      Levante account for sbatch submission" >&2
    echo "  START_DATE:           Optional start date (default: 1979-01-01T00:00:00Z)" >&2
    echo "  END_DATE:             Optional end date (default: 1979-01-01T06:00:00Z)" >&2
    echo "  NUM_NODES:            Optional number of nodes (default: 1)" >&2
    echo "  TIME_LIMIT:           Optional time limit in HH:MM:SS format (default: 02:00:00)" >&2
    exit 1
fi

ICON_BUILD_DIR="$1"
COMIN_PLUGIN_SCRIPT="$2"
LEVANTE_ACCOUNT="$3"
START_DATE="${4:-1979-01-01T00:00:00Z}"
END_DATE="${5:-1979-01-01T06:00:00Z}"
NUM_NODES="${6:-1}"
TIME_LIMIT="${7:-02:00:00}"

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
sed -i "s|start_date=\\$\{start_date:=.*}|start_date=\\$\{start_date:="",
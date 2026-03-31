#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

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

if [[ ! -d "$ICON_BUILD_DIR" ]]; then
    echo "Error: ICON_BUILD_DIR does not exist: $ICON_BUILD_DIR" >&2
    exit 1
fi

if [[ ! -f "$COMIN_PLUGIN_SCRIPT" ]]; then
    echo "Error: COMIN_PLUGIN_SCRIPT does not exist: $COMIN_PLUGIN_SCRIPT" >&2
    exit 1
fi

ICON_RUN_SCRIPT="${ICON_BUILD_DIR}/run/exp.atm_tracer_Hadley_comin_portability.run"
if [[ ! -f "$ICON_RUN_SCRIPT" ]]; then
    echo "Error: ICON runscript not found: $ICON_RUN_SCRIPT" >&2
    exit 1
fi

PYTHON_ADAPTER_LIB="${ICON_BUILD_DIR}/externals/comin/build/plugins/python_adapter/libpython_adapter.so"
if [[ ! -f "$PYTHON_ADAPTER_LIB" ]]; then
    echo "Error: Python adapter library not found: $PYTHON_ADAPTER_LIB" >&2
    exit 1
fi

# Minimal updates to the existing runscript before submission.
sed -i "s|^#SBATCH --account=.*|#SBATCH --account=${LEVANTE_ACCOUNT}|" "$ICON_RUN_SCRIPT"
sed -i "s|^[[:space:]]*plugin_list(1)%plugin_library[[:space:]]*=.*|  plugin_list(1)%plugin_library = \"${PYTHON_ADAPTER_LIB}\"|" "$ICON_RUN_SCRIPT"
sed -i "s|^[[:space:]]*plugin_list(1)%options[[:space:]]*=.*|  plugin_list(1)%options        = \"${COMIN_PLUGIN_SCRIPT}\"|" "$ICON_RUN_SCRIPT"


# clean the experiment directory to avoid issues with previous runs
ICON_EXP_DIR="${ICON_BUILD_DIR}/experiments/"
echo "Cleaning experiment directory: ${ICON_EXP_DIR}"
rm -rf "${ICON_EXP_DIR}/*"


echo "Submitting: $ICON_RUN_SCRIPT"
sbatch "$ICON_RUN_SCRIPT"

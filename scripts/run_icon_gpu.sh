#!/usr/bin/env bash
# Create an ICON experiment with mkexp from an mkexp config file and submit it,
# with a MEssE ComIn plugin (ComIn 1.0) loaded through ComIn's python adapter.
set -euo pipefail
IFS=$'\n\t'

usage() {
    cat >&2 << EOF
Usage: $0 <ICON_DIR> <MKEXP_CONFIG> <COMIN_PLUGIN_SCRIPT> <LEVANTE_ACCOUNT> <EXPERIMENT_NAME> [NUM_NODES] [KEY=VALUE ...]
  ICON_DIR:             ICON base directory with src_code/icon-model, build_dir/icon-model and py_env
                        (as made by build_env_gpu.sh), e.g. /work/mh0033/m300883/Project_week_global/ICON
  MKEXP_CONFIG:         mkexp experiment config, e.g. MEssE/scripts/amip-aes_R2B8_gpu.config
  COMIN_PLUGIN_SCRIPT:  Path to the ComIn plugin Python script
  LEVANTE_ACCOUNT:      Levante account for sbatch submission
  EXPERIMENT_NAME:      Experiment id (mkexp EXP_ID)
  NUM_NODES:            Optional number of GPU nodes (default: the config's jobs.run.nodes)
  KEY=VALUE:            Optional further mkexp settings, e.g. jobs.run.time_limit=00:30:00
                        MESSE_RF_CONFIG_FILE=/path/to/config.yaml FINAL_DATE=2020-01-10T00:00:00
Set NO_SUBMIT=1 to create the experiment scripts without submitting them.
EOF
    exit 1
}

if [[ $# -lt 5 ]]; then
    usage
fi

ICON_DIR="$(realpath "$1")"
MKEXP_CONFIG="$(realpath "$2")"
COMIN_PLUGIN_SCRIPT="$(realpath "$3")"
LEVANTE_ACCOUNT="$4"
EXPERIMENT_NAME="$5"
shift 5

ARGS=(
    "EXP_ID=${EXPERIMENT_NAME}"
    "ACCOUNT=${LEVANTE_ACCOUNT}"
    "COMIN_PLUGIN_SCRIPT=${COMIN_PLUGIN_SCRIPT}"
    "MESSE_PYTHON=${ICON_DIR}/py_env/bin/python3"
)
if [[ $# -gt 0 && "$1" != *=* ]]; then
    if ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: NUM_NODES must be a positive integer, got: $1" >&2
        exit 1
    fi
    ARGS+=("jobs.run.nodes=$1")
    shift
fi
for arg in "$@"; do
    if [[ "$arg" != *=* ]]; then
        echo "Error: expected KEY=VALUE, got: $arg" >&2
        exit 1
    fi
    ARGS+=("$arg")
done

ICON_SRC_DIR="${ICON_DIR}/src_code/icon-model"
ICON_BUILD_DIR="${ICON_DIR}/build_dir/icon-model"
PYTHON="${ICON_DIR}/py_env/bin/python3"
PYTHON_ADAPTER_LIB="${ICON_BUILD_DIR}/externals/comin/build/_icon/plugins/python_adapter/libpython_adapter.so"

for path in "$ICON_SRC_DIR/etc/Modules" "$ICON_BUILD_DIR/run/SETUP.config" "$ICON_BUILD_DIR/bin/icon" \
            "$PYTHON" "$PYTHON_ADAPTER_LIB" "$MKEXP_CONFIG" "$COMIN_PLUGIN_SCRIPT"; do
    if [[ ! -e "$path" ]]; then
        echo "Error: not found: $path" >&2
        exit 1
    fi
done

# ComIn's python adapter runs the plugin in the interpreter it was linked against and
# stops if the env's Python has another version (the env holds torch and messe).
ADAPTER_PY="$(readelf -d "$PYTHON_ADAPTER_LIB" | grep -o 'libpython3\.[0-9]*' | head -1 | sed 's/^lib//')"
ENV_PY="$("$PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
if [[ "$ADAPTER_PY" != "$ENV_PY" ]]; then
    echo "Error: ComIn's python adapter is linked against ${ADAPTER_PY}, but ${PYTHON} is ${ENV_PY}." >&2
    echo "       Rebuild ICON with configure argument PYTHON=${PYTHON} (see build_env_gpu.sh)." >&2
    exit 1
fi

# mkexp and getexp come with ICON's module; ICON_BUILD_DIR points it at the out-of-source build.
if ! type module > /dev/null 2>&1; then
    source /usr/share/Modules/init/bash
fi
export ICON_BUILD_DIR
module use "${ICON_SRC_DIR}/etc/Modules"
module load icon-levante

# The config is found by its path, ICON's standard experiments through MKEXP_PATH.
cd "$(dirname "$MKEXP_CONFIG")"
getexp_key() {
    getexp -k "$1" "$MKEXP_CONFIG" "${ARGS[@]}"
}
SCRIPT_DIR="$(getexp_key SCRIPT_DIR)"
WORK_DIR="$(getexp_key WORK_DIR)"

# A leftover saved_models/ in the run directory would be resumed by the plugin.
if [[ -d "$WORK_DIR" ]]; then
    echo "Cleaning existing run directory: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
fi

mkexp "$MKEXP_CONFIG" "${ARGS[@]}"

if [[ "${NO_SUBMIT:-0}" == "1" ]]; then
    echo "NO_SUBMIT=1: not submitting. Submit with: cd ${SCRIPT_DIR} && sbatch ${EXPERIMENT_NAME}.run_start"
    exit 0
fi
cd "$SCRIPT_DIR"
echo "Submitting: ${SCRIPT_DIR}/${EXPERIMENT_NAME}.run_start"
sbatch "${EXPERIMENT_NAME}.run_start"

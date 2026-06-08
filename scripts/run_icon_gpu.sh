#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: $0 <ICON_BUILD_DIR> <COMIN_PLUGIN_SCRIPT> <LEVANTE_ACCOUNT> <EXPERIMENT_NAME> [NUM_NODES]" >&2
    echo "  ICON_BUILD_DIR:       Path to ICON build directory" >&2
    echo "  COMIN_PLUGIN_SCRIPT:  Path to the ComIn plugin Python script" >&2
    echo "  LEVANTE_ACCOUNT:      Levante account for sbatch submission" >&2
    echo "  EXPERIMENT_NAME:      Name for the experiment (used as EXPNAME and output folder)" >&2
    echo "  NUM_NODES:            Optional number of nodes (default: 1)" >&2
    exit 1
fi

ICON_BUILD_DIR="$1"
COMIN_PLUGIN_SCRIPT="$2"
LEVANTE_ACCOUNT="$3"
EXPERIMENT_NAME="$4"
NUM_NODES="${5:-1}"

if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_NODES must be a positive integer, got: $NUM_NODES" >&2
    exit 1
fi

if [[ ! -d "$ICON_BUILD_DIR" ]]; then
    echo "Error: ICON_BUILD_DIR does not exist: $ICON_BUILD_DIR" >&2
    exit 1
fi

if [[ ! -f "$COMIN_PLUGIN_SCRIPT" ]]; then
    echo "Error: COMIN_PLUGIN_SCRIPT does not exist: $COMIN_PLUGIN_SCRIPT" >&2
    exit 1
fi

# ICON_RUN_SCRIPT="${ICON_BUILD_DIR}/run/exp.aes_amip_messe_test.run"
ICON_RUN_SCRIPT="${ICON_BUILD_DIR}/run/exp.atm_nwp_jsbach_xpp_r2b4.run"

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
sed -i "s|^#SBATCH --nodes=.*|#SBATCH --nodes=${NUM_NODES}|" "$ICON_RUN_SCRIPT"
sed -i "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${EXPERIMENT_NAME}|" "$ICON_RUN_SCRIPT"
sed -i "s|^export EXPNAME=.*|export EXPNAME=\"${EXPERIMENT_NAME}\"|" "$ICON_RUN_SCRIPT"
sed -i "s|^job_name=.*|job_name=\"exp.${EXPERIMENT_NAME}.run\"|" "$ICON_RUN_SCRIPT"
sed -i "s|^[[:space:]]*plugin_list(1)%plugin_library[[:space:]]*=.*|  plugin_list(1)%plugin_library = \"${PYTHON_ADAPTER_LIB}\"|" "$ICON_RUN_SCRIPT"
sed -i "s|^[[:space:]]*plugin_list(1)%options[[:space:]]*=.*|  plugin_list(1)%options        = \"${COMIN_PLUGIN_SCRIPT}\"|" "$ICON_RUN_SCRIPT"

# Apply DKRZ-recommended OpenMPI runtime settings (https://docs.dkrz.de/doc/levante/running-jobs/runtime-settings.html).
# Force UCX as the PML and disable other BTLs so Open MPI does not fall back to
# BTL sm (which fails with ptrace_scope=3 set on Levante after the May 2026 maintenance).
LEVANTE_WRAPPER="${ICON_BUILD_DIR}/run/run_wrapper/levante.sh"
sed -i "s|export UCX_TLS=.*|export OMPI_MCA_pml=ucx\n    export OMPI_MCA_btl=self\n    export UCX_RNDV_SCHEME=put_zcopy\n    export UCX_IB_GPU_DIRECT_RDMA=yes\n    export UCX_MEMTYPE_CACHE=n\n    export UCX_TLS=cma,mm,rc,cuda_ipc,cuda_copy,gdr_copy|g" "$LEVANTE_WRAPPER"


# Create or clean the experiment-specific directory.
ICON_EXP_DIR="${ICON_BUILD_DIR}/experiments/${EXPERIMENT_NAME}"
if [[ -d "$ICON_EXP_DIR" ]]; then
    echo "Cleaning existing experiment directory: ${ICON_EXP_DIR}"
    rm -rf "${ICON_EXP_DIR}"
fi
echo "Creating experiment directory: ${ICON_EXP_DIR}"
mkdir -p "${ICON_EXP_DIR}"


echo "Submitting: $ICON_RUN_SCRIPT"
sbatch "$ICON_RUN_SCRIPT"

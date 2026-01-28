#!/bin/bash
#SBATCH --job-name=testjob
#SBATCH --account=mh1498
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --output=slurm.%j.out
#SBATCH --exclusive
#SBATCH --mem-per-cpu=960
#SBATCH --time=00:40:00

### Define paths ###
# Use absolute paths since sbatch is executed from experiment directory
export EXPDIR=/work/mh1498/m301257/work/MEssE/experiment
export BASE_DIR=/work/mh1498/m301257/work/MEssE
export ICONDIR=$BASE_DIR/build/gcc

### ENV ###
env
set -xe

unset SLURM_EXPORT_ENV 
unset SLURM_MEM_PER_NODE
unset SBATCH_EXPORT


# limit stacksize ... adjust to your programs need
# and core file size
ulimit -c 0
ulimit -l unlimited

# Initialize module system for SLURM batch jobs
if [ -f /usr/share/Modules/init/bash ]; then
  source /usr/share/Modules/init/bash
fi

export SLURM_DIST_PLANESIZE="32"
export OMPI_MCA_btl="self"
export OMPI_MCA_coll="^ml,hcoll"
export OMPI_MCA_io="romio321"
export OMPI_MCA_osc="ucx"
export OMPI_MCA_pml="ucx"
export UCX_HANDLE_ERRORS="bt"
export UCX_TLS="shm,dc_mlx5,dc_x,self"
export UCX_UNIFIED_MODE="y"
export MALLOC_TRIM_THRESHOLD_="-1"
export OMPI_MCA_pml_ucx_opal_mem_hooks=1

module load eccodes

export ECCODES_DEFINITION_PATH=/pool/data/ICON/ICON_training/eccodes/definitions.edzw-2.27.0-1:$ECCODES_DEFINITION_PATH
export OMP_NUM_THREADS=1

# Use correct ICON binary path
export MODEL=$ICONDIR/bin/icon

# Source your own environment
source ${BASE_DIR}/environment/activate_levante_env

# Activate Python virtual environment (needed for comin plugin with torch)
if [ -f "${BASE_DIR}/environment/python/py_venv/bin/activate" ]; then
    source ${BASE_DIR}/environment/python/py_venv/bin/activate
    echo "Activated Python virtual environment"
    # Verify torch is available
    python -c "import torch; print('Torch version:', torch.__version__)" || echo "ERROR: Torch not found!"
    which python
fi

# Load Python packages if spack is available
if command -v spack >/dev/null 2>&1; then
    spack load py-mpi4py 2>/dev/null || echo "Warning: Could not load py-mpi4py"
    spack load py-cartopy 2>/dev/null || echo "Warning: Could not load py-cartopy"
fi

# Export Python path for MPI processes
export PYTHONPATH="${BASE_DIR}/environment/python/py_venv/lib/python3.9/site-packages:${PYTHONPATH}"

srun -l --cpu_bind=verbose --hint=nomultithread --distribution=block:cyclic $MODEL


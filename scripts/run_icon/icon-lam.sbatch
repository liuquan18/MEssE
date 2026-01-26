#!/bin/bash
#SBATCH --job-name=testjob
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --output=slurm.%j.out
#SBATCH --exclusive
#SBATCH --mem-per-cpu=960
#SBATCH --time=00:30:00

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
export MODEL=$ICONDIR/build/bin/icon

source /scratch/m/m300883/icon_exercise_comin/activate_levante_env

srun -l --cpu_bind=verbose --hint=nomultithread --distribution=block:cyclic $MODEL


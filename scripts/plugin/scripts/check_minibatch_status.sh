#!/bin/bash
# Quick status check for Mini-batch GNN job

JOB_ID=22285966
SLURM_OUT="/work/mh1498/m301257/work/MEssE/experiment/slurm.${JOB_ID}.out"

echo "=============================================="
echo "🚀 Mini-batch GNN Job Status"
echo "=============================================="
echo ""

# Job status
echo "📊 Job Information:"
echo "---"
squeue -j $JOB_ID 2>/dev/null || echo "Job completed or not found"
echo ""

# Count completed timesteps
echo "⏱️  Training Progress:"
echo "---"
if [ -f "$SLURM_OUT" ]; then
    completed=$(grep -c "Mini-batch GNN training completed" "$SLURM_OUT")
    echo "Completed timesteps: $completed"
    
    # Show latest losses
    echo ""
    echo "📈 Latest Training Results:"
    echo "---"
    grep -E "Final loss:|Average loss:" "$SLURM_OUT" | tail -6
else
    echo "Output file not found yet"
fi

echo ""
echo "💾 Output Files:"
echo "---"
echo "Loss logs:"
ls -1 /scratch/m/m301257/icon_exercise_comin/log_*.txt 2>/dev/null | wc -l | xargs echo "  Count:"
echo ""
echo "NetCDF outputs:"
ls -1 /work/mh1498/m301257/work/MEssE/experiment/NWP_LAM_DOM01_*.nc 2>/dev/null | wc -l | xargs echo "  Count:"

echo ""
echo "=============================================="
echo "💡 Monitoring Commands:"
echo "=============================================="
echo ""
echo "Real-time output:"
echo "  tail -f $SLURM_OUT | grep -E 'Batch|Loss|completed'"
echo ""
echo "Check all losses:"
echo "  grep 'Average loss' $SLURM_OUT"
echo ""
echo "View this summary:"
echo "  ./scripts/plugin/scripts/check_minibatch_status.sh"
echo ""

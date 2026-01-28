#!/bin/bash
# Monitor ICON job status

if [ -z "$1" ]; then
    # Get the latest job
    LATEST_JOB=$(squeue -u $USER -o "%.18i" -h | head -1)
    if [ -z "$LATEST_JOB" ]; then
        echo "No running jobs found. Checking recent jobs..."
        LATEST_JOB=$(sacct -u $USER --format=JobID -n -X | tail -1 | tr -d ' ')
    fi
    JOB_ID=$LATEST_JOB
else
    JOB_ID=$1
fi

if [ -z "$JOB_ID" ]; then
    echo "No job ID provided and no jobs found."
    echo "Usage: $0 [job_id]"
    exit 1
fi

echo "======================================"
echo "Monitoring Job: $JOB_ID"
echo "======================================"
echo ""

# Check job status
echo "Job Status:"
sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList -X

echo ""
echo "--------------------------------------"
echo "Recent output from SLURM log:"
echo "--------------------------------------"

SLURM_LOG="/work/mh1498/m301257/work/MEssE/experiment/slurm.${JOB_ID}.out"

if [ -f "$SLURM_LOG" ]; then
    tail -30 "$SLURM_LOG"
    echo ""
    echo "Full log: $SLURM_LOG"
else
    echo "SLURM output file not yet available: $SLURM_LOG"
    echo "Job may still be pending or just started."
fi

echo ""
echo "--------------------------------------"
echo "To monitor in real-time, run:"
echo "  tail -f $SLURM_LOG"
echo "--------------------------------------"

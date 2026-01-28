#!/bin/bash
# Quick fix script to update time limit and resubmit job

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/icon-lam.sbatch"

echo "=========================================="
echo "🔧 Fixing Job Time Limit"
echo "=========================================="
echo ""

# Check if sbatch file exists
if [ ! -f "$SBATCH_FILE" ]; then
    echo "❌ Error: sbatch file not found: $SBATCH_FILE"
    exit 1
fi

# Backup original file
BACKUP_FILE="${SBATCH_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$SBATCH_FILE" "$BACKUP_FILE"
echo "✅ Backup created: $BACKUP_FILE"

# Check current time limit
CURRENT_TIME=$(grep "^#SBATCH --time=" "$SBATCH_FILE" || echo "Not found")
echo "📋 Current time limit: $CURRENT_TIME"

# Ask user for new time limit
echo ""
echo "Suggested time limits:"
echo "  1) 01:00:00  (1 hour)   - For quick tests"
echo "  2) 04:00:00  (4 hours)  - For partial runs"
echo "  3) 08:00:00  (8 hours)  - For complete training (recommended)"
echo "  4) 12:00:00  (12 hours) - For very long runs"
echo "  5) Custom"
echo ""
read -p "Select option [1-5] (default: 3): " CHOICE

case ${CHOICE:-3} in
    1)
        NEW_TIME="01:00:00"
        ;;
    2)
        NEW_TIME="04:00:00"
        ;;
    3)
        NEW_TIME="08:00:00"
        ;;
    4)
        NEW_TIME="12:00:00"
        ;;
    5)
        read -p "Enter custom time (HH:MM:SS): " NEW_TIME
        ;;
    *)
        echo "Invalid option, using default: 08:00:00"
        NEW_TIME="08:00:00"
        ;;
esac

# Update time limit
sed -i "s/^#SBATCH --time=.*/#SBATCH --time=$NEW_TIME/" "$SBATCH_FILE"

# Verify change
NEW_TIME_CHECK=$(grep "^#SBATCH --time=" "$SBATCH_FILE")
echo ""
echo "✅ Updated time limit: $NEW_TIME_CHECK"

# Show diff
echo ""
echo "📝 Changes made:"
echo "----------------------------------------"
diff "$BACKUP_FILE" "$SBATCH_FILE" || true
echo "----------------------------------------"

# Ask if user wants to submit
echo ""
read -p "🚀 Submit job now? [Y/n]: " SUBMIT

if [[ ${SUBMIT,,} =~ ^(y|yes|)$ ]]; then
    echo ""
    echo "Submitting job..."
    cd "$SCRIPT_DIR"
    JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $NF}')
    
    if [ -n "$JOB_ID" ]; then
        echo ""
        echo "=========================================="
        echo "✅ Job submitted successfully!"
        echo "=========================================="
        echo "Job ID: $JOB_ID"
        echo "Time limit: $NEW_TIME"
        echo ""
        echo "📊 Monitor job status:"
        echo "   squeue -u \$USER"
        echo "   squeue -j $JOB_ID"
        echo ""
        echo "📁 Output file will be:"
        echo "   $SCRIPT_DIR/slurm.$JOB_ID.out"
        echo ""
        echo "📈 View loss curve after job starts:"
        echo "   cd /work/mh1498/m301257/work/MEssE/scripts/run_icon"
        echo "   ./quick_plot_loss.sh $JOB_ID"
        echo ""
        echo "🌐 Start web monitor:"
        echo "   cd /work/mh1498/m301257/work/MEssE/scripts/plugin/monitor"
        echo "   ./start_monitor.sh"
        echo "=========================================="
    else
        echo "❌ Failed to submit job"
        exit 1
    fi
else
    echo ""
    echo "Job not submitted. You can submit manually with:"
    echo "   cd $SCRIPT_DIR"
    echo "   sbatch icon-lam.sbatch"
fi

echo ""
echo "✅ Done!"

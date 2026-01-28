#!/bin/bash
# 快速查看训练状态 - 一键命令

LOG_DIR="/scratch/m/m301257/icon_exercise_comin"
EXPERIMENT_DIR="/work/mh1498/m301257/work/MEssE/experiment"

# 获取Job信息
JOB_ID=$(squeue -u $USER -o "%.18i %.2t %.10M" -h | head -1)
if [ -n "$JOB_ID" ]; then
    echo "🟢 Job: $JOB_ID"
else
    echo "⚪ No running job"
fi

# 统计完成的timesteps
SLURM_FILE=$(ls -t ${EXPERIMENT_DIR}/slurm.*.out 2>/dev/null | head -1)
if [ -f "$SLURM_FILE" ]; then
    COMPLETED=$(grep -c "✓ Mini-batch GNN training completed" "$SLURM_FILE")
    echo "✅ Timesteps完成: $COMPLETED"
    
    # 获取最新Loss
    LATEST_LOSS=$(grep "Average loss:" "$SLURM_FILE" | tail -1 | awk '{print $NF}')
    if [ -n "$LATEST_LOSS" ]; then
        echo "📉 最新Loss: $LATEST_LOSS"
    fi
fi

# Loss日志统计
LOG_COUNT=$(ls -1 ${LOG_DIR}/log_*.txt 2>/dev/null | wc -l)
echo "📁 Loss日志: $LOG_COUNT 个文件"

# 显示最新Loss趋势
echo ""
echo "最近Loss趋势 (最后5个值):"
if [ -f "$SLURM_FILE" ]; then
    grep "Average loss:" "$SLURM_FILE" | tail -5 | awk '{printf "  %s\n", $NF}'
fi

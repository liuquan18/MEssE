#!/bin/bash
# 实时监控训练 - 类似PyTorch tqdm风格

EXPERIMENT_DIR="/work/mh1498/m301257/work/MEssE/experiment"
LOG_DIR="/scratch/m/m301257/icon_exercise_comin"

# 获取最新Job ID
JOB_ID=$(squeue -u $USER -o "%.18i" -h | head -1)
if [ -z "$JOB_ID" ]; then
    SLURM_FILE=$(ls -t ${EXPERIMENT_DIR}/slurm.*.out 2>/dev/null | head -1)
    if [ -z "$SLURM_FILE" ]; then
        echo "❌ 没有找到运行中的作业或输出文件"
        exit 1
    fi
    JOB_ID=$(basename "$SLURM_FILE" | sed 's/slurm\.\([0-9]*\)\.out/\1/')
else
    SLURM_FILE="${EXPERIMENT_DIR}/slurm.${JOB_ID}.out"
fi

echo "监控 Job $JOB_ID..."
echo "按 Ctrl+C 停止监控"
echo ""

# 实时跟踪输出
tail -f "$SLURM_FILE" 2>/dev/null | grep --line-buffered -E "(🚀|📦|Batch|Loss:|✓|Average|Mini-batch GNN)" | while read line; do
    # 高亮显示重要信息
    if [[ "$line" =~ "Loss:" ]]; then
        echo -e "\033[0;36m$line\033[0m"  # 青色
    elif [[ "$line" =~ "✓" ]]; then
        echo -e "\033[0;32m$line\033[0m"  # 绿色
    elif [[ "$line" =~ "🚀" ]]; then
        echo -e "\033[1;35m$line\033[0m"  # 紫色加粗
    else
        echo "$line"
    fi
done

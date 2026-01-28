#!/bin/bash
# 实时监控训练进度 - 简单终端版本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="/work/mh1498/m301257/work/MEssE/experiment"
LOG_DIR="/scratch/m/m301257/icon_exercise_comin"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 清屏
clear

echo -e "${PURPLE}${BOLD}"
echo "════════════════════════════════════════════════════════════════"
echo "          MEssE V.01 - GNN训练实时监控（终端版）"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# 获取最新的Job ID
get_latest_job() {
    squeue -u $USER -o "%.18i" -h | head -1
}

# 获取最新的SLURM输出文件
get_latest_slurm() {
    ls -t ${EXPERIMENT_DIR}/slurm.*.out 2>/dev/null | head -1
}

JOB_ID=$(get_latest_job)
SLURM_FILE=$(get_latest_slurm)

if [ -z "$JOB_ID" ]; then
    echo -e "${YELLOW}⚠️  没有运行中的作业${NC}"
    if [ -n "$SLURM_FILE" ]; then
        echo -e "${CYAN}📁 使用最新的输出文件: $(basename $SLURM_FILE)${NC}"
        JOB_ID=$(basename "$SLURM_FILE" | sed 's/slurm\.\([0-9]*\)\.out/\1/')
    else
        echo -e "${RED}❌ 没有找到任何输出文件${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ 运行中的作业: $JOB_ID${NC}"
    SLURM_FILE="${EXPERIMENT_DIR}/slurm.${JOB_ID}.out"
fi

echo ""

# 显示作业信息
echo -e "${CYAN}${BOLD}📊 作业信息${NC}"
echo "─────────────────────────────────────────────────────────────"
if [ -n "$(squeue -j $JOB_ID 2>/dev/null)" ]; then
    squeue -j $JOB_ID -o "Job ID: %.18i | Status: %.2t | Time: %.10M | Nodes: %.6D %R"
else
    echo -e "${YELLOW}Job $JOB_ID 已完成或不在队列中${NC}"
fi
echo ""

# 统计训练进度
echo -e "${CYAN}${BOLD}📈 训练进度${NC}"
echo "─────────────────────────────────────────────────────────────"

if [ -f "$SLURM_FILE" ]; then
    # 统计完成的timesteps
    COMPLETED=$(grep -c "✓ Mini-batch GNN training completed" "$SLURM_FILE" 2>/dev/null || echo "0")
    STARTED=$(grep -c "🚀 Mini-batch GNN Training on" "$SLURM_FILE" 2>/dev/null || echo "0")
    
    echo -e "✓ 完成的时间步: ${GREEN}${BOLD}$COMPLETED${NC}"
    echo -e "🔄 启动的时间步: ${YELLOW}$STARTED${NC}"
    echo -e "📦 每时间步Batch数: ${CYAN}8${NC}"
    echo -e "💾 总Batch训练次数: ${PURPLE}$((COMPLETED * 8))${NC}"
fi
echo ""

# 显示Loss日志统计
echo -e "${CYAN}${BOLD}📁 Loss日志文件${NC}"
echo "─────────────────────────────────────────────────────────────"
if [ -d "$LOG_DIR" ]; then
    LOG_COUNT=$(ls -1 ${LOG_DIR}/log_*.txt 2>/dev/null | wc -l)
    echo -e "📝 日志文件数: ${GREEN}$LOG_COUNT${NC}"
    
    if [ $LOG_COUNT -gt 0 ]; then
        LATEST_LOG=$(ls -t ${LOG_DIR}/log_*.txt 2>/dev/null | head -1)
        LATEST_TIME=$(basename "$LATEST_LOG" | sed 's/log_\(.*\)\.txt/\1/')
        echo -e "🕐 最新日志: ${CYAN}$LATEST_TIME${NC}"
        
        # 显示最新的几个Loss值
        echo ""
        echo -e "${YELLOW}最近10个Loss值:${NC}"
        tail -10 "$LATEST_LOG" | awk '{printf "  %.6e\n", $1}'
    fi
fi
echo ""

# 显示最近的训练输出
echo -e "${CYAN}${BOLD}📄 最近的训练输出 (最后20行)${NC}"
echo "─────────────────────────────────────────────────────────────"
if [ -f "$SLURM_FILE" ]; then
    tail -20 "$SLURM_FILE" | grep -E "(🚀|📦|Loss:|✓|Average loss)" | tail -15
fi
echo ""

echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}💡 提示:${NC}"
echo "  • 实时查看完整输出: ${YELLOW}tail -f $SLURM_FILE${NC}"
echo "  • 查看Loss日志: ${YELLOW}tail -f ${LOG_DIR}/log_*.txt${NC}"
echo "  • 生成Loss图表: ${YELLOW}cd $SCRIPT_DIR && ./quick_plot_loss.sh $JOB_ID${NC}"
echo "  • 重新运行监控: ${YELLOW}$0${NC}"
echo ""

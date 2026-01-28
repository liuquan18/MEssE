# 🚀 GNN训练监控工具 - 快速使用指南

## 三个简单命令，实时监控训练

---

## 1️⃣ 快速状态查看（推荐）

```bash
./status.sh
```

**功能**：一键查看当前训练状态
- ✅ Job运行状态和时间
- ✅ 完成的Timesteps数量
- ✅ 最新Loss值
- ✅ Loss日志统计
- ✅ 最近5个Loss趋势

**示例输出**：
```
🟢 Job:           22290687  R      10:43
✅ Timesteps完成: 5
📉 最新Loss: 2.053989e+00
📁 Loss日志: 1462 个文件

最近Loss趋势 (最后5个值):
  9.572431e-03
  1.672102e+01
  7.073073e+00
  3.450150e+00
  2.053989e+00
```

**使用场景**：快速检查训练进展

---

## 2️⃣ 实时监控（类似tqdm）

```bash
./watch_training.sh
```

**功能**：实时流式显示训练过程
- 🔄 自动跟踪最新输出
- 🎨 颜色高亮重要信息
- 📊 显示每个Batch的Loss
- ⌨️ Ctrl+C 停止监控

**示例输出**：
```
监控 Job 22290687...
按 Ctrl+C 停止监控

  0: 🚀 Mini-batch GNN Training on 37488 nodes
  0:   📦 Batch 1/8: nodes [0:5000]
  0:      Loss: 2.705625e-02
  0:   📦 Batch 2/8: nodes [5000:10000]
  0:      Loss: 1.463016e-02
  ...
  0: ✓ Mini-batch GNN training completed
  0:   Average loss: 9.572431e-03
```

**使用场景**：想看到详细的训练过程，类似PyTorch训练时的实时输出

---

## 3️⃣ 详细监控报告

```bash
./monitor_training.sh
```

**功能**：完整的训练状态报告
- 📊 作业详细信息
- 📈 训练进度统计
- 📁 Loss日志文件信息
- 📄 最近20行训练输出
- 💡 实用命令提示

**示例输出**：
```
════════════════════════════════════════════════════════════════
          MEssE V.01 - GNN训练实时监控（终端版）
════════════════════════════════════════════════════════════════

✅ 运行中的作业: 22290687

📊 作业信息
─────────────────────────────────────────────────────────────
Job ID: 22290687 | Status: R | Time: 10:43 | Nodes: 4 l[40166-40169]

📈 训练进度
─────────────────────────────────────────────────────────────
✓ 完成的时间步: 5
🔄 启动的时间步: 6
📦 每时间步Batch数: 8
💾 总Batch训练次数: 40

📁 Loss日志文件
─────────────────────────────────────────────────────────────
📝 日志文件数: 1462
🕐 最新日志: 2021-07-14 00-20-00

最近10个Loss值:
  1.234567e-02
  ...

📄 最近的训练输出 (最后20行)
─────────────────────────────────────────────────────────────
  0: 🚀 Mini-batch GNN Training on 37488 nodes
  ...
```

**使用场景**：全面了解训练状态，生成报告

---

## 🎯 推荐工作流程

### 快速检查
```bash
./status.sh
```

### 持续监控
```bash
./watch_training.sh
# 保持终端打开，实时看到训练进度
```

### 详细分析
```bash
./monitor_training.sh
```

### 生成Loss图表
```bash
./quick_plot_loss.sh 22290687
```

---

## �� 其他有用命令

### 查看队列
```bash
squeue -u $USER
```

### 实时查看完整日志
```bash
tail -f /work/mh1498/m301257/work/MEssE/experiment/slurm.22290687.out
```

### 查看Loss日志
```bash
ls -lt /scratch/m/m301257/icon_exercise_comin/log_*.txt | head
tail /scratch/m/m301257/icon_exercise_comin/log_*.txt
```

### 取消作业
```bash
scancel 22290687
```

---

## 💡 使用技巧

1. **开两个终端**：
   - 终端1：运行 `./watch_training.sh` 实时监控
   - 终端2：定期运行 `./status.sh` 快速检查

2. **定时检查**：
   ```bash
   watch -n 10 ./status.sh  # 每10秒更新一次
   ```

3. **保存监控记录**：
   ```bash
   ./monitor_training.sh > training_report_$(date +%Y%m%d_%H%M%S).txt
   ```

4. **后台监控**：
   ```bash
   # 每分钟记录一次状态
   while true; do 
       date >> monitor.log
       ./status.sh >> monitor.log
       echo "" >> monitor.log
       sleep 60
   done &
   ```

---

## 🎨 颜色说明

- 🟢 **绿色**：完成/成功
- 🔵 **青色**：Loss值
- 🟣 **紫色**：训练阶段标记
- 🟡 **黄色**：警告/提示

---

## ❓ 常见问题

**Q: 显示"没有找到运行中的作业"？**
A: 作业可能已完成或被取消，脚本会自动使用最新的输出文件

**Q: Loss值一直不更新？**
A: 检查作业是否还在运行：`squeue -u $USER`

**Q: 想看更详细的输出？**
A: 直接查看SLURM输出：`tail -f /work/mh1498/m301257/work/MEssE/experiment/slurm.22290687.out`

---

## 🚀 快速开始

```bash
cd /work/mh1498/m301257/work/MEssE/scripts/run_icon

# 快速查看状态
./status.sh

# 实时监控训练
./watch_training.sh
```

就这么简单！享受您的GNN训练监控体验！ 🎉

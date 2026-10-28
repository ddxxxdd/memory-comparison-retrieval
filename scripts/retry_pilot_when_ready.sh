#!/usr/bin/env bash
# 等待 LLM/向量服务恢复后，自动重跑 pilot_150 组合来源实验。
# 有界重试：最多 18 次，每次间隔 300 秒（约 1.5 小时窗口）。
set -u
cd "$(dirname "$0")/.."
P=./.venv/Scripts/python.exe
RUN=combined-source-pilot-150old-12q-r3
LOG="runs/experiments/${RUN}.stdout.log"
mkdir -p runs/experiments

echo "[$(date '+%F %H:%M:%S')] 重试看守启动，等待模型服务恢复..." | tee "$LOG"

for attempt in $(seq 1 18); do
  echo "[$(date '+%F %H:%M:%S')] 第 $attempt 次连通性检查" | tee -a "$LOG"
  if "$P" check_api_connection.py --timeout 40 >/dev/null 2>&1; then
    echo "[$(date '+%F %H:%M:%S')] 服务已恢复，启动实验" | tee -a "$LOG"
    if "$P" -X utf8 scripts/run_experiment.py \
        --dataset data/combined_sources_pilot_150_v1 --limit 12 --workers 4 \
        --arms proposed --run-name "$RUN" 2>&1 | tee -a "$LOG"; then
      echo "[$(date '+%F %H:%M:%S')] EXPERIMENT_DONE" | tee -a "$LOG"
      exit 0
    else
      echo "[$(date '+%F %H:%M:%S')] 实验失败（可能仍为服务波动），进入下一轮等待" | tee -a "$LOG"
    fi
  else
    echo "[$(date '+%F %H:%M:%S')] 服务未恢复（LLM 或向量不可用），等待 300 秒" | tee -a "$LOG"
  fi
  sleep 300
done

echo "[$(date '+%F %H:%M:%S')] 重试窗口耗尽，未成功。请检查模型服务。" | tee -a "$LOG"
exit 1
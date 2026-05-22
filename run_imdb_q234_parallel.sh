#!/bin/bash
set -euo pipefail

# Parallel runner for q2, q3, q4, q5 on imdb_cnn
# Runs all 4 qtypes simultaneously with 20 question IDs each
# 
# Usage:
#   ./run_imdb_q234_parallel.sh                    # Run all qtypes, all qids
#   QUESTION_IDS="0 1 2 5" ./run_imdb_q234_parallel.sh  # Run specific qids
#   Q_TYPES="2 3" ./run_imdb_q234_parallel.sh      # Run only q2, q3
#   QUESTION_IDS="0 1" Q_TYPES="2" ./run_imdb_q234_parallel.sh  # Subset both

DATASET="${DATASET:-imdb_cnn}"
QUESTION_IDS="${QUESTION_IDS:-all}"
MODE="${MODE:-test}"
VLM="${VLM:-tinker/Qwen3.6-35B-A3B}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
MODELS_DIR="${MODELS_DIR:-/Users/rraghavkaushik/xai-agent/models_to_read}"
OUTPUT_BASE="${OUTPUT_BASE:-./outputs_parallel_imdb_q2345}"
Q_TYPES="${Q_TYPES:-2 3 4 5}"

# Host capacity snapshot
echo "============================================================"
echo "HOST CAPACITY SNAPSHOT"
echo "============================================================"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
MEM=$(sysctl -n hw.memsize 2>/dev/null || echo "unknown")
DISK_FREE=$(df . 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1048576}')
LOAD=$(uptime | awk -F'load average:' '{print $2}' | xargs)

echo "CPU cores      : ${CORES}"
echo "Memory (bytes) : ${MEM}"
echo "Disk free (GiB): ${DISK_FREE}"
echo "Load averages  : ${LOAD}"
echo ""

# Warn if low resources
if (( $(echo "$DISK_FREE < 2.0" | bc -l) )); then
  echo "⚠️  WARNING: Disk space < 2 GiB; parallel jobs may fail"
  echo ""
fi

# Ensure output directories exist
mkdir -p "${OUTPUT_BASE}/logs"

echo "Plan: run qtypes [${Q_TYPES}] in parallel as separate processes."
echo "============================================================"

# Trap for cleanup on exit
cleanup() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    echo "Script interrupted or failed (exit code ${exit_code}). Terminating child processes..."
    jobs -p | xargs -r kill 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Launch parallel jobs - store PIDs and qtype indices
pids=()
qtypes_array=()
results=()

for q in ${Q_TYPES}; do
  LOG="${OUTPUT_BASE}/logs/q${q}.log"
  OUT_DIR="${OUTPUT_BASE}/q${q}"
  
  # Log file redirection + background
  python run_pipeline_batch.py \
    --datasets "${DATASET}" \
    --q_types "${q}" \
    --question_ids "${QUESTION_IDS}" \
    --mode "${MODE}" \
    --vlm "${VLM}" \
    --dataset_dir "${DATASET_DIR}" \
    --models_dir "${MODELS_DIR}" \
    --output_dir "${OUT_DIR}" \
    >"${LOG}" 2>&1 &
  
  pid=$!
  pids+=("$pid")
  qtypes_array+=("$q")
  echo "[START] q${q} -> ${LOG}"
done

echo ""

# Wait for all processes to complete and collect results
for i in "${!pids[@]}"; do
  q_idx=${qtypes_array[$i]}
  pid=${pids[$i]}
  
  if wait "$pid" 2>/dev/null; then
    results+=("[PASS] q${q_idx}")
    echo "[PASS] q${q_idx}"
  else
    results+=("[FAIL] q${q_idx} (see ${OUTPUT_BASE}/logs/q${q_idx}.log)")
    echo "[FAIL] q${q_idx} (see ${OUTPUT_BASE}/logs/q${q_idx}.log)"
  fi
done

echo ""
echo "Logs: ${OUTPUT_BASE}/logs/"

# Aggregate summary
PASS_COUNT=$(echo "${results[@]}" | grep -c "PASS" || true)
FAIL_COUNT=$(echo "${results[@]}" | grep -c "FAIL" || true)

echo ""
echo "============================================================"
echo "SUMMARY: ${PASS_COUNT} PASS, ${FAIL_COUNT} FAIL"
echo "============================================================"

if (( FAIL_COUNT > 0 )); then
  exit 1
fi

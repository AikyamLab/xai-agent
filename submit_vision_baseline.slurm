#!/bin/bash
#SBATCH --job-name=vision_baseline_q6
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=0-79  # 80 jobs total (4 datasets × 20 qids)

# Vision Baseline Q6 Submission Script
# Usage:
#   sbatch submit_vision_baseline.slurm                    # Q6 all 80 jobs
#   sbatch --array=0-19 submit_vision_baseline.slurm       # Q6 dataset 1 only (20 jobs)
#   sbatch --time=08:00:00 submit_vision_baseline.slurm    # Increase timeout to 8 hours

set -e

# Config
REPO_ROOT="/path/to/xai-agent"  # Update this path
Q_TYPE=6
VLM="tinker/qwen3.6-35B-A3B"
OUTPUT_DIR="outputs_vision_baseline_q6_parallel"
LOG_DIR="logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

# Map array job index to dataset and qid
# 0-19: dataset 1, qids 0-19
# 20-39: dataset 2, qids 0-19
# 40-59: dataset 3, qids 0-19
# 60-79: dataset 4, qids 0-19
DATASET_NUM=$(( ($SLURM_ARRAY_TASK_ID / 20) + 1 ))
QID=$(( $SLURM_ARRAY_TASK_ID % 20 ))

DATASET="cub_densenet"
if [ $DATASET_NUM -eq 1 ]; then DATASET="cub_densenet"; fi
if [ $DATASET_NUM -eq 2 ]; then DATASET="cub_resnet"; fi
if [ $DATASET_NUM -eq 3 ]; then DATASET="stl10_densenet"; fi
if [ $DATASET_NUM -eq 4 ]; then DATASET="stl10_resnet"; fi

echo "=== Vision Baseline Q6 Execution ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Q Type: $Q_TYPE"
echo "QID: $QID"
echo "VLM: $VLM"
echo "======================================="

# Run single job
python run_pipeline_batch.py \
  --q_types $Q_TYPE \
  --question_ids $QID \
  --dataset $DATASET \
  --dataset_dir "dataset_full" \
  --vlm "$VLM" \
  --output_dir "$OUTPUT_DIR" \
  --no-improvement \
  --no-sf

echo "✓ Job complete: $DATASET Q$Q_TYPE QID=$QID"

#!/usr/bin/env bash
# =============================================================================
# run_baseline_agents.sh
#
# Convenience wrapper around run_baseline_agent_batch.py for running baseline
# agents (naive / cot / react / tot) using the full 3-agent pipeline
# with injected reasoning preambles.
#
# Usage:
#   bash run_baseline_agents.sh --baselines "naive cot react tot" \
#       --modality tabular --q_types "1 2 3" --question_ids "0 1 2 3 4" \
#       --vlm "tinker/Qwen3-VL-30B-A3B-Instruct"
#
#   bash run_baseline_agents.sh --baselines "cot" \
#       --datasets "adult_tabnn cancer_tabnn" --q_types "1" \
#       --question_ids "0-9" --vlm "tinker/Qwen3-VL-30B-A3B-Instruct"
# =============================================================================
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source .env for API keys
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    source "${SCRIPT_DIR}/.env"
fi

# Disable Tinker telemetry
export TINKER_TELEMETRY=0

# ── Defaults ─────────────────────────────────────────────────────────────────
BASELINES="naive"
MODALITY=""
DATASETS=""
Q_TYPES="1"
QUESTION_IDS="0"
VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_baseline_agents"
DATASET_DIR="${SCRIPT_DIR}/dataset"
MODELS_DIR="${SCRIPT_DIR}/models_to_read"
MODE="test"
TINKER_CHECKPOINT=""
TINKER_LORA_RANK=16
DRY_RUN=false

# ── Help ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Baseline Agent Runner
Runs the full 3-agent pipeline with reasoning preambles injected into prompts.

Options:
    --baselines "b1 b2 ..."  Baselines to run: naive, cot, react, tot (default: naive)
    --modality MOD           Modality: vision, text, tabular, all
    --datasets "d1 d2 ..."   Specific datasets
    --q_types "1 2 3 ..."    Question types (default: 1)
    --question_ids "0-4"     Question IDs (default: 0)
    --vlm MODEL              VLM model ID (default: Qwen/Qwen3-VL-8B-Instruct)
    --output_dir DIR         Output directory (default: outputs_baseline_agents)
    --dataset_dir DIR        Dataset directory (default: dataset)
    --models_dir DIR         Models directory (default: models_to_read)
    --mode MODE              Dataset split: train or test (default: test)
    --tinker_checkpoint CKP  Tinker checkpoint for fine-tuned model eval
    --tinker_lora_rank N     LoRA rank (default: 16)
    --dry_run                Print commands without executing
    -h, --help               Show this help

Baselines:
    naive  — Standard pipeline, no reasoning scaffolding (--no-improvement --no-sf)
    cot    — Chain-of-Thought preamble prepended to proposer & actor prompts
    react  — ReAct (Reasoning + Acting) preamble prepended
    tot    — Tree of Thought preamble prepended

Examples:
    # Run all 4 baselines on tabular Q1-Q5, 10 questions each
    bash $(basename "$0") --baselines "naive cot react tot" \\
        --modality tabular --q_types "1 2 3 4 5" --question_ids "0-9" \\
        --vlm "tinker/Qwen3-VL-30B-A3B-Instruct"

    # Dry run for CoT on adult_tabnn Q1
    bash $(basename "$0") --baselines "cot" --datasets "adult_tabnn" \\
        --q_types "1" --question_ids "0-4" --dry_run
EOF
    exit 0
}

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baselines)    BASELINES="$2"; shift 2 ;;
        --modality)     MODALITY="$2"; shift 2 ;;
        --datasets)     DATASETS="$2"; shift 2 ;;
        --q_types)      Q_TYPES="$2"; shift 2 ;;
        --question_ids) QUESTION_IDS="$2"; shift 2 ;;
        --vlm)          VLM_MODEL="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --dataset_dir)  DATASET_DIR="$2"; shift 2 ;;
        --models_dir)   MODELS_DIR="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --tinker_checkpoint) TINKER_CHECKPOINT="$2"; shift 2 ;;
        --tinker_lora_rank)  TINKER_LORA_RANK="$2"; shift 2 ;;
        --dry_run)      DRY_RUN=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

# ── Build python command ─────────────────────────────────────────────────────
CMD=(python3 run_baseline_agent_batch.py)

# Baselines
CMD+=(--baseline $BASELINES)

# Dataset selection
if [[ -n "$DATASETS" ]]; then
    CMD+=(--datasets $DATASETS)
elif [[ -n "$MODALITY" ]]; then
    CMD+=(--modality $MODALITY)
else
    echo "Error: Must specify --modality or --datasets"
    exit 1
fi

# Question types and IDs
CMD+=(--q_types $Q_TYPES)
CMD+=(--question_ids "$QUESTION_IDS")

# Paths and model
CMD+=(--vlm "$VLM_MODEL")
CMD+=(--output_dir "$OUTPUT_DIR")
CMD+=(--dataset_dir "$DATASET_DIR")
CMD+=(--models_dir "$MODELS_DIR")
CMD+=(--mode "$MODE")

# Optional tinker checkpoint
if [[ -n "$TINKER_CHECKPOINT" ]]; then
    CMD+=(--tinker_checkpoint "$TINKER_CHECKPOINT")
    CMD+=(--tinker_lora_rank "$TINKER_LORA_RANK")
fi

# Dry run
if [[ "$DRY_RUN" == true ]]; then
    CMD+=(--dry_run)
fi

# ── Execute ──────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "BASELINE AGENT RUNNER"
echo "======================================================================"
echo "  Baselines:    $BASELINES"
echo "  VLM:          $VLM_MODEL"
echo "  Output:       $OUTPUT_DIR"
echo "  Command:      ${CMD[*]}"
echo "======================================================================"

exec "${CMD[@]}"

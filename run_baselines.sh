#!/usr/bin/env bash
#
# run_baselines.sh — Canonical baseline entrypoint.
#
# This script now routes baseline runs through run_baseline_agent_batch.py so all
# baseline types share the standard pipeline backbone with:
#   --no-improvement --no-sf
#
# Legacy custom-agent path:
#   run_baseline.py remains available for historical/explicit experiments.
#
# Environment variables (override defaults):
#   BASELINES        Space-separated list: naive cot react tot all    (default: "naive cot react tot")
#   DATASETS         Space-separated list of dataset keys             (default: stl10_resnet stl10_densenet)
#   MODALITY         Optional: vision text tabular all                (default: unset; DATASETS is used)
#   Q_TYPES          Space-separated list of question type IDs        (default: "1 2 3")
#   Q_IDS            Question IDs/ranges string                       (default: "0 1 2")
#   VLM              VLM model ID                                     (default: Qwen3.6-35B-A3B)
#   MODE             train or test                                    (default: test)
#   OUTPUT_ROOT      Root output directory                            (default: ./outputs_baselines)
#   DATASET_DIR      Dataset directory                                (default: ./dataset)
#   MODELS_DIR       Models directory                                 (default: ./models_to_read)
#   NO_EVAL          Set to 1 to skip evaluation                      (default: 0)
#   USE_TEST_VARIANT Set to 1 to prefer *_test JSON files             (default: 0)
#   TINKER_CHECKPOINT Optional checkpoint path                        (default: empty)
#   TINKER_LORA_RANK LoRA rank for checkpoint loading                 (default: 16)
#   DRY_RUN          Set to 1 to print commands without running       (default: 0)
#   VERBOSE          Set to 1 for verbose batch logging               (default: 0)
#
# Backward-compatibility note:
#   TOT_BRANCHES / REACT_MAX_ITER / MAX_PARALLEL are legacy knobs from the
#   old custom-agent route and are ignored in this canonical path.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source .env for API keys (if present)
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    source "${SCRIPT_DIR}/.env"
fi

export TINKER_TELEMETRY=0

BASELINES="${BASELINES:-naive cot react tot}"
DATASETS="${DATASETS:-stl10_resnet stl10_densenet}"
MODALITY="${MODALITY:-}"
Q_TYPES="${Q_TYPES:-1 2 3}"
Q_IDS="${Q_IDS:-0 1 2}"
VLM="${VLM:-Qwen3.6-35B-A3B}"
MODE="${MODE:-test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs_baselines}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
MODELS_DIR="${MODELS_DIR:-./models_to_read}"
NO_EVAL="${NO_EVAL:-0}"
USE_TEST_VARIANT="${USE_TEST_VARIANT:-0}"
TINKER_CHECKPOINT="${TINKER_CHECKPOINT:-}"
TINKER_LORA_RANK="${TINKER_LORA_RANK:-16}"
DRY_RUN="${DRY_RUN:-0}"
VERBOSE="${VERBOSE:-0}"

TOT_BRANCHES="${TOT_BRANCHES:-3}"
REACT_MAX_ITER="${REACT_MAX_ITER:-6}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARNING: $*" >&2; }
die() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; exit 1; }

if [[ "$TOT_BRANCHES" != "3" ]]; then
    warn "TOT_BRANCHES is ignored in canonical baseline path (value: ${TOT_BRANCHES})."
fi
if [[ "$REACT_MAX_ITER" != "6" ]]; then
    warn "REACT_MAX_ITER is ignored in canonical baseline path (value: ${REACT_MAX_ITER})."
fi
if [[ "$MAX_PARALLEL" != "0" ]]; then
    warn "MAX_PARALLEL is ignored in canonical baseline path (value: ${MAX_PARALLEL})."
fi

read -r -a baseline_tokens <<< "$BASELINES"
read -r -a dataset_tokens <<< "$DATASETS"
read -r -a modality_tokens <<< "$MODALITY"
read -r -a qtype_tokens <<< "$Q_TYPES"

[[ ${#baseline_tokens[@]} -gt 0 ]] || die "BASELINES must not be empty."
[[ ${#qtype_tokens[@]} -gt 0 ]] || die "Q_TYPES must not be empty."
if [[ ${#dataset_tokens[@]} -eq 0 && ${#modality_tokens[@]} -eq 0 ]]; then
    die "Provide DATASETS or MODALITY."
fi

expanded_baselines=()
add_unique_baseline() {
    local candidate="$1"
    local existing
    for existing in "${expanded_baselines[@]}"; do
        [[ "$existing" == "$candidate" ]] && return 0
    done
    expanded_baselines+=("$candidate")
}

for b in "${baseline_tokens[@]}"; do
    case "$b" in
        all)
            add_unique_baseline "naive"
            add_unique_baseline "cot"
            add_unique_baseline "react"
            add_unique_baseline "tot"
            ;;
        naive|cot|react|tot)
            add_unique_baseline "$b"
            ;;
        *)
            die "Invalid baseline '${b}'. Allowed: naive cot react tot all"
            ;;
    esac
done

CMD=(python3 run_baseline_agent_batch.py)
CMD+=(--baseline "${expanded_baselines[@]}")

if [[ ${#dataset_tokens[@]} -gt 0 ]]; then
    CMD+=(--datasets "${dataset_tokens[@]}")
else
    CMD+=(--modality "${modality_tokens[@]}")
fi

CMD+=(--q_types "${qtype_tokens[@]}")
CMD+=(--question_ids "$Q_IDS")
CMD+=(--vlm "$VLM")
CMD+=(--output_dir "$OUTPUT_ROOT")
CMD+=(--dataset_dir "$DATASET_DIR")
CMD+=(--models_dir "$MODELS_DIR")
CMD+=(--mode "$MODE")

if [[ "$NO_EVAL" == "1" ]]; then
    CMD+=(--no-eval)
fi
if [[ "$USE_TEST_VARIANT" == "1" ]]; then
    CMD+=(--use_test_variant)
fi
if [[ -n "$TINKER_CHECKPOINT" ]]; then
    CMD+=(--tinker_checkpoint "$TINKER_CHECKPOINT")
    CMD+=(--tinker_lora_rank "$TINKER_LORA_RANK")
fi
if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=(--dry_run)
fi
if [[ "$VERBOSE" == "1" ]]; then
    CMD+=(--verbose)
fi

log "============================================"
log "CANONICAL BASELINE BATCH RUNNER"
log "============================================"
log "Baselines:  ${expanded_baselines[*]}"
if [[ ${#dataset_tokens[@]} -gt 0 ]]; then
    log "Datasets:   ${dataset_tokens[*]}"
else
    log "Modalities: ${modality_tokens[*]}"
fi
log "Q types:    ${Q_TYPES}"
log "Q IDs:      ${Q_IDS}"
log "VLM:        ${VLM}"
log "Mode:       ${MODE}"
log "Output:     ${OUTPUT_ROOT}"
log "No eval:    ${NO_EVAL}"
log "Dry run:    ${DRY_RUN}"
printf '[%s] Command:    ' "$(date '+%H:%M:%S')"
printf '%q ' "${CMD[@]}"
echo
log "============================================"

exec "${CMD[@]}"

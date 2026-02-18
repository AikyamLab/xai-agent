#!/usr/bin/env bash
#
# run_baselines.sh — Run baseline agents across datasets, question types, and question IDs.
#
# Usage:
#   # All baselines on all vision datasets, Q1-Q3, first 3 questions:
#   bash run_baselines.sh
#
#   # Specific baselines / datasets / questions:
#   BASELINES="cot react" DATASETS="stl10_resnet" Q_TYPES="1 2" Q_IDS="0 1" bash run_baselines.sh
#
#   # Run all baselines on a single dataset-question:
#   BASELINES="all" DATASETS="imdb_cnn" Q_TYPES="1" Q_IDS="0" bash run_baselines.sh
#
# Environment variables (override defaults):
#   BASELINES        Space-separated list: naive cot react tot all    (default: "naive cot react tot")
#   DATASETS         Space-separated list of dataset keys             (default: stl10_resnet stl10_densenet)
#   Q_TYPES          Space-separated list of question type IDs        (default: "1 2 3")
#   Q_IDS            Space-separated list of question IDs             (default: "0 1 2")
#   VLM              VLM model ID                                     (default: Qwen/Qwen3-VL-8B-Instruct)
#   MODE             train or test                                    (default: test)
#   OUTPUT_ROOT      Root output directory                            (default: ./outputs_baselines)
#   DATASET_DIR      Dataset directory                                (default: ./dataset)
#   MODELS_DIR       Models directory                                 (default: ./models_to_read)
#   NO_EVAL          Set to 1 to skip evaluation                      (default: 0)
#   TOT_BRANCHES     Number of ToT branches                           (default: 3)
#   REACT_MAX_ITER   Max ReAct iterations                             (default: 6)
#   MAX_PARALLEL     Max parallel jobs (0 = sequential)               (default: 0)
#   DRY_RUN          Set to 1 to print commands without running       (default: 0)
#
set -euo pipefail

# ========================================================================
# Defaults
# ========================================================================
BASELINES="${BASELINES:-naive cot react tot}"
DATASETS="${DATASETS:-stl10_resnet stl10_densenet}"
Q_TYPES="${Q_TYPES:-1 2 3}"
Q_IDS="${Q_IDS:-0 1 2}"
VLM="${VLM:-Qwen/Qwen3-VL-8B-Instruct}"
MODE="${MODE:-test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs_baselines}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
MODELS_DIR="${MODELS_DIR:-./models_to_read}"
NO_EVAL="${NO_EVAL:-0}"
TOT_BRANCHES="${TOT_BRANCHES:-3}"
REACT_MAX_ITER="${REACT_MAX_ITER:-6}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ========================================================================
# Dataset → model mapping (same as run_pipeline_batch.py)
# ========================================================================
declare -A DATASET_MODEL_MAP=(
    # Vision
    [stl10_resnet]="vision/stl10_resnet.pth"
    [stl10_densenet]="vision/stl10_densenet.pth"
    [cub_resnet]="vision/cub_resnet.pth"
    [cub_densenet]="vision/cub_densenet.pth"
    # Text
    [imdb_cnn]="text/imdb_cnn.pth"
    [imdb_2layernn]="text/imdb_2layernn.pth"
    [snli_cnn]="text/snli_cnn.pth"
    [snli_2layernn]="text/snli_2layernn.pth"
    # Tabular
    [adult_census]="tabular/adult_census.pth"
    [adult_tabnn]="tabular/adult_tabnn.pth"
    [adult_2layernn]="tabular/adult_2layernn.pth"
    [cancer_2nn]="tabular/cancer_2nn.pth"
    [cancer_tabnn]="tabular/cancer_tabnn.pth"
    [cancer_2layernn]="tabular/cancer_2layernn.pth"
)

declare -A DATASET_MODALITY_MAP=(
    [stl10_resnet]="vision"
    [stl10_densenet]="vision"
    [cub_resnet]="vision"
    [cub_densenet]="vision"
    [imdb_cnn]="text"
    [imdb_2layernn]="text"
    [snli_cnn]="text"
    [snli_2layernn]="text"
    [adult_census]="tabular"
    [adult_tabnn]="tabular"
    [adult_2layernn]="tabular"
    [cancer_2nn]="tabular"
    [cancer_tabnn]="tabular"
    [cancer_2layernn]="tabular"
)

# ========================================================================
# Helpers
# ========================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="${OUTPUT_ROOT}/logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

total_jobs=0
completed_jobs=0
failed_jobs=0
skipped_jobs=0
FAILED_LIST=()

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log_error() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

# ========================================================================
# Build and execute jobs
# ========================================================================
run_job() {
    local baseline="$1"
    local dataset="$2"
    local q_type="$3"
    local q_id="$4"
    local modality="$5"
    local model_url="$6"
    local dataset_json="$7"

    local job_id="${baseline}_${dataset}_q${q_type}_${q_id}"
    local job_output_dir="${OUTPUT_ROOT}/${baseline}/${modality}/${dataset}/q${q_type}/${q_id}"
    local log_file="${LOG_DIR}/${job_id}.log"

    mkdir -p "${job_output_dir}"

    # Build command
    local cmd=(
        python "${SCRIPT_DIR}/run_baseline.py"
        --baseline "${baseline}"
        --dataset "${dataset_json}"
        --question_id "${q_id}"
        --model_url "${model_url}"
        --vlm "${VLM}"
        --output_dir "${job_output_dir}"
        --dataset_dir "${DATASET_DIR}"
        --models_dir "${MODELS_DIR}"
        --mode "${MODE}"
        --tot_branches "${TOT_BRANCHES}"
        --react_max_iter "${REACT_MAX_ITER}"
    )

    if [[ "${NO_EVAL}" == "1" ]]; then
        cmd+=(--no-eval)
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "[DRY-RUN] ${cmd[*]}"
        return 0
    fi

    log "▶ ${job_id}"
    if "${cmd[@]}" > "${log_file}" 2>&1; then
        log "  ✓ ${job_id}"
        return 0
    else
        local rc=$?
        log_error "  ✗ ${job_id}  (exit code ${rc}, log: ${log_file})"
        return 1
    fi
}

# ========================================================================
# Main loop
# ========================================================================
log "============================================"
log "BASELINE BATCH RUNNER"
log "============================================"
log "Baselines:  ${BASELINES}"
log "Datasets:   ${DATASETS}"
log "Q types:    ${Q_TYPES}"
log "Q IDs:      ${Q_IDS}"
log "VLM:        ${VLM}"
log "Mode:       ${MODE}"
log "Output:     ${OUTPUT_ROOT}"
log "Log dir:    ${LOG_DIR}"
log "Dry run:    ${DRY_RUN}"
log "============================================"

for baseline in ${BASELINES}; do
    for dataset in ${DATASETS}; do
        model_url="${DATASET_MODEL_MAP[${dataset}]:-}"
        modality="${DATASET_MODALITY_MAP[${dataset}]:-}"

        if [[ -z "${model_url}" ]]; then
            log_error "Unknown dataset: ${dataset}. Skipping."
            continue
        fi

        for q_type in ${Q_TYPES}; do
            # Find dataset JSON
            dataset_json="${DATASET_DIR}/${MODE}/${modality}/${dataset}_q${q_type}.json"

            # Try _test variant
            if [[ ! -f "${dataset_json}" ]]; then
                dataset_json="${DATASET_DIR}/${MODE}/${modality}/${dataset}_q${q_type}_test.json"
            fi

            if [[ ! -f "${dataset_json}" ]]; then
                log "  ⊘ Skipping ${dataset}_q${q_type}: JSON not found"
                ((skipped_jobs++)) || true
                continue
            fi

            for q_id in ${Q_IDS}; do
                ((total_jobs++)) || true

                if run_job "${baseline}" "${dataset}" "${q_type}" "${q_id}" \
                    "${modality}" "${model_url}" "${dataset_json}"; then
                    ((completed_jobs++)) || true
                else
                    ((failed_jobs++)) || true
                    FAILED_LIST+=("${baseline}_${dataset}_q${q_type}_${q_id}")
                fi
            done
        done
    done
done

# ========================================================================
# Summary
# ========================================================================
echo ""
log "============================================"
log "BATCH RUN COMPLETE"
log "============================================"
log "Total jobs:     ${total_jobs}"
log "Completed:      ${completed_jobs}"
log "Failed:         ${failed_jobs}"
log "Skipped:        ${skipped_jobs}"

if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    log ""
    log "FAILED JOBS:"
    for f in "${FAILED_LIST[@]}"; do
        log "  - ${f}"
    done
fi

log "Logs saved to: ${LOG_DIR}"
log "Outputs saved to: ${OUTPUT_ROOT}"

# Write summary JSON
cat > "${LOG_DIR}/summary.json" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "baselines": "${BASELINES}",
    "datasets": "${DATASETS}",
    "q_types": "${Q_TYPES}",
    "q_ids": "${Q_IDS}",
    "total_jobs": ${total_jobs},
    "completed": ${completed_jobs},
    "failed": ${failed_jobs},
    "skipped": ${skipped_jobs}
}
EOF

exit ${failed_jobs}

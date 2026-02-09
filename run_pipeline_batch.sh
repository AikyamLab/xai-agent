#!/bin/bash
#
# XAI Pipeline Batch Runner
# Runs xai_pipeline_v2.py across multiple modalities, question types, and question IDs
#
# Usage:
#   ./run_pipeline_batch.sh [OPTIONS]
#
# Examples:
#   # Run all vision datasets, q1-q4, first 5 questions
#   ./run_pipeline_batch.sh --modality vision --q_types "1 2 3 4" --question_ids "0 1 2 3 4"
#
#   # Run specific dataset
#   ./run_pipeline_batch.sh --datasets "stl10_resnet" --q_types "1" --question_ids "0"
#
#   # Run text modality, specific question
#   ./run_pipeline_batch.sh --modality text --datasets "imdb_cnn" --q_types "1 2" --question_ids "0 1 2"
#
#   # Use _test variant datasets (if available)
#   ./run_pipeline_batch.sh --datasets "stl10_resnet" --q_types "1 4" --use_test_variant
#

set -e  # Exit on error

# ============================================================================
# Default Configuration
# ============================================================================
BASE_DIR="/standard/AikyamLab/yuyang/xai_agent/framework/trial_2"
DATASET_DIR="${BASE_DIR}/dataset"
MODELS_DIR="${BASE_DIR}/models_to_read"
OUTPUT_DIR="${BASE_DIR}/outputs"

# Default parameters (can be overridden by command line args)
MODALITIES="vision"
Q_TYPES="1"
QUESTION_IDS="0"
DATASETS=""  # Empty means auto-detect based on modality
USE_TEST_VARIANT=false
DRY_RUN=false
NO_EVAL=false
NO_IMPROVEMENT=false
NO_SF=false
FAITHFULNESS_THRESHOLD=0.1
VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
PARALLEL=false
MAX_PARALLEL_JOBS=4

# ============================================================================
# Dataset and Model Mappings
# ============================================================================
declare -A VISION_DATASETS
VISION_DATASETS["stl10_resnet"]="vision/stl10_resnet.pth"
VISION_DATASETS["stl10_densenet"]="vision/stl10_densenet.pth"
VISION_DATASETS["cub_resnet"]="vision/cub_resnet.pth"
VISION_DATASETS["cub_densenet"]="vision/cub_densenet.pth"

declare -A TEXT_DATASETS
TEXT_DATASETS["imdb_cnn"]="text/imdb_cnn.pth"
TEXT_DATASETS["imdb_2layernn"]="text/imdb_2layernn.pth"
TEXT_DATASETS["snli_cnn"]="text/snli_cnn.pth"
TEXT_DATASETS["snli_2layernn"]="text/snli_2layernn.pth"

declare -A TABULAR_DATASETS
TABULAR_DATASETS["adult_census"]="tabular/adult_census.pth"
TABULAR_DATASETS["adult_tabnn"]="tabular/adult_tabnn.pth"
TABULAR_DATASETS["cancer_2nn"]="tabular/cancer_2nn.pth"
TABULAR_DATASETS["cancer_tabnn"]="tabular/cancer_tabnn.pth"

# ============================================================================
# Helper Functions
# ============================================================================
print_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
}

print_subheader() {
    echo ""
    echo "--- $1 ---"
}

log_info() {
    echo "[INFO] $1"
}

log_warn() {
    echo "[WARN] $1" >&2
}

log_error() {
    echo "[ERROR] $1" >&2
}

usage() {
    cat << EOF
XAI Pipeline Batch Runner

Usage: $0 [OPTIONS]

Options:
    --modality MODALITY      Modalities to run: vision, text, tabular, or "all" (default: vision)
                             Can specify multiple: "vision text"
    --datasets DATASETS      Specific datasets to run (default: auto-detect based on modality)
                             Examples: "stl10_resnet imdb_cnn" or "stl10_resnet"
    --q_types Q_TYPES        Question types to run (default: 1)
                             Examples: "1 2 3 4" or "1" or "1 2 3 4 5 6 8 9 10"
    --question_ids IDS       Question IDs to run (default: 0)
                             Examples: "0 1 2 3 4" or "0" or range "0-9"
    --use_test_variant       Use _test variant dataset files if available
    --dry_run                Print commands without executing
    --no_eval                Skip faithfulness evaluation
    --no_improvement         Skip improvement phase
    --no_sf                  Skip strategy faithfulness evaluation
    --faithfulness_threshold Threshold for faithfulness (default: 0.1)
    --vlm MODEL              VLM model ID (default: Qwen/Qwen3-VL-8B-Instruct)
                             API models: gemini-2.5-pro, gemini-3-pro
    --output_dir DIR         Output directory (default: ${BASE_DIR}/outputs)
    --parallel               Run jobs in parallel
    --max_jobs N             Maximum parallel jobs (default: 4)
    -h, --help               Show this help message

Examples:
    # Run vision modality, q1, first question
    $0 --modality vision --q_types "1" --question_ids "0"

    # Run specific dataset with multiple q_types
    $0 --datasets "stl10_resnet" --q_types "1 2 3 4" --question_ids "0 1 2"

    # Run all modalities, all q_types, range of questions
    $0 --modality "all" --q_types "1 2 3 4 5 6" --question_ids "0-4"

    # Dry run to see what would be executed
    $0 --datasets "stl10_resnet" --q_types "1" --question_ids "0" --dry_run

Available Datasets:
    Vision:  stl10_resnet, stl10_densenet, cub_resnet, cub_densenet
    Text:    imdb_cnn, imdb_2layernn, snli_cnn, snli_2layernn
    Tabular: adult_census, adult_tabnn, cancer_2nn, cancer_tabnn
EOF
    exit 0
}

# Parse range like "0-4" into "0 1 2 3 4"
expand_range() {
    local input="$1"
    local result=""

    for item in $input; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start="${BASH_REMATCH[1]}"
            local end="${BASH_REMATCH[2]}"
            for ((i=start; i<=end; i++)); do
                result="$result $i"
            done
        else
            result="$result $item"
        fi
    done

    echo "$result" | xargs  # Trim whitespace
}

# Get datasets for a modality
get_datasets_for_modality() {
    local modality="$1"

    case "$modality" in
        vision)
            echo "stl10_resnet stl10_densenet cub_resnet cub_densenet"
            ;;
        text)
            echo "imdb_cnn imdb_2layernn snli_cnn snli_2layernn"
            ;;
        tabular)
            echo "adult_census adult_tabnn cancer_2nn cancer_tabnn"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Get model path for a dataset
get_model_path() {
    local dataset="$1"

    # Check each mapping
    if [[ -n "${VISION_DATASETS[$dataset]}" ]]; then
        echo "${MODELS_DIR}/${VISION_DATASETS[$dataset]}"
    elif [[ -n "${TEXT_DATASETS[$dataset]}" ]]; then
        echo "${MODELS_DIR}/${TEXT_DATASETS[$dataset]}"
    elif [[ -n "${TABULAR_DATASETS[$dataset]}" ]]; then
        echo "${MODELS_DIR}/${TABULAR_DATASETS[$dataset]}"
    else
        # Try common patterns
        if [[ -f "${MODELS_DIR}/vision/${dataset}.pth" ]]; then
            echo "${MODELS_DIR}/vision/${dataset}.pth"
        elif [[ -f "${MODELS_DIR}/text/${dataset}.pth" ]]; then
            echo "${MODELS_DIR}/text/${dataset}.pth"
        elif [[ -f "${MODELS_DIR}/tabular/${dataset}.pth" ]]; then
            echo "${MODELS_DIR}/tabular/${dataset}.pth"
        else
            echo ""
        fi
    fi
}

# Get modality for a dataset
get_modality_for_dataset() {
    local dataset="$1"

    if [[ -n "${VISION_DATASETS[$dataset]}" ]]; then
        echo "vision"
    elif [[ -n "${TEXT_DATASETS[$dataset]}" ]]; then
        echo "text"
    elif [[ -n "${TABULAR_DATASETS[$dataset]}" ]]; then
        echo "tabular"
    else
        echo ""
    fi
}

# Build dataset file path
get_dataset_path() {
    local dataset="$1"
    local q_type="$2"
    local use_test="$3"
    local modality="$4"

    local base_name="${dataset}_q${q_type}"
    local dataset_subdir=""

    case "$modality" in
        vision)
            dataset_subdir="vision"
            ;;
        text)
            dataset_subdir="text"
            ;;
        tabular)
            dataset_subdir="tabular"
            ;;
    esac

    # Try _test variant first if requested
    if [[ "$use_test" == "true" ]]; then
        local test_path="${DATASET_DIR}/${dataset_subdir}/${base_name}_test.json"
        if [[ -f "$test_path" ]]; then
            echo "$test_path"
            return
        fi
    fi

    # Try standard path
    local standard_path="${DATASET_DIR}/${dataset_subdir}/${base_name}.json"
    if [[ -f "$standard_path" ]]; then
        echo "$standard_path"
        return
    fi

    # Not found
    echo ""
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modality)
            MODALITIES="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --q_types)
            Q_TYPES="$2"
            shift 2
            ;;
        --question_ids)
            QUESTION_IDS="$2"
            shift 2
            ;;
        --use_test_variant)
            USE_TEST_VARIANT=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --no_eval)
            NO_EVAL=true
            shift
            ;;
        --no_improvement)
            NO_IMPROVEMENT=true
            shift
            ;;
        --no_sf)
            NO_SF=true
            shift
            ;;
        --faithfulness_threshold)
            FAITHFULNESS_THRESHOLD="$2"
            shift 2
            ;;
        --vlm)
            VLM_MODEL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --max_jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# ============================================================================
# Expand and Validate Parameters
# ============================================================================
print_header "XAI Pipeline Batch Runner"

# Expand modalities
if [[ "$MODALITIES" == "all" ]]; then
    MODALITIES="vision text tabular"
fi

# Expand question IDs (handle ranges)
QUESTION_IDS=$(expand_range "$QUESTION_IDS")

log_info "Configuration:"
log_info "  Modalities: $MODALITIES"
log_info "  Datasets: ${DATASETS:-auto-detect}"
log_info "  Q Types: $Q_TYPES"
log_info "  Question IDs: $QUESTION_IDS"
log_info "  Use test variant: $USE_TEST_VARIANT"
log_info "  Dry run: $DRY_RUN"
log_info "  No eval: $NO_EVAL"
log_info "  No improvement: $NO_IMPROVEMENT"
log_info "  No strategy faithfulness: $NO_SF"
log_info "  Faithfulness threshold: $FAITHFULNESS_THRESHOLD"
log_info "  Output dir: $OUTPUT_DIR"

# ============================================================================
# Build and Run Jobs
# ============================================================================
declare -a JOBS=()
TOTAL_JOBS=0
SUCCESS_JOBS=0
FAILED_JOBS=0

# Build list of datasets to process
DATASETS_TO_RUN=""
if [[ -n "$DATASETS" ]]; then
    DATASETS_TO_RUN="$DATASETS"
else
    # Auto-detect based on modalities
    for modality in $MODALITIES; do
        DATASETS_TO_RUN="$DATASETS_TO_RUN $(get_datasets_for_modality $modality)"
    done
fi

print_subheader "Building job list"

for dataset in $DATASETS_TO_RUN; do
    # Get modality for this dataset
    modality=$(get_modality_for_dataset "$dataset")
    if [[ -z "$modality" ]]; then
        log_warn "Could not determine modality for dataset: $dataset, skipping"
        continue
    fi

    # Get model path
    model_path=$(get_model_path "$dataset")
    if [[ -z "$model_path" || ! -f "$model_path" ]]; then
        log_warn "Model not found for dataset: $dataset, skipping"
        continue
    fi

    for q_type in $Q_TYPES; do
        # Get dataset file path
        dataset_path=$(get_dataset_path "$dataset" "$q_type" "$USE_TEST_VARIANT" "$modality")

        if [[ -z "$dataset_path" || ! -f "$dataset_path" ]]; then
            log_warn "Dataset file not found: ${dataset}_q${q_type}.json, skipping"
            continue
        fi

        for question_id in $QUESTION_IDS; do
            # Build command
            CMD="python ${BASE_DIR}/xai_pipeline_v2.py"
            CMD="$CMD --dataset $dataset_path"
            CMD="$CMD --question_id $question_id"
            CMD="$CMD --model_url $model_path"
            CMD="$CMD --dataset_dir $DATASET_DIR"
            CMD="$CMD --models_dir $MODELS_DIR"
            CMD="$CMD --output_dir $OUTPUT_DIR"
            CMD="$CMD --vlm $VLM_MODEL"
            CMD="$CMD --faithfulness_threshold $FAITHFULNESS_THRESHOLD"

            if [[ "$NO_EVAL" == "true" ]]; then
                CMD="$CMD --no-eval"
            fi

            if [[ "$NO_IMPROVEMENT" == "true" ]]; then
                CMD="$CMD --no-improvement"
            fi

            if [[ "$NO_SF" == "true" ]]; then
                CMD="$CMD --no-sf"
            fi

            JOBS+=("$CMD")
            TOTAL_JOBS=$((TOTAL_JOBS + 1))

            log_info "Job $TOTAL_JOBS: ${dataset}_q${q_type}_${question_id}"
        done
    done
done

print_subheader "Job Summary"
log_info "Total jobs to run: $TOTAL_JOBS"

if [[ "$DRY_RUN" == "true" ]]; then
    print_subheader "Dry Run - Commands to execute"
    for job in "${JOBS[@]}"; do
        echo "$job"
        echo ""
    done
    exit 0
fi

# ============================================================================
# Execute Jobs
# ============================================================================
print_subheader "Executing jobs"

# Create log directory
LOG_DIR="${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
log_info "Logs will be saved to: $LOG_DIR"

run_job() {
    local job_num="$1"
    local cmd="$2"
    local log_file="${LOG_DIR}/job_${job_num}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting job $job_num" | tee -a "$log_file"
    echo "Command: $cmd" >> "$log_file"
    echo "" >> "$log_file"

    if eval "$cmd" >> "$log_file" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $job_num completed successfully"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $job_num FAILED"
        return 1
    fi
}

if [[ "$PARALLEL" == "true" ]]; then
    # Run jobs in parallel
    log_info "Running jobs in parallel (max $MAX_PARALLEL_JOBS concurrent)"

    job_num=0
    for job in "${JOBS[@]}"; do
        ((++job_num)) || true

        # Wait if we have too many background jobs
        while [[ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]]; do
            sleep 1
        done

        run_job "$job_num" "$job" &
    done

    # Wait for all jobs to complete
    wait
else
    # Run jobs sequentially
    job_num=0
    for job in "${JOBS[@]}"; do
        ((++job_num)) || true

        print_subheader "Job $job_num / $TOTAL_JOBS"

        if run_job "$job_num" "$job"; then
            ((++SUCCESS_JOBS)) || true
        else
            ((++FAILED_JOBS)) || true
        fi
    done
fi

# ============================================================================
# Summary
# ============================================================================
print_header "Execution Summary"
log_info "Total jobs: $TOTAL_JOBS"
log_info "Successful: $SUCCESS_JOBS"
log_info "Failed: $FAILED_JOBS"
log_info "Logs saved to: $LOG_DIR"

if [[ $FAILED_JOBS -gt 0 ]]; then
    log_warn "Some jobs failed. Check logs for details."
    exit 1
fi

echo ""
echo "Done!"

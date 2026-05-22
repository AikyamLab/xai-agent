# Text Baselines Execution Status

## Command to Run Text Baselines

```bash
cd /Users/rraghavkaushik/xai-agent

python run_pipeline_batch.py \
    --vlm gemini-2.5-pro \
    --mode test \
    --modality text \
    --q_types 1 2 3 4 \
    --question_ids 0-9 \
    --output_dir outputs/ \
    --parallel --max_workers 4
```

This will run:
- **4 text datasets**: imdb_cnn, imdb_2layernn, snli_cnn, snli_2layernn
- **4 question types**: Q1, Q2, Q3, Q4
- **10 question IDs**: 0-9
- **Total jobs**: 160 (4 datasets × 4 q_types × 10 question_ids)

### Flags Explanation
- `--vlm gemini-2.5-pro`: Uses Gemini 2.5 Pro as the vision language model
- `--mode test`: Uses test split of datasets
- `--modality text`: Runs only text datasets
- `--q_types 1 2 3 4`: Runs all 4 question types
- `--question_ids 0-9`: Runs question IDs 0 through 9
- `--parallel --max_workers 4`: Runs 4 jobs in parallel
- **By default**: Both `--no-sf` (strategy faithfulness) and `--no-improvement` flags are OFF, meaning SF and improvement loops are ENABLED (this is the baseline configuration)

## Dependency Issue

There is a **library version compatibility issue** preventing the baselines from running:

### Error
```
ValueError: Invalid pattern: '**' can only be an entire path component
```

### Root Cause
The `datasets==4.5.0` library (in requirements.txt) has an incompatible glob pattern syntax with the HuggingFace Hub API when trying to load IMDB dataset during model loading.

### Solution
You need to downgrade the `datasets` package to a compatible version:

```bash
pip install datasets==2.18.0
```

Then run the batch command again.

## Files Modified
- **run_pipeline_batch.py**: Fixed `BASE_DIR` to use dynamic path instead of hardcoded `/standard/AikyamLab/...` path

## Output Structure
Results will be saved in:
```
outputs/
  imdb_cnn_q{1-4}_{0-9}/
    results.json           # Full pipeline output
  imdb_2layernn_q{1-4}_{0-9}/
    results.json
  snli_cnn_q{1-4}_{0-9}/
    results.json
  snli_2layernn_q{1-4}_{0-9}/
    results.json
  strategy_faithfulness/   # SF evaluation files
  logs/                    # Per-job logs
```

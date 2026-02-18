# XAI Agent Framework — Trial 2

A three-agent (Proposer → Actor → Critic) pipeline for generating and evaluating XAI explanations across vision, text, and tabular modalities.

---

## Setup

### 1. Create / activate virtualenv

```bash
python -m venv ~/.venvs/xai_agent
source ~/.venvs/xai_agent/bin/activate
```

Install dependencies (PyTorch with CUDA first, then the rest):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### 2. Set environment variables

Copy and fill in the keys before running (or add them to your shell profile):

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache
export TORCH_HOME=/path/to/torch_cache

export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export TINKER_API_KEY=...
```

The helper script `setup_env.sh` sets all of the above for this installation:

```bash
source setup_env.sh
```

---

## Dataset layout

Benchmark JSON files live under `dataset/{mode}/{modality}/`:

```
dataset/
  train/
    vision/    stl10_resnet_q1.json, ...
    text/      imdb_cnn_q2.json, ...
    tabular/   adult_census_q3.json, ...
  test/
    vision/    ...
    text/      ...
    tabular/   ...
```

Raw images for vision tasks are under `dataset/image/{dataset_name}/`.

---

## Running a single question

```bash
python xai_pipeline_v2.py \
    --dataset  dataset/test/vision/stl10_resnet_q1.json \
    --question_id 0 \
    --model_url  models_to_read/vision/stl10_resnet.pth \
    --vlm  gemini-2.5-pro \
    --mode test \
    --output_dir outputs/
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--mode` | `test` | Which split to use (`train` or `test`) |
| `--vlm` | `Qwen/Qwen3-VL-8B-Instruct` | VLM backend. Options: local Qwen, `gemini-2.5-pro`, `claude-sonnet-4-5-20250929`, `tinker/...` |
| `--no-eval` | off | Skip faithfulness evaluation |
| `--no-improvement` | off | Skip improvement loop |
| `--no-sf` | off | Skip strategy faithfulness evaluation |
| `--sf_max_samples` | None | Cap tool-config samples for strategy faithfulness (2^N full by default) |
| `--faithfulness_threshold` | `0.1` | Score below which improvement is triggered |

---

## Batch runner

Two equivalent interfaces — a shell script and a Python script.

### Shell (`run_pipeline_batch.sh`)

```bash
./run_pipeline_batch.sh \
    --vlm gemini-2.5-pro \
    --mode test \
    --datasets "stl10_resnet cub_resnet" \
    --q_types "1 2 3 4" \
    --question_ids "0-9" \
    --output_dir outputs/ \
    --parallel \
    --max_jobs 4
```

### Python (`run_pipeline_batch.py`)

```bash
python run_pipeline_batch.py \
    --vlm gemini-2.5-pro \
    --mode test \
    --datasets stl10_resnet cub_resnet \
    --q_types 1 2 3 4 \
    --question_ids 0-9 \
    --output_dir outputs/ \
    --parallel --max_workers 4
```

Both accept `--modality {vision,text,tabular,all}` instead of `--datasets` to run all datasets for a modality. Use `--dry_run` to preview commands without executing.

### Config file

```bash
python run_pipeline_batch.py --config batch_config_example.json
```

See `batch_config_example.json` for all available fields (including `"mode"`).

---

## SLURM

Edit `myjob.slurm` to set your datasets, question IDs, and VLM, then submit:

```bash
sbatch myjob.slurm
```

Logs go to `outputs/logs/<timestamp>/`.

---

## Available datasets

| Name | Modality | Model file |
|---|---|---|
| `stl10_resnet` / `stl10_densenet` | vision | `models_to_read/vision/stl10_*.pth` |
| `cub_resnet` / `cub_densenet` | vision | `models_to_read/vision/cub_*.pth` |
| `imdb_cnn` / `imdb_2layernn` | text | `models_to_read/text/imdb_*.pth` |
| `snli_cnn` / `snli_2layernn` | text | `models_to_read/text/snli_*.pth` |
| `adult_census` / `adult_tabnn` / `adult_2layernn` | tabular | `models_to_read/tabular/adult_*.pth` |
| `cancer_2nn` / `cancer_tabnn` / `cancer_2layernn` | tabular | `models_to_read/tabular/cancer_*.pth` |

---

## Output structure

```
outputs/
  {dataset}_{qtype}_{question_id}/
    results.json          # full pipeline output
  strategy_faithfulness/  # SF evaluation files
  training_data/          # saved training datapoints
    {modality}/{dataset}/{qtype}/{row_no}/training_datapoint.json
  logs/<timestamp>/       # per-job logs from batch runner
```

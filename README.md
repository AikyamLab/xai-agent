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
| `--dataset_variant` | `default` | Path preset for dataset/model roots (`default` or `ood`) |
| `--dataset_dir` / `--models_dir` | variant-dependent | Explicit root overrides for benchmark JSONs and model checkpoints |

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

For OOD runs, use `--dataset_variant ood` (or pass explicit `--dataset_dir` / `--models_dir`).

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

---

# Running Baseline Agents

This branch contains implementations of multiple baseline agents for XAI explanation generation. Baselines are alternative explanation strategies to compare against the three-agent pipeline.

## Available Baseline Agents

| Agent Type | File | Description |
|---|---|---|
| **Naive Agent** | `baselines/naive_agent.py` | Simple prompt-based explanation (no tools) |
| **CoT Agent** | `baselines/cot_agent.py` | Chain-of-thought reasoning with step-by-step analysis |
| **ReAct Agent** | `baselines/react_agent.py` | Reasoning + Acting with tool use for verification |
| **ToT Agent** | `baselines/tot_agent.py` | Tree-of-Thought with multi-path exploration and ranking |

## Quick Start: Running Baselines

### 1. Setup

Ensure you've installed dependencies and set environment variables (see Setup section above):

```bash
pip install -r requirements.txt
export TINKER_API_KEY=...
export GEMINI_API_KEY=...
# Add other API keys as needed
```

**IMPORTANT: Fix dataset library version**

The current `requirements.txt` has a compatibility issue with HuggingFace datasets. Fix this before running:

```bash
pip install --upgrade datasets==2.18.0
```

This prevents the `Invalid pattern: '**' can only be an entire path component` error when loading text datasets.

### 2. Run a Single Baseline

To test a single question with a baseline agent:

```bash
python baselines/naive_agent.py \
    --dataset dataset/test/text/imdb_cnn_q1.json \
    --question_id 0 \
    --model_path models_to_read/text/imdb_cnn.pth \
    --vlm tinker/qwen3.6-35B-A3B \
    --output_dir outputs/baselines/
```

### 3. Run All Baselines in Batch

Use the batch runner to systematically compare baselines across datasets and question types:

```bash
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality text \
    --datasets "imdb_cnn imdb_2layernn" \
    --agents "naive cot react tot" \
    --q_types 1 2 3 4 \
    --question_ids 0-19 \
    --output_dir outputs/baselines/ \
    --parallel --max_workers 4
```

### 4. Using the Shell Script

For convenience, use the provided shell script:

```bash
./run_baseline_agents.sh \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality text \
    --agents "naive cot react tot" \
    --q_types 1 2 3 4 \
    --question_ids all \
    --parallel
```

## Command Reference

### Main Parameters

| Parameter | Options | Default | Description |
|---|---|---|---|
| `--vlm` | `tinker/...`, `gemini-2.5-pro`, `claude-sonnet-...` | `tinker/qwen3.6-35B-A3B` | Vision Language Model to use |
| `--mode` | `test`, `train` | `test` | Dataset split |
| `--modality` | `text`, `vision`, `tabular`, `all` | required | Which modality to run |
| `--datasets` | space-separated list | auto-detect | Specific datasets (or use `--modality`) |
| `--agents` | `naive cot react tot` (space-separated) | all | Which baselines to run |
| `--q_types` | 1-10 (space-separated) | 1 2 3 4 | Question types to evaluate |
| `--question_ids` | `0-19`, `all`, or space-separated | `0-9` | Question IDs to test |
| `--output_dir` | path | `outputs/baselines/` | Where to save results |
| `--parallel` | flag | off | Run jobs in parallel |
| `--max_workers` | N | 4 | Number of parallel workers |
| `--dry_run` | flag | off | Show commands without executing |

### Comparing with Main Pipeline

To run the three-agent pipeline on the same data for comparison:

```bash
# Text modality, all 10 question types, 20 qids
python run_pipeline_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality text \
    --q_types 1 2 3 4 5 6 7 8 9 10 \
    --question_ids 0-19 \
    --output_dir outputs/pipeline/ \
    --parallel --max_workers 8
```

This generates results comparable to the baselines for analysis and evaluation.

## Expected Output Structure

```
outputs/baselines/
  {dataset}_{agent}_{qtype}_{question_id}/
    result.json           # Agent response and metadata
    explanation.txt       # Generated explanation text
    timing.json          # Execution timing
  logs/
    {agent}_{timestamp}.log  # Per-agent logs
```

## Baseline Agent Characteristics

### Naive Agent
- **Strategy**: Direct prompt without tools or reasoning
- **Use case**: Baseline / lower bound for comparison
- **Pros**: Fast, simple, deterministic
- **Cons**: May lack depth or reasoning transparency
- **Typical time**: 5-10 seconds per question

### Chain-of-Thought (CoT) Agent
- **Strategy**: Multi-step reasoning before generating explanation
- **Use case**: Adding reasoning transparency
- **Pros**: Shows reasoning steps, more interpretable
- **Cons**: Slightly slower, may be verbose
- **Typical time**: 10-20 seconds per question

### ReAct Agent
- **Strategy**: Reason about question, use tools to verify answers
- **Use case**: Tool-grounded explanations
- **Pros**: Can verify facts, ground in data
- **Cons**: Tool-dependent quality, more complex prompts
- **Typical time**: 20-40 seconds per question

### Tree-of-Thought (ToT) Agent
- **Strategy**: Explore multiple reasoning paths and select best
- **Use case**: High-quality explanations with exploration
- **Pros**: Best explanation quality, considers alternatives
- **Cons**: Slowest, high cost
- **Typical time**: 40-120 seconds per question

## Troubleshooting

### Issue: "Invalid pattern: '**' can only be an entire path component"
**Solution**: Downgrade datasets:
```bash
pip install datasets==2.18.0
```

### Issue: VLM API timeouts or rate limits
**Solution**: Reduce `--max_workers` or run sequentially (no `--parallel`)
```bash
python run_baseline_agent_batch.py ... --max_workers 2
```

### Issue: TINKER_API_KEY not found
**Solution**: Set environment variable:
```bash
export TINKER_API_KEY=your_key_here
```

### Issue: Model file not found (e.g., `imdb_cnn.pth`)
**Solution**: Ensure model is in `models_to_read/` directory. Check available:
```bash
ls models_to_read/text/
ls models_to_read/vision/
ls models_to_read/tabular/
```

## Example Workflow: Running Full Baseline Comparison

### Step 1: Test Setup
```bash
# Quick smoke test with 1 dataset, 1 agent, 1 qtype, 3 qids
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality text \
    --datasets imdb_cnn \
    --agents naive \
    --q_types 1 \
    --question_ids 0-2 \
    --output_dir outputs/baselines/smoke_test/ \
    --dry_run
```

### Step 2: Run All Baselines on Text Modality
```bash
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality text \
    --agents "naive cot react tot" \
    --q_types 1 2 3 4 \
    --question_ids 0-19 \
    --output_dir outputs/baselines/text/ \
    --parallel --max_workers 4
```

Expected time: ~4-6 hours (depends on VLM speed and parallelism)

### Step 3: Evaluate Results
```bash
# Analyze baseline outputs (evaluation script)
python summarize_eval_scores.py outputs/baselines/text/
```

## Contributing New Baselines

To add a new baseline agent:

1. Create `baselines/my_agent.py` inheriting from `BaseAgent`
2. Implement the required methods:
   - `__init__(vlm_config)`: Initialize VLM
   - `generate_explanation(question, context, model_input)`: Generate explanation
   - `invoke_vlm(prompt)`: VLM interface
3. Register in `baselines/__init__.py`
4. Add to agent list in `run_baseline_agent_batch.py`

See `baselines/naive_agent.py` for a minimal example.

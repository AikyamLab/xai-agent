# XAI Agent Framework — Copilot Instructions

## Overview

The XAI Agent Framework is a three-agent (Proposer → Actor → Critic) pipeline for generating and evaluating explainable AI (XAI) explanations across **vision**, **text**, and **tabular** modalities using Vision-Language Models (VLMs).

The project is research-focused with modular architecture organized around:
- **Agents**: Core reasoning entities (`proposer_agent.py`, `actor_agent.py`, `critic_agent.py`)
- **Prompts**: Question-specific prompt builders organized by question type (Q1–Q10)
- **Evaluation**: Modality-specific evaluators and faithfulness scoring
- **Models**: Dataset-specific model loaders under `models_to_read/`

## Setup & Environment

### Environment Variables (Required)

Before running any pipeline:

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache
export TORCH_HOME=/path/to/torch_cache
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export TINKER_API_KEY=...
```

Use the provided helper: `source setup_env.sh`

### Installation

1. **Create virtual environment** (Python 3.11):
   ```bash
   python -m venv ~/.venvs/xai_agent
   source ~/.venvs/xai_agent/bin/activate
   ```

2. **Install PyTorch with CUDA 13.0 first** (required before other packages):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements.txt
   ```

## Running Pipelines

### Single Question (End-to-End)

```bash
python xai_pipeline_v2.py \
    --dataset dataset/test/vision/stl10_resnet_q1.json \
    --question_id 0 \
    --model_url models_to_read/vision/stl10_resnet.pth \
    --vlm gemini-2.5-pro \
    --mode test
```

**Key flags:**
- `--vlm`: Backend VLM (`gemini-2.5-pro`, `claude-sonnet-4-5-20250929`, local `Qwen/Qwen3-VL-8B-Instruct`, etc.)
- `--mode`: `train` or `test` split
- `--no-eval`: Skip faithfulness evaluation
- `--no-improvement`: Skip improvement loop (no re-prompting when faithfulness is low)
- `--faithfulness_threshold`: Score (0–1) below which improvement is triggered (default: 0.1)

### Batch Runner

Two equivalent interfaces:

**Python interface** (recommended for parallel jobs):
```bash
python run_pipeline_batch.py \
    --vlm gemini-2.5-pro \
    --mode test \
    --datasets stl10_resnet cub_resnet \
    --q_types 1 2 3 4 \
    --question_ids 0-9 \
    --parallel --max_workers 4
```

**Shell interface** (for SLURM compatibility):
```bash
./run_pipeline_batch.sh \
    --vlm gemini-2.5-pro \
    --mode test \
    --datasets stl10_resnet cub_resnet \
    --q_types 1 2 3 4 \
    --question_ids 0-9 \
    --parallel --max_jobs 4
```

**Config file** (for reproducible runs):
```bash
python run_pipeline_batch.py --config batch_config.json
```

See `batch_config_example.json` for all available config fields.

### SLURM Submission

Edit `myjob.slurm` with your parameters, then:
```bash
sbatch myjob.slurm
```

Logs save to `outputs/logs/<timestamp>/`.

## Architecture & Key Modules

### Three-Agent Pipeline (`three_agent_system_new.py`)

**Flow**: Proposer → Actor → Critic

1. **ProposerAgent** (`agents/proposer_agent.py`)
   - Receives question, model, prediction, and image/text/tabular data
   - Selects XAI strategy (e.g., "highlight most important features")
   - Returns structured strategy object

2. **ActorAgent** (`agents/actor_agent.py`)
   - Receives strategy from Proposer
   - Executes XAI tools (gradient-based, attention-based, perturbation, etc.)
   - Generates multimodal explanations (text + visualizations)

3. **CriticAgent** (`agents/critic_agent.py`)
   - Evaluates explanation faithfulness
   - Can trigger improvement loop if score is below `--faithfulness_threshold`
   - Returns quantitative evaluation metrics

### Question Types & Prompts (`prompts/`)

**10 question types** (Q1–Q10) span three modalities:

| Q Type | Modality | Module | Focus |
|--------|----------|--------|-------|
| Q1–Q4 | Vision, Text, Tabular | `q1_most_responsible.py` – `q4_contrastive_instances.py` | Attribution (feature importance) |
| Q5–Q7 | Vision, Text, Tabular | `q5_mask_prediction.py` – `q7_change_prediction.py` | Prediction change under perturbation |
| Q8–Q10 | Vision, Text, Tabular | `q8_irrelevant_parts.py` – `q10_similar_different.py` | Explanation quality & similarity |

**Adding a new question:**
1. Create `prompts/q{N}_name.py` with a class inheriting `PromptBuilder`
2. Register in `question_templates_new.py` under `QUESTION_METADATA`
3. Implement `build_proposer_prompt()`, `build_actor_prompt()`, `build_critic_prompt()`

### Evaluation (`evaluation/`)

**Explanation Faithfulness** (`evaluation/explanation_faithfulness/`)
- Q8, Q9, Q10 evaluators with question-specific metrics
- Integrated into Critic agent

**Strategy Faithfulness** (`evaluation/strategy_faithfulness/`)
- Tool attribution: tracks which tools contributed to explanation
- Caching: `cache_manager.py` deduplicates expensive VLM calls

**Masking & Perturbation** (`evaluation/masking_utils.py`)
- Vision: Inpainting-based masking (NVIDIA SD or custom)
- Text: Token removal
- Tabular: Value substitution

### Model Loaders (`models_to_read/`)

Each dataset requires a custom loader module with this signature:

```python
# models_to_read/{modality}/load_{dataset_name}.py
def load_model(model_path, device='cuda'):
    """Load model weights and architecture."""
    
def load_data(data_path):
    """Load and preprocess inputs (images/text/features)."""
    
def predict(model, input_data):
    """Forward pass returning predictions (logits or class indices)."""
```

**Existing loaders:**
- Vision: STL-10, CUB (ResNet, DenseNet variants)
- Text: IMDB, SNLI (CNN, 2-layer NN)
- Tabular: Adult Census, Cancer (2-layer NN, TabNN)

### VLM Wrapper (`vlm_wrapper.py`)

Unified interface for multiple VLM backends:

```python
vlm = create_vlm(
    model_name="gemini-2.5-pro",  # or "claude-sonnet-...", "Qwen/..."
    api_key="...",                 # Optional for API models
    device="cuda"                  # For local models
)

response = vlm.query(prompt, image=img_path, tools=[...])
```

Supports:
- **API models**: Claude, Gemini, Tinker
- **Local models**: Qwen3-VL-8B-Instruct (requires `qwen-vl-utils`)

## Dataset Layout

Benchmarks are organized by split and modality:

```
dataset/
  train/
    vision/    (stl10_resnet_q1.json, cub_resnet_q2.json, ...)
    text/      (imdb_cnn_q1.json, snli_cnn_q2.json, ...)
    tabular/   (adult_census_q1.json, cancer_2nn_q2.json, ...)
  test/
    vision/
    text/
    tabular/

dataset/image/{dataset_name}/  # Raw vision inputs (JPG/PNG)
```

Dataset JSON format:
```json
{
  "dataset_name": "stl10_resnet",
  "modality": "vision",
  "q_type": 1,
  "question_id": 0,
  "context": {
    "image_path": "dataset/image/stl10/sample_0.jpg",
    "prediction": 2,
    "ground_truth": 2,
    "model_info": {...}
  }
}
```

## Output Structure

```
outputs/
  {dataset}_{qtype}_{question_id}/
    results.json              # Full pipeline output (proposer strategy, actor explanation, critic evaluation)
  strategy_faithfulness/      # SF evaluation artifacts (tool attribution caching)
  training_data/              # Saved training datapoints for evaluation
    {modality}/{dataset}/{qtype}/{row_no}/training_datapoint.json
  logs/{timestamp}/           # Per-job logs from batch runner (SLURM)
```

Each `results.json` contains:
- `proposer_output`: Strategy selection with rationale
- `actor_output`: Generated explanation, tool calls, visualizations
- `critic_output`: Faithfulness scores, improvement loop iterations
- `metadata`: Timestamps, VLM model, temperature, etc.

## Common Patterns & Conventions

### Modality Abstraction

The framework handles vision, text, and tabular data uniformly:

- **Vision**: PIL Images, tensor operations via masking and inpainting
- **Text**: Token-level operations, NLTK tokenization
- **Tabular**: Feature replacement, pandas DataFrames

Always check the modality via `question_template.modality` when writing modality-specific code.

### Prompt Construction

Prompts are built dynamically based on context:

```python
template = get_question_template(q_type=1, modality="vision")
builder = get_prompt_builder(q_type=1, modality="vision")

context = {
    "question_id": 0,
    "prediction": 2,
    "image_path": "...",
    "model_info": {...}
}

proposer_prompt = builder.build_proposer_prompt(context)
actor_prompt = builder.build_actor_prompt(context, proposer_strategy)
critic_prompt = builder.build_critic_prompt(context, actor_results)
```

### Error Handling & Retries

- JSON parsing failures: Use `BaseAgent.safe_parse_json()` with fallback extraction
- VLM timeouts: Wrapped with retry logic in `vlm_wrapper.py`
- Missing model loaders: Dynamic import with clear FileNotFoundError messages

### Logging & Debugging

Set `LOGLEVEL` environment variable:
```bash
export LOGLEVEL=DEBUG
python xai_pipeline_v2.py ...
```

Critical debugging points:
- `xai_pipeline_v2.py`: High-level orchestration
- `three_agent_system_new.py`: Agent flow and orchestration
- `agents/`: Individual agent logic
- `evaluation/`: Evaluation metrics and masking logic

### Tips for Modifications

1. **Add a new VLM backend**: Extend `VisionLanguageModel` base class in `vlm_wrapper.py`
2. **Add a new evaluation metric**: Subclass `BaseEvaluator` in `evaluation/base_evaluator.py`
3. **Modify agent behavior**: Edit respective agent file; always call parent `__init__` and respect the agent's contract
4. **Add a new modality**: Ensure loaders exist, update `Modality` enum in `prompts/base_prompt.py`, add modality-specific evaluator

## Performance & Resource Notes

- **GPU memory**: ~24GB for batch inference with Qwen3-VL-8B-Instruct
- **VLM API costs**: Costs scale with batch size; `--parallel` with `--max_workers 4` is typical
- **Masking overhead**: Inpainting-based masking is slower than token-level masking; disable with `--no-eval` for prototyping
- **Cache strategy**: Strategy faithfulness evaluation caches VLM calls; clear cache in `outputs/strategy_faithfulness/` if stale

## MCP Server Configuration

For optimal development experience, enable these MCP servers:

### Python REPL (`python`)
**Purpose**: Interactive debugging, prototyping agent behavior, testing prompt outputs

**Example use**:
```python
# Debug Proposer strategy selection
from three_agent_system_new import create_three_agent_system
from vlm_wrapper import create_vlm

vlm = create_vlm("Qwen/Qwen3-VL-8B-Instruct")
proposer, actor, critic = create_three_agent_system(vlm)
strategy = proposer.run(question, template, model_info, image_path, prediction)
print(strategy)  # Inspect strategy structure
```

### Command Execution (`bash` or `shell`)
**Purpose**: Running pipelines, batch jobs, and validation scripts

**Example commands**:
```bash
python xai_pipeline_v2.py --help
python run_pipeline_batch.py --dry_run --config test_config.json
python -m pytest evaluation/ -v  # If tests are added
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: no module named 'qwen_vl_utils'` | Run `pip install qwen-vl-utils` |
| `FileNotFoundError: Loader module not found` | Ensure model file exists and matching `load_{model_name}.py` is in `models_to_read/{modality}/` |
| `CUDA out of memory` | Reduce batch size, disable `--no-eval`, or use smaller VLM |
| `VLM API rate limit` | Reduce `--max_workers`, add delays, or switch to local model |
| JSON parsing errors in VLM output | Check VLM temperature (lower = more consistent); enable debug logging |

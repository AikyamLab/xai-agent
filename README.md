# MEA: Multi-modal Explanation Agent

A three-agent framework for generating faithful, multi-modal XAI explanations across tabular, text, and vision domains.

## Overview

MEA consists of three collaborating agents in a Proposer → Actor → Critic pipeline:

- **Proposer**: Analyzes the question and selects an XAI strategy (tool selection, analysis plan)
- **Actor**: Executes the chosen XAI tools and assembles the explanation
- **Critic**: Evaluates the explanation quality against the model's actual behavior

The framework addresses 10 question types (Q1–Q10) spanning feature attribution, contrastive reasoning, decision boundaries, robustness analysis, and counterfactual queries, across three modalities.

### Question Types

| Type | Description |
|------|-------------|
| Q1 | Feature importance — which features most influence the prediction |
| Q2 | Comparative feature importance across instances |
| Q3 | Contrastive explanation — why prediction A rather than B |
| Q4 | Contrastive instance pair analysis |
| Q5 | Decision boundary characterization |
| Q6 | Global vs. local feature importance |
| Q7 | Prediction robustness under feature perturbation |
| Q8 | Robustness analysis across multiple instances |
| Q9 | Counterfactual: minimal change to flip prediction |
| Q10 | Multi-instance counterfactual analysis |

### XAI Tools

LIME, SHAP, SmoothGrad, GradCAM, Guided Backpropagation, Integrated Gradients, Sensitivity Analysis

---

## Setup

### 1. Install dependencies

```bash
python -m venv ~/.venvs/mea
source ~/.venvs/mea/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### 2. Environment variables

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache
export TORCH_HOME=/path/to/torch_cache

export TINKER_API_KEY=...       # required for Qwen3.6-35B-A3B (primary model)
export GEMINI_API_KEY=...       # optional, for Gemini API baseline
export ANTHROPIC_API_KEY=...    # optional, for Claude API baseline
export OPENAI_API_KEY=...       # optional, for GPT API baseline
```

The helper script `setup_env.sh` sets all of the above for this installation:

```bash
source setup_env.sh
```

---

## Datasets

### In-distribution

| Dataset | Modality | Model |
|---------|----------|-------|
| Adult Census (`adult_tabnn`, `adult_2layernn`) | tabular | 2-layer NN / TabNN |
| Breast Cancer (`cancer_tabnn`, `cancer_2layernn`) | tabular | 2-layer NN / TabNN |
| SNLI (`snli_cnn`, `snli_2layernn`) | text | CNN / 2-layer NN |
| IMDb (`imdb_cnn`, `imdb_2layernn`) | text | CNN / 2-layer NN |
| CUB-200 (`cub_resnet`, `cub_densenet`) | vision | ResNet / DenseNet |
| STL-10 (`stl10_resnet`, `stl10_densenet`) | vision | ResNet / DenseNet |

Benchmark JSONs are in `dataset/{train,test}/{modality}/`. Model checkpoints are under `models_to_read/`.

### Out-of-distribution (OOD)

| Dataset | Modality | Model |
|---------|----------|-------|
| German Credit (`german_credit_3layernn`) | tabular | 3-layer NN |
| Yelp (`yelp_bert`) | text | BERT |
| CIFAR-10 (`cifar_resnet`) | vision | ResNet-18 |

OOD benchmarks are in `dataset_ood/`. Use `--dataset_variant ood` or explicit `--dataset_dir`/`--models_dir` overrides.

---

## Running the pipeline

### Single question

```bash
python MEA_pipeline.py \
    --dataset  dataset/test/vision/stl10_resnet_q1.json \
    --question_id 0 \
    --model_url  models_to_read/vision/stl10_resnet.pth \
    --vlm  tinker/Qwen/Qwen3.6-35B-A3B \
    --mode test \
    --output_dir outputs/
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--vlm` | — | VLM backend. Primary: `tinker/Qwen/Qwen3.6-35B-A3B`. API: `gemini-2.5-pro`, `claude-sonnet-4-5`, `gpt-4o`. Local: `Qwen/Qwen3-VL-8B-Instruct` |
| `--mode` | `test` | Dataset split (`train` or `test`) |
| `--no-eval` | off | Skip faithfulness evaluation |
| `--no-improvement` | off | Skip the improvement loop |
| `--temperature` | `0.0` | Sampling temperature |
| `--tinker_checkpoint` | — | Tinker LoRA checkpoint to evaluate (`tinker/<run_id>--<step>`) |
| `--tinker_lora_rank` | `32` | LoRA rank (must match training) |
| `--dataset_dir` / `--models_dir` | `./dataset` / `./models_to_read` | Root overrides for benchmark JSONs and model files |

### Batch runner

Shell interface:

```bash
./run_pipeline_batch.sh \
    --vlm tinker/Qwen/Qwen3.6-35B-A3B \
    --mode test \
    --datasets "stl10_resnet cub_resnet" \
    --q_types "1 2 3 4" \
    --question_ids "0-9" \
    --output_dir outputs/ \
    --parallel --max_jobs 4
```

Python interface:

```bash
python run_pipeline_batch.py \
    --vlm tinker/Qwen/Qwen3.6-35B-A3B \
    --mode test \
    --datasets stl10_resnet cub_resnet \
    --q_types 1 2 3 4 \
    --question_ids 0-9 \
    --output_dir outputs/ \
    --parallel --max_workers 4
```

Use `--modality {vision,text,tabular,all}` instead of `--datasets` to run all datasets for a modality. Use `--dry_run` to preview commands.

Config file:

```bash
python run_pipeline_batch.py --config batch_config_example.json
```

---

## SLURM

Edit `myjob.slurm` to set your datasets, question IDs, and VLM, then:

```bash
sbatch myjob.slurm
```

Logs go to `outputs/logs/<timestamp>/`.

---

## Training

MEA is trained with GRPO (Group Relative Policy Optimization) on `Qwen3.6-35B-A3B` via the Tinker SDK. The pipeline itself serves as the RL environment: faithfulness scores from the Critic are used as rewards to guide the Proposer and Actor.

Training uses LoRA (rank 32) with a hybrid strategy combining in-distribution and OOD questions.

```bash
python training/rl/train.py \
    --dataset_name stl10_resnet \
    --mode train \
    --q_types 1 2 3 \
    --model_name Qwen/Qwen3.6-35B-A3B \
    --output_dir checkpoints/grpo_vision \
    --num_rollouts 4 \
    --no-improvement
```

SLURM job templates: `training/rl/train_grpo_*.slurm`.

---

## Faithfulness Evaluation

Faithfulness is measured by a perturb-and-measure protocol: the evaluator masks or replaces the top-ranked features identified by the explanation and checks whether the model's prediction changes accordingly. A soft faithfulness score in [0, 1] is computed per question type.

Evaluators for Q1–Q10 are in `evaluation/explanation_faithfulness/`.

---

## Output structure

```
outputs/
  {dataset}_{qtype}_{question_id}/
    results.json            # full pipeline output (strategy, explanation, evaluation)
  training_data/            # GRPO training datapoints (passed faithfulness threshold)
    {modality}/{dataset}/{qtype}/{row_no}/training_datapoint.json
  logs/<timestamp>/         # per-job logs from batch runner
```

---

## Code structure

```
MEA_pipeline.py             # Main pipeline orchestrator (MEAPipeline class)
MEA_agent_system.py         # Three-agent system (Proposer, Actor, Critic)
run_pipeline_batch.py       # Python batch runner
run_pipeline_batch.sh       # Shell batch runner
agents/                     # Agent base classes
prompts/                    # Per-question-type prompt builders (Q1–Q10)
evaluation/                 # Faithfulness evaluators (Q1–Q10)
training/rl/                # GRPO training (env, trainer, dataset, rollout)
data/                       # Dataset loader scripts (local only, not tracked in git)
models_to_read/             # Pre-trained model checkpoints (local only)
```

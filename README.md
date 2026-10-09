# MEA: A Reward-Driven Multi-Agent System for Faithful Model Explanations

[![arXiv](https://img.shields.io/badge/arXiv-2610.02480-b31b1b.svg)](https://arxiv.org/abs/2610.02480)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MEA--Benchmark-yellow)](https://huggingface.co/datasets/EstherrrCheng/mea-benchmark)

Official code for **MEA** (Multimodal Explainability Agent), a multi-agent framework that turns post-hoc explanation tools into faithful natural-language explanations across tabular, text, and vision models. Paper: [arXiv:2610.02480](https://arxiv.org/abs/2610.02480) · Benchmark: [MEA-Benchmark on Hugging Face](https://huggingface.co/datasets/EstherrrCheng/mea-benchmark).

## Overview

MEA consists of two collaborating agents in a Proposer → Actor pipeline:

- **Proposer**: Selects and configures explanation tools based on the question and modality
- **Actor**: Turns the tool outputs into a natural-language explanation grounded in model behavior; it is optimized end-to-end against faithfulness

Faithfulness is evaluated by a perturbation-based protocol (not an agent). The benchmark has 10 question types (Q1–Q10) spanning feature attribution, counterfactual reasoning, and spurious feature detection, across tabular, text, and vision. Optimizing against faithfulness rewards with a modality-adaptive penalty improves faithfulness over the untrained backbone by +28% (tabular), +21% (text), and +34% (vision), and MEA outperforms post-hoc explainers, agentic, and closed-source baselines across six datasets.

### Question Types

| Type | Description |
|------|-------------|
| Q1 | Which part of the input was **most** responsible for the prediction? |
| Q2 | Which part of the input was **least** responsible for the prediction? |
| Q3 | Which parts distinguish the prediction from the next-best alternative? |
| Q4 | Why are instances A and B given different predictions? |
| Q5 | If a certain part is masked, would the prediction change? |
| Q6 | How should the instance change to flip the prediction to a target class? |
| Q7 | If one important part is removed/changed, how would the prediction change? |
| Q8 | Is there an irrelevant (spurious) part causing the wrong prediction? |
| Q9 | What shared feature makes multiple misclassified inputs difficult? |
| Q10 | Why are two similar instances given different predictions (one correct, one wrong)? |

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

The benchmark is hosted on [Hugging Face](https://huggingface.co/datasets/EstherrrCheng/mea-benchmark); download it into `dataset/` so that JSONs are in `dataset/{train,test}/{modality}/`. Model checkpoints are under `models_to_read/`.

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

Submit a batch evaluation job:

```bash
sbatch --job-name=mea_pipeline \
       --partition=gpu \
       --gres=gpu:a100:1 \
       --constraint=a100_80gb \
       --mem=128G \
       --cpus-per-task=8 \
       --time=24:00:00 \
       --wrap="python run_pipeline_batch.py \
           --vlm tinker/Qwen/Qwen3.6-35B-A3B \
           --datasets stl10_resnet cub_resnet \
           --q_types 1 2 3 4 \
           --output_dir outputs/"
```

Logs go to `outputs/logs/<timestamp>/`.

---

## Training

MEA is trained with GRPO (Group Relative Policy Optimization) on `Qwen3.6-35B-A3B` via the Tinker SDK. The pipeline itself serves as the RL environment: faithfulness scores are used as rewards to guide the Proposer and Actor.

Training uses LoRA (rank 32) with a hybrid strategy combining in-distribution and OOD questions.

```bash
sbatch --job-name=mea_grpo \
       --partition=gpu \
       --gres=gpu:a100:1 \
       --constraint=a100_80gb \
       --mem=256G \
       --cpus-per-task=16 \
       --time=48:00:00 \
       --wrap="python -m training.rl.train \
           --dataset_name stl10_resnet \
           --mode train \
           --q_types 1 2 3 \
           --model_name Qwen/Qwen3.6-35B-A3B \
           --output_dir checkpoints/grpo_vision \
           --num_rollouts 4 \
           --no-improvement"
```

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
MEA_agent_system.py         # Agent system (Proposer, Actor)
run_pipeline_batch.py       # Python batch runner
run_pipeline_batch.sh       # Shell batch runner
agents/                     # Agent base classes
prompts/                    # Per-question-type prompt builders (Q1–Q10)
evaluation/                 # Faithfulness evaluators (Q1–Q10)
training/rl/                # GRPO training (env, trainer, dataset, rollout)
models_to_read/             # Model checkpoints and loader scripts
```

---

## Citation

```bibtex
@article{cheng2026mea,
  title         = {{MEA}: A Reward-Driven Multi-Agent System for Faithful Model Explanations},
  author        = {Cheng, Yuyang and Ravi, Raghav Kaushik and Sridhar, Srivarshinee and Saha, Sriparna and Ghosh, Akash and Agarwal, Chirag},
  journal       = {arXiv preprint arXiv:2610.02480},
  year          = {2026},
  eprint        = {2610.02480},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

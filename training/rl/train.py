"""
GRPO Training Entry Point for XAI Agent (Tinker backend)

Wires together:
  1. XAIRLDataset           – questions loaded by dataset_name + mode + q_types
  2. XAIEnv / XAIMultiTurnEnv – MDP environment per question
  3. GRPOTrainer            – Tinker-based GRPO (forward_backward + optim_step)

Design of env_factory_fn
────────────────────────
`env_factory_fn(question)` is called K times per question to create K fresh
environments (one per rollout). Each env holds references to:
  - actor_agent  (XAI tool execution – local CPU/GPU)
  - critic_agent (faithfulness evaluation – local)
  - model_info   (target XAI model + data – local)

These are loaded ONCE per question outside the env and injected.
The Tinker VLM handles generation; local code handles everything else.

Usage
─────
    python -m training.rl.train \\
        --dataset_name  adult_2layernn \\
        --mode          train \\
        --q_types       1 2 3 \\
        --model_name    Qwen/Qwen3-VL-30B-A3B-Instruct \\
        --models_dir    models_to_read/tabular \\
        --output_dir    checkpoints/grpo_tabular \\
        --num_rollouts  4 \\
        --batch_size    4 \\
        --lr            1e-5 \\
        --kl_coef       0.01 \\
        --use_reflection        (flag to enable XAIMultiTurnEnv)

SLURM
─────
See training/rl/train_grpo.slurm for the SLURM job template.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.rl.dataset import XAIRLDataset
from training.rl.env import XAIEnv, XAIMultiTurnEnv
from training.rl.grpo_trainer import GRPOConfig, GRPOTrainer, compute_advantages


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_proposer_prompt(
    question: Dict[str, Any],
    question_template: Any,
    model_info: Optional[Dict] = None,
    prediction: Optional[Dict] = None,
    input_path: Optional[str] = None,
) -> str:
    """
    Build the Proposer turn-0 prompt.

    Uses the existing prompt builder from `prompts/` if available,
    otherwise falls back to a compact template.
    """
    try:
        from question_templates_new import get_prompt_builder
        pb = get_prompt_builder(
            q_type=question.get("q_type", 1),
            modality=question.get("modality", "tabular"),
        )
        return pb.build_proposer_prompt(
            question=question,
            model_info=model_info or {},
            prediction=prediction or {},
        )
    except Exception:
        pass

    # Fallback compact prompt
    q_type   = question.get("q_type", 1)
    modality = question.get("modality", "tabular")
    features = question.get("features", {})
    pred_cls = (prediction or {}).get("predicted_class", "unknown")
    dataset  = question.get("dataset", "unknown")

    tool_list = (
        "lime_tabular, shap_tabular (tabular) | "
        "gradcam, lime_image, occlusion (vision) | "
        "lime_text, shap_text (text)"
    )

    return f"""You are an XAI strategy planner.

TASK: Generate an XAI analysis strategy for the question below.

Question Type: Q{q_type}
Modality:      {modality}
Dataset:       {dataset}
Question:      {question.get('question', question.get('example', ''))}
Features:      {json.dumps(features, ensure_ascii=False)[:400]}
Model prediction: {pred_cls}

Available tools: {tool_list}

Output a JSON strategy with this exact schema:
{{
  "strategy_type": "<reasoning_type>",
  "selected_tools": [
    {{"tool_name": "<name>", "parameters": {{...}}, "rationale": "<why>"}}
  ],
  "rationale": "<overall_rationale>"
}}

Output ONLY the JSON block, no extra text."""


def build_actor_prompt(
    question: Dict[str, Any],
    question_template: Any,
    strategy: Dict[str, Any],
    tool_results: Dict[str, Any],
    model_info: Optional[Dict] = None,
    prediction: Optional[Dict] = None,
) -> tuple:
    """
    Build the Actor turn-1 prompt + extract any visualization images.

    Returns: (prompt_text: str, images: list)
    """
    try:
        from question_templates_new import get_prompt_builder
        pb = get_prompt_builder(
            q_type=question.get("q_type", 1),
            modality=question.get("modality", "tabular"),
        )
        text = pb.build_actor_prompt(
            question=question,
            strategy=strategy,
            tool_results=tool_results,
            model_info=model_info or {},
            prediction=prediction or {},
        )
        images = _load_images_from_paths(tool_results.get("visualization_paths", []))
        return text, images
    except Exception:
        pass

    # Fallback compact prompt
    results_summary  = tool_results.get("tool_results_summary", "No tool results.")
    tool_results_text = json.dumps(
        tool_results.get("tool_results", {}), ensure_ascii=False
    )[:800]

    images = _load_images_from_paths(tool_results.get("visualization_paths", []))

    text = f"""You are an XAI explanation expert.

TASK: Generate a structured explanation based on the strategy and tool results below.

Question: {question.get('question', question.get('example', ''))}
Question Type: Q{question.get('q_type', 1)}
Strategy used: {json.dumps(strategy, ensure_ascii=False)[:300]}

Tool Results Summary: {results_summary}
Tool Results (detail): {tool_results_text}

Generate the final explanation in the format required for Q{question.get('q_type', 1)}.
Be specific: refer to feature names, importance values, and how they affect the prediction.
"""
    return text, images


def build_reflection_proposer_prompt(
    reflection_text: str,
    question: Dict[str, Any],
    question_template: Any,
    model_info: Optional[Dict] = None,
    prediction: Optional[Dict] = None,
    original_strategy: Optional[Dict] = None,
    input_path: Optional[str] = None,
) -> str:
    """
    Build the improved-Proposer prompt for Turn 2 (after reflection).
    """
    original = build_proposer_prompt(
        question, question_template, model_info, prediction, input_path
    )
    return (
        f"{original}\n\n"
        f"{'─'*60}\n"
        f"{reflection_text}\n"
        f"{'─'*60}\n\n"
        f"Your previous strategy:\n"
        f"{json.dumps(original_strategy or {}, ensure_ascii=False)[:400]}\n\n"
        f"Based on the critic feedback above, output an IMPROVED JSON strategy."
    )


def _load_images_from_paths(viz_paths: List[str]) -> List[Any]:
    """Load PIL images from visualization paths."""
    images = []
    try:
        from PIL import Image
        for p in viz_paths:
            if p and Path(p).exists():
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
    except ImportError:
        pass
    return images


# ── Question setup ────────────────────────────────────────────────────────────

def setup_question(
    question: Dict[str, Any],
    models_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Load the target XAI model + data for one question.

    Follows xai_pipeline_v2.py pattern:
      loader = DataModelLoader(model_name=dataset_name, modality=modality)
      data   = loader.load_sample(index=row_no, split=split)
      pred   = loader.predict(loader.get_current_input())

    model_name = question["dataset_name"] (injected by XAIRLDataset.from_dataset_name).
    This is the same as the .pth file stem, e.g. "adult_2layernn".
    """
    from agents.actor_agent import ActorAgent
    from agents.critic_agent import CriticAgent
    from question_templates_new import get_question_template

    modality = question.get("modality", "tabular")
    q_type   = question.get("q_type", 1)
    row_no   = question.get("row_no", 0)

    # VLM (Tinker) is called during rollout; actor/critic use local XAI tools only.
    class _NoVLM:
        pass

    actor_agent  = ActorAgent(vlm=_NoVLM(), output_dir=output_dir)
    critic_agent = CriticAgent(vlm=_NoVLM(), output_dir=output_dir)

    loader        = None
    model_info    = None
    current_input = None
    prediction    = None
    input_path    = None

    model_name = _infer_model_name(question)
    if model_name:
        try:
            from DataModelLoader import DataModelLoader
            import torch

            loader = DataModelLoader(model_name=model_name, modality=modality)

            model      = loader.get_model()
            processor  = loader.get_processor()
            loader_mod = loader.loader_module

            # Build model_info matching xai_pipeline_v2.py structure
            extra_info: Dict[str, Any] = {}
            if hasattr(loader_mod, "get_model_info"):
                extra_info = loader_mod.get_model_info(model) or {}

            model_info = {
                "success":        True,
                "model":          model,
                "processor":      processor,
                "model_name":     model_name,
                "model_type":     "local_pth",
                "device":         str(getattr(loader_mod, "DEVICE", "cpu")),
                "architecture":   model.__class__.__name__,
                "num_classes":    extra_info.get("num_classes"),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "label_map":      extra_info.get("label_map", {}),
                "feature_names":  extra_info.get("feature_names") or [],
            }

            # Tabular: override feature_names from question if available
            if modality == "tabular":
                feats = question.get("features", {})
                if isinstance(feats, dict) and feats:
                    model_info["feature_names"] = list(feats.keys())

            # Determine split (cancer Q4/Q8-Q10 use full dataset)
            split = "test"
            if "cancer" in model_name and q_type in (4, 8, 9, 10):
                split = "full"

            # Load sample and make prediction
            sample_data   = loader.load_sample(index=row_no, split=split)
            current_input = loader.get_current_input()

            if current_input is not None:
                prediction = loader.predict(current_input)

            # Vision: grab image path
            if modality == "vision" and sample_data:
                input_path = sample_data.get("image_path")

            # Wire up tools and critic
            actor_agent.initialize_tools(data_model_loader=loader)
            if hasattr(critic_agent, "set_model"):
                critic_agent.set_model(model)

        except Exception as e:
            print(f"  [setup_question] Warning: DataModelLoader failed for "
                  f"'{model_name}' ({modality}): {e}")
            import traceback; traceback.print_exc()

    question_template = get_question_template(q_type, modality)

    return {
        "actor_agent":       actor_agent,
        "critic_agent":      critic_agent,
        "loader":            loader,
        "model_info":        model_info,
        "input_tensor":      current_input,
        "prediction":        prediction,
        "input_path":        input_path,
        "question_template": question_template,
    }


def _infer_model_name(question: Dict) -> Optional[str]:
    """
    Return the DataModelLoader model_name for a question.

    Primary source: question["dataset_name"] injected by
    XAIRLDataset.from_dataset_name() — this is exactly the .pth stem
    (e.g. "adult_2layernn", "stl10_resnet", "imdb_cnn").

    Fallback: pattern-match from the human-readable "dataset" field.
    """
    # Best source: injected by from_dataset_name()
    ds_name = question.get("dataset_name")
    if ds_name:
        return str(ds_name)

    # Fallback heuristics from the human-readable "dataset" field
    dataset  = str(question.get("dataset", "")).lower()
    modality = question.get("modality", "tabular")

    if modality == "tabular":
        if "adult"  in dataset: return "adult_2layernn"
        if "cancer" in dataset or "breast" in dataset: return "cancer_2layernn"
    elif modality == "vision":
        if "stl"    in dataset: return "stl10_resnet"
        if "cub"    in dataset: return "cub_resnet"
    elif modality == "text":
        if "imdb"   in dataset: return "imdb_cnn"
        if "snli"   in dataset: return "snli_cnn"
    return None


# ── env_factory_fn ─────────────────────────────────────────────────────────────

def make_env_factory(
    models_dir: str,
    output_dir: str,
    faithfulness_threshold: float = 0.5,
    use_reflection: bool = False,
):
    """
    Returns a closure: `env_factory_fn(question) -> XAIEnv`.

    Called once per rollout per question. Loads the target XAI model/data
    and returns a fresh env backed by local actor/critic agents.
    The Tinker VLM is invoked separately during rollout (not inside the env).
    """
    def env_factory_fn(question: Dict[str, Any]) -> XAIEnv:
        ctx = setup_question(
            question=question,
            models_dir=models_dir,
            output_dir=output_dir,
        )

        common_kwargs = dict(
            question=question,
            question_template=ctx["question_template"],
            actor_agent=ctx["actor_agent"],
            critic_agent=ctx["critic_agent"],
            proposer_prompt_fn=build_proposer_prompt,
            actor_prompt_fn=build_actor_prompt,
            faithfulness_threshold=faithfulness_threshold,
            input_path=ctx["input_path"],
            model_info=ctx["model_info"],
            prediction=ctx["prediction"],
            input_tensor=ctx["input_tensor"],
        )

        if use_reflection:
            return XAIMultiTurnEnv(
                **common_kwargs,
                reflection_proposer_prompt_fn=build_reflection_proposer_prompt,
                max_reflection_turns=1,
            )
        return XAIEnv(**common_kwargs)

    return env_factory_fn


# ── Tinker client setup ────────────────────────────────────────────────────────

async def setup_tinker_clients(
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    lora_rank: int = 32,
    load_checkpoint_path: Optional[str] = None,
):
    """
    Async: initialise Tinker ServiceClient + LoRA TrainingClient + Tokenizer.

    Reads TINKER_API_KEY from the environment.

    Args:
        model_name:            Tinker model ID
        lora_rank:             LoRA rank for create_lora_training_client_async
        load_checkpoint_path:  Optional path to resume from a saved state

    Returns:
        (training_client, tokenizer)
    """
    import tinker

    print(f"Setting up Tinker clients for model: {model_name}")
    print(f"  TINKER_API_KEY: {'set' if os.environ.get('TINKER_API_KEY') else 'NOT SET'}")

    service_client = tinker.ServiceClient()

    print(f"  Creating LoRA TrainingClient (lora_rank={lora_rank})...")
    if load_checkpoint_path:
        # Resume from saved state (weights + optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            load_checkpoint_path
        )
        print(f"  Loaded checkpoint: {load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            model_name, rank=lora_rank
        )
    print(f"  TrainingClient ready.")

    # Tokenizer comes from Tinker (no local HuggingFace download needed)
    tokenizer = training_client.get_tokenizer()
    print(f"  Tokenizer ready.")

    return training_client, tokenizer


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO Training for XAI Agent (Tinker)")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--dataset_name", type=str, nargs="+", required=True,
                   help="One or more dataset names, e.g. adult_2layernn cancer_tabnn "
                        "(modality auto-inferred per name)")
    p.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                   help="Dataset split: 'train' or 'test'")
    p.add_argument("--q_types", type=int, nargs="+", default=None,
                   help="Question types to load, e.g. --q_types 1 2 3. None → all Q1-Q10.")
    p.add_argument("--max_questions", type=int, default=None,
                   help="Cap dataset size (useful for debugging)")
    p.add_argument("--eval_ratio", type=float, default=0.1,
                   help="Fraction of questions held out for eval")

    # ── Model (Tinker) ────────────────────────────────────────────────────────
    p.add_argument("--model_name", type=str,
                   default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                   help="Tinker model name")
    p.add_argument("--lora_rank", type=int, default=32,
                   help="LoRA rank for create_lora_training_client_async")
    p.add_argument("--load_checkpoint", type=str, default=None,
                   help="Resume from a Tinker checkpoint path (weights + optimizer state)")

    # ── Target model (local XAI) ───────────────────────────────────────────────
    p.add_argument("--models_dir", type=str, default="models_to_read",
                   help="Directory with target XAI model loaders")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",      type=str,   default="checkpoints/grpo")
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--kl_coef",         type=float, default=0.01)
    p.add_argument("--num_rollouts",    type=int,   default=4)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_epochs",      type=int,   default=3)
    p.add_argument("--max_new_tokens",  type=int,   default=512)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--save_every",      type=int,   default=50)
    p.add_argument("--eval_every",      type=int,   default=20)

    # ── Environment ───────────────────────────────────────────────────────────
    p.add_argument("--use_reflection", action="store_true",
                   help="Enable XAIMultiTurnEnv (reflection + regeneration)")
    p.add_argument("--faithfulness_threshold", type=float, default=0.5)

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()
    import random
    random.seed(args.seed)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    if len(args.dataset_name) == 1:
        dataset = XAIRLDataset.from_dataset_name(
            dataset_name=args.dataset_name[0],
            mode=args.mode,
            q_types=args.q_types,
            max_questions=args.max_questions,
            shuffle=True,
            seed=args.seed,
        )
    else:
        dataset = XAIRLDataset.from_multiple_datasets(
            dataset_names=args.dataset_name,
            mode=args.mode,
            q_types=args.q_types,
            max_questions=args.max_questions,
            shuffle=True,
            seed=args.seed,
        )
    train_ds, eval_ds = dataset.split(eval_ratio=args.eval_ratio, seed=args.seed)

    # ── 2. Set up Tinker clients (async) ──────────────────────────────────────
    training_client, tokenizer = await setup_tinker_clients(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        load_checkpoint_path=args.load_checkpoint,
    )

    # ── 3. Build GRPO config ──────────────────────────────────────────────────
    cfg = GRPOConfig(
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        eval_every=args.eval_every,
        lora_rank=args.lora_rank,
        output_dir=args.output_dir,
    )

    # ── 4. Build env factory ──────────────────────────────────────────────────
    env_factory_fn = make_env_factory(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        faithfulness_threshold=args.faithfulness_threshold,
        use_reflection=args.use_reflection,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        training_client=training_client,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    async def eval_fn(training_client, step, sampling_client):
        """Near-greedy eval using the current sampling_client passed from fit()."""
        from training.rl.rollout import do_rollout

        eval_questions = list(eval_ds.questions)[:min(8, len(eval_ds))]
        rewards = []

        for question in eval_questions:
            env = env_factory_fn(question)
            try:
                traj = do_rollout(
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    env=env,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=0.1,
                )
                rewards.append(traj.total_reward)
            except Exception as e:
                print(f"  Eval error: {e}")

        if rewards:
            print(
                f"  [Eval step={step}] "
                f"mean_reward={sum(rewards)/len(rewards):.4f}  "
                f"n={len(rewards)}"
            )

    await trainer.fit(train_ds, env_factory_fn=env_factory_fn, eval_fn=eval_fn)


if __name__ == "__main__":
    asyncio.run(main())

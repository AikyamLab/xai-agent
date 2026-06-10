"""
GRPO Training Entry Point for XAI Agent (Tinker backend).

Uses MEAPipeline directly for all XAI logic: prompt building, tool
execution, Q-type routing (Q4 contrastive, Q9/Q10 multi-instance,
Q1-Q3/Q5-Q8 standard), improvement loop, and strategy faithfulness
evaluation.

RLSamplingVLM is injected as the pipeline's VLM backend so every VLM
call (proposer strategy generation, actor feature extraction + explanation
generation) is captured as an RLTransition for GRPO gradient computation.

Flow:
  1. Load dataset (XAIRLDataset — stores _source_path per question)
  2. Set up Tinker TrainingClient + Tokenizer
  3. Create RLSamplingVLM (Tinker-backed, records transitions)
  4. Create MEAPipeline(vlm=rl_vlm, ...) — pipeline owns all XAI logic
  5. Create XAIRLEnv(pipeline, rl_vlm, ...) — thin RL wrapper
  6. Run GRPOTrainer.fit(dataset, env, ...)

Pipeline behaviour flags (forwarded to pipeline.run()):
  --no-improvement    Disable proposer/actor improvement loop.
                      Recommended for GRPO: improvement VLM calls would add
                      extra transitions with a mixed reward signal.

Usage:
    python -m training.rl.train \\
        --dataset_name adult_2layernn \\
        --mode         train \\
        --q_types      1 2 3 \\
        --model_name   Qwen/Qwen3-VL-30B-A3B-Instruct \\
        --output_dir   checkpoints/grpo_tabular \\
        --num_rollouts 4 \\
        --no-improvement
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.rl.dataset   import XAIRLDataset, parse_data_range
from training.rl.env       import XAIRLEnv
from training.rl.rl_vlm    import RLSamplingVLM
from training.rl.rollout   import do_group_rollout
from training.rl.grpo_trainer import GRPOConfig, GRPOTrainer


# ── Tinker client setup ────────────────────────────────────────────────────────

async def setup_tinker_clients(
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    lora_rank: int = 32,
    load_checkpoint_path: Optional[str] = None,
):
    """
    Initialise Tinker ServiceClient, LoRA TrainingClient, and Tokenizer.

    Returns:
        (training_client, tokenizer)
    """
    import tinker

    print(f"Setting up Tinker clients for model: {model_name}")
    print(f"  TINKER_API_KEY: {'set' if os.environ.get('TINKER_API_KEY') else 'NOT SET'}")

    service_client = tinker.ServiceClient()

    print(f"  Creating LoRA TrainingClient (lora_rank={lora_rank})...")
    if load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            load_checkpoint_path
        )
        print(f"  Loaded checkpoint: {load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            model_name, rank=lora_rank
        )
    print("  TrainingClient ready.")

    tokenizer = training_client.get_tokenizer()
    print("  Tokenizer ready.")

    return training_client, tokenizer


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO Training for XAI Agent (Tinker)")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--dataset_name", type=str, nargs="+", required=True,
                   help="One or more dataset names, e.g. adult_2layernn cancer_tabnn")
    p.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    p.add_argument("--q_types", type=int, nargs="+", default=None,
                   help="Question types to load, e.g. --q_types 1 2 3. None → all Q1-Q10.")
    p.add_argument("--max_questions", type=int, default=None,
                   help="Cap dataset size (useful for debugging)")
    p.add_argument("--per_dataset_q_types", type=str, nargs="*", default=None,
                   metavar="DATASET:Q1,Q2,...",
                   help="Per-dataset q_type overrides, e.g. "
                        "cancer_tabnn:1,3,5,6,7,9 cancer_2layernn:1,3,5,6,7,9")
    p.add_argument("--data_range", type=str, default="all",
                   help="Per-dataset question range applied before merging. "
                        "'all' → no limit (default). "
                        "'START-END' → take questions at interleaved indices "
                        "START..END-1 from each dataset (balanced across Q-types). "
                        "E.g. '0-100' uses 20 questions per Q-type per dataset "
                        "when there are 5 Q-types.")
    p.add_argument("--eval_ratio", type=float, default=0.1,
                   help="Fraction of questions held out for eval")

    # ── Tinker model ──────────────────────────────────────────────────────────
    p.add_argument("--model_name", type=str,
                   default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                   help="Tinker model name")
    p.add_argument("--lora_rank", type=int, default=32,
                   help="LoRA rank for create_lora_training_client_async")
    p.add_argument("--load_checkpoint", type=str, default=None,
                   help="Resume from a Tinker checkpoint path")
    p.add_argument("--start_step", type=int, default=0,
                   help="Skip the first N steps (use with --load_checkpoint to resume training). "
                        "E.g. --load_checkpoint checkpoint_step_25 --start_step 26")

    # ── Pipeline directories ──────────────────────────────────────────────────
    p.add_argument("--output_dir",  type=str, default="checkpoints/grpo",
                   help="Output directory for logs and checkpoints")
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="XAI benchmark dataset dir (default: <project_root>/dataset)")
    p.add_argument("--models_dir",  type=str, default=None,
                   help="XAI model .pth dir (default: <project_root>/models_to_read)")

    # ── Pipeline behaviour flags ───────────────────────────────────────────────
    p.add_argument("--no-improvement", dest="enable_improvement",
                   action="store_false", default=True,
                   help="Disable proposer/actor improvement loop "
                        "(recommended for GRPO to avoid mixed-reward transitions)")
    p.add_argument("--no-tool-penalty", dest="use_tool_penalty",
                   action="store_false", default=True,
                   help="Disable L1 tool-count penalty from the reward signal")
    p.add_argument("--tool-diversity-lambda", dest="diversity_lambda",
                   type=float, default=0.0,
                   help="Coefficient for the tool-diversity bonus "
                        "(+lambda * n_unique_tools / n_max). "
                        "0.0 = disabled (default). "
                        "Positive counterpart of the L1 tool-count penalty: "
                        "rewards using a wider variety of distinct tools.")

    # ── GRPO training ─────────────────────────────────────────────────────────
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--kl_coef",        type=float, default=0.0)
    p.add_argument("--num_rollouts",   type=int,   default=4)
    p.add_argument("--high_k_qtypes",  type=int,   nargs="+", default=[1, 3, 8, 9],
                   help="Q-types that use --high_k rollouts instead of --num_rollouts. "
                        "Default: 1 3 8 9 (hard tabular types).")
    p.add_argument("--high_k",         type=int,   default=8,
                   help="Rollout count for Q-types listed in --high_k_qtypes. Default: 8.")
    p.add_argument("--no-stratify",    dest="stratify_by_qtype",
                   action="store_false", default=True,
                   help="Disable stratified-by-Q-type batch sampling (use plain shuffle).")
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--num_epochs",     type=int,   default=3)
    p.add_argument("--max_new_tokens", type=int,   default=512)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--save_every",     type=int,   default=50)
    p.add_argument("--eval_every",     type=int,   default=20)
    p.add_argument("--eval_max_questions", type=int, default=None,
                   help="Max eval questions per eval callback. None → use all eval questions.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--num_workers",    type=int,   default=1,
                   help="Parallel rollout workers per batch step (default: 1 = sequential). "
                        "Each worker gets its own independent env from a pre-created pool. "
                        "Recommend: set equal to batch_size for full question-level parallelism.")
    p.add_argument("--parallel-rollouts", dest="parallel_rollouts",
                   action="store_true", default=False,
                   help="Run K rollouts per question in parallel. Each rollout gets its own "
                        "env from a pre-created rollout pool. Pool size is controlled by "
                        "--max-rollout-workers (0 = auto: batch_size × max_K).")
    p.add_argument("--max-rollout-workers", dest="max_rollout_workers",
                   type=int, default=0,
                   help="Rollout env pool size when --parallel-rollouts is set. "
                        "0 = auto (batch_size × max_K). Caps total concurrent Tinker "
                        "API calls from rollouts. E.g. 64 = at most 64 simultaneous rollouts.")
    p.add_argument("--no-think", dest="enable_thinking",
                   action="store_false", default=True,
                   help="Pre-fill empty <think></think> block so the model skips "
                        "chain-of-thought reasoning and outputs the answer directly.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()
    import random
    random.seed(args.seed)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    data_range = parse_data_range(args.data_range)
    if data_range is not None:
        start, end = data_range
        print(f"  [data_range] Using questions {start}–{end-1} per dataset "
              f"(balanced across Q-types, ~{(end-start)//max(len(args.q_types or [1]),1)} per Q-type).")

    # Parse per_dataset_q_types: ["cancer_tabnn:1,3,5,6,7,9", ...] → dict
    per_dataset_q_types = None
    if args.per_dataset_q_types:
        per_dataset_q_types = {}
        for entry in args.per_dataset_q_types:
            ds_name, qt_str = entry.split(":")
            per_dataset_q_types[ds_name.strip()] = [int(x) for x in qt_str.split(",")]
        print(f"  [per_dataset_q_types] {per_dataset_q_types}")

    if len(args.dataset_name) == 1:
        dataset = XAIRLDataset.from_dataset_name(
            dataset_name=args.dataset_name[0],
            mode=args.mode,
            q_types=args.q_types,
            base_dir=args.dataset_dir,
            max_questions=args.max_questions,
            data_range=data_range,
            shuffle=True,
            seed=args.seed,
        )
    else:
        dataset = XAIRLDataset.from_multiple_datasets(
            dataset_names=args.dataset_name,
            mode=args.mode,
            q_types=args.q_types,
            per_dataset_q_types=per_dataset_q_types,
            base_dir=args.dataset_dir,
            max_questions=args.max_questions,
            data_range=data_range,
            shuffle=True,
            seed=args.seed,
        )
    train_ds, eval_ds = dataset.split(eval_ratio=args.eval_ratio, seed=args.seed)

    # ── 2. Set up Tinker clients ───────────────────────────────────────────────
    training_client, tokenizer = await setup_tinker_clients(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        load_checkpoint_path=args.load_checkpoint,
    )

    # ── 3. Get initial sampling client ────────────────────────────────────────
    print("Getting initial sampling client...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # ── 4. Create RLSamplingVLM ───────────────────────────────────────────────
    # This VLM is injected into MEAPipeline so every VLM call is recorded
    # as an RLTransition for GRPO gradient computation.
    rl_vlm = RLSamplingVLM(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )

    # ── 5. Create MEAPipeline with rl_vlm injected ─────────────────────────
    # The pipeline handles all XAI logic: prompt building, tool execution,
    # Q4/Q9/Q10 routing, improvement loop, SF evaluation, critic evaluation.
    from MEA_pipeline import MEAPipeline
    from prompts.output_size_config import OutputSizeConfig
    _output_size_cfg = OutputSizeConfig(
        fixed_percentage=0.25,
        apply_to_tabular=True,
        apply_to_text=True,
        apply_to_vision=False,
    )
    pipeline = MEAPipeline(
        vlm=rl_vlm,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        mode=args.mode,
        output_size_config=_output_size_cfg,
    )

    # ── 6. Create RL environment ──────────────────────────────────────────────
    env = XAIRLEnv(
        pipeline=pipeline,
        rl_vlm=rl_vlm,
        enable_improvement=args.enable_improvement,
        use_tool_penalty=args.use_tool_penalty,
        diversity_lambda=args.diversity_lambda,
    )

    # ── 6b. Env factory for parallel rollout worker pool ──────────────────────
    # Each worker in the pool owns an independent (RLSamplingVLM + pipeline +
    # env) triple.  They all share the same `sampling_client` reference (safe:
    # Tinker's client is I/O-bound and thread-safe) and `tokenizer` (read-only).
    # The mutable per-episode state (_transitions list) lives inside each
    # worker's own RLSamplingVLM instance, so there is no shared mutable state.
    def make_env() -> XAIRLEnv:
        """Create a fresh (rl_vlm, pipeline, env) for one rollout pool slot."""
        w_rl_vlm = RLSamplingVLM(
            sampling_client=sampling_client,   # updated per-rollout by do_group_rollout
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        w_pipeline = MEAPipeline(
            vlm=w_rl_vlm,
            output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
            models_dir=args.models_dir,
            mode=args.mode,
            output_size_config=_output_size_cfg,
        )
        return XAIRLEnv(
            pipeline=w_pipeline,
            rl_vlm=w_rl_vlm,
            enable_improvement=args.enable_improvement,
            use_tool_penalty=args.use_tool_penalty,
            diversity_lambda=args.diversity_lambda,
        )

    # ── 7. Build GRPO config + trainer ────────────────────────────────────────
    rollouts_override = {qt: args.high_k for qt in (args.high_k_qtypes or [])}
    cfg = GRPOConfig(
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        start_step=args.start_step,
        num_rollouts=args.num_rollouts,
        rollouts_override=rollouts_override,
        stratify_by_qtype=args.stratify_by_qtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        eval_every=args.eval_every,
        lora_rank=args.lora_rank,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        parallel_rollouts=args.parallel_rollouts,
        max_rollout_workers=args.max_rollout_workers,
    )
    # env_factory is needed whenever any form of parallelism uses a pool:
    # - batch parallelism (num_workers > 1)
    # - rollout parallelism (parallel_rollouts=True, even with num_workers=1)
    need_factory = args.num_workers > 1 or args.parallel_rollouts
    trainer = GRPOTrainer(
        training_client=training_client,
        tokenizer=tokenizer,
        cfg=cfg,
        env_factory=make_env if need_factory else None,
    )

    # ── 8. Eval callback ──────────────────────────────────────────────────────
    async def eval_fn(training_client, step, sampling_client):
        """
        Near-greedy eval using the current policy snapshot.

        Runs in parallel when trainer._rollout_pool is available (i.e. when
        --parallel-rollouts is set), using the same env pool as training.
        Falls back to sequential eval using the primary env otherwise.

        Questions are randomly sampled each eval so the metric is unbiased
        across Q-types even when eval_max_questions < len(eval_ds).

        Returns a metrics dict that grpo_trainer writes to grpo_log.jsonl
        with type="eval".  Per-question rewards and q_type breakdown are
        included so convergence can be analysed per question type.
        """
        import math as _math
        import concurrent.futures as _cf

        eval_questions = list(eval_ds.questions)
        random.shuffle(eval_questions)
        if args.eval_max_questions is not None:
            eval_questions = eval_questions[:args.eval_max_questions]

        rewards   = []
        per_q     = []   # list of {q_type, row_no, reward, n_transitions}
        n_errors  = 0

        if trainer._rollout_pool is not None:
            # ── Parallel eval: reuse rollout env pool ─────────────────────────
            # Each worker checks out an env, sets temperature=0.1, runs the
            # episode, restores temperature, and returns the env to the pool.
            def _run_eval(question):
                w_env = trainer._rollout_pool.get()
                old_temp = w_env.rl_vlm.temperature
                try:
                    w_env.rl_vlm.temperature = 0.1
                    w_env.rl_vlm.update_sampling_client(sampling_client)
                    traj = w_env.run_episode(question, rollout_id=0)
                    return traj, None
                except Exception as exc:
                    return None, exc
                finally:
                    w_env.rl_vlm.temperature = old_temp
                    trainer._rollout_pool.put(w_env)

            n_workers = min(len(eval_questions), trainer._rollout_max_workers or 32)
            print(f"  [Eval step={step}] Running {len(eval_questions)} questions "
                  f"with {n_workers} parallel workers...")
            with _cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
                future_to_q = {ex.submit(_run_eval, q): q for q in eval_questions}
                for fut in _cf.as_completed(future_to_q):
                    question = future_to_q[fut]
                    try:
                        traj, err = fut.result()
                        if err:
                            print(f"  Eval error: {err}")
                            n_errors += 1
                        elif traj is not None:
                            reward = traj.total_reward
                            rewards.append(reward)
                            per_q.append({
                                "q_type":        question.get("q_type"),
                                "row_no":        question.get("row_no"),
                                "dataset":       question.get("dataset_name"),
                                "reward":        round(reward, 6),
                                "n_transitions": len(traj.transitions),
                            })
                    except Exception as e:
                        print(f"  Eval error: {e}")
                        n_errors += 1

        else:
            # ── Sequential eval: use primary env ──────────────────────────────
            from training.rl.rollout import do_group_rollout as _rollout
            original_temp = env.rl_vlm.temperature
            env.rl_vlm.temperature = 0.1   # near-greedy for eval

            for question in eval_questions:
                try:
                    group = _rollout(
                        env=env,
                        question=question,
                        sampling_client=sampling_client,
                        num_rollouts=1,
                    )
                    traj   = group.trajectories[0]
                    reward = traj.total_reward
                    rewards.append(reward)
                    per_q.append({
                        "q_type":        question.get("q_type"),
                        "row_no":        question.get("row_no"),
                        "dataset":       question.get("dataset_name"),
                        "reward":        round(reward, 6),
                        "n_transitions": len(traj.transitions),
                    })
                except Exception as e:
                    print(f"  Eval error: {e}")
                    n_errors += 1

            env.rl_vlm.temperature = original_temp

        if not rewards:
            return None

        mean_r = sum(rewards) / len(rewards)
        var_r  = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r  = _math.sqrt(var_r)

        # Per-q_type breakdown
        by_qtype: dict = {}
        for entry in per_q:
            qt = entry["q_type"]
            by_qtype.setdefault(qt, []).append(entry["reward"])
        qtype_means = {
            f"qtype_{qt}_mean_reward": round(sum(v) / len(v), 6)
            for qt, v in by_qtype.items()
        }

        print(
            f"  [Eval step={step}] "
            f"mean={mean_r:.4f}  std={std_r:.4f}  "
            f"n={len(rewards)}  errors={n_errors}"
        )

        return {
            "eval_mean_reward": mean_r,
            "eval_std_reward":  std_r,
            "eval_max_reward":  max(rewards),
            "eval_min_reward":  min(rewards),
            "eval_n":           len(rewards),
            "eval_n_errors":    n_errors,
            "eval_per_question": per_q,
            **qtype_means,
        }

    # ── 9. Train ──────────────────────────────────────────────────────────────
    await trainer.fit(
        dataset=train_ds,
        env=env,
        sampling_client=sampling_client,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    asyncio.run(main())

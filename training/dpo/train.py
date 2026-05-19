"""
DPO Training Loop for XAI ProposerAgent (reflection mode) using Tinker SDK.

Implements trajectory-level DPO with cross-instance B×C pairs and quality weighting.

  chosen   = Type B full trajectory → improved strategy (positive delta)
  rejected = Type C full trajectory → degraded strategy (negative delta)

Each pair has separate prompt_chosen / prompt_rejected (self-consistent per instance).
Quality weights: high pairs (delta>0.3, imp_faith>0.5) → weight=2.0, medium → weight=1.0

How it works:
  1. Load (prompt_chosen, prompt_rejected, chosen, rejected, weight) pairs from JSONL
  2. Tokenize each side with its own prompt, build Datum with loss mask
  3. Compute reference logprobs via ref_sampler.compute_logprobs()
  4. Run forward_backward_custom() with quality-weighted DPO loss
  5. Call optim_step() to update LoRA weights
  6. Repeat; save checkpoint every N steps

Usage:
  python train.py --model Qwen/Qwen3-4B-Instruct-2507 --batch_size 4 --steps 200

Env:
  TINKER_API_KEY  (required)
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import tinker
import tinker.types as tt

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
RUNS_DIR = BASE_DIR / "training/dpo/runs"

# ── System prompt (must match prepare_data.py) ─────────────────────────────────
# Minimal — full task description lives in the user message (via PromptBuilder),
# matching how ProposerAgent calls invoke_vlm_for_json at inference time.
SYSTEM_PROMPT = "You are a helpful AI assistant."

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Tokenization ───────────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, user: str, assistant: str) -> list[int]:
    """Encode full conversation (system + user + assistant) to token ids."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )


def get_prompt_len(tokenizer, user: str) -> int:
    """Return number of tokens in the prompt (system + user + assistant header)."""
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]
    return len(tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    ))


def make_datum(
    full_tokens: list[int],
    prompt_len: int,
    max_length: int,
) -> tuple[tt.Datum, list[int]]:
    """
    Build a tinker.Datum for one sequence.

      model_input      = full_tokens[:-1]          (input to model)
      target_tokens    = full_tokens[1:]            (what model predicts)
      weights          = loss mask: 0=prompt, 1=response

    Also returns the (possibly truncated) full_tokens for reference logprob queries.
    """
    if len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]

    N = len(full_tokens)
    target_tokens = full_tokens[1:]
    T = len(target_tokens)

    # In target space (shifted by 1): response starts at index (prompt_len - 1)
    resp_start = max(0, prompt_len - 1)
    weights    = [0.0] * resp_start + [1.0] * max(0, T - resp_start)

    datum = tt.Datum(
        model_input=tt.ModelInput.from_ints(full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tt.TensorData(
                data=target_tokens, dtype="int64", shape=[T]
            ),
            "weights": tt.TensorData(
                data=weights, dtype="float32", shape=[T]
            ),
        },
    )
    return datum, full_tokens


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_batches(records: list[dict], batch_size: int, seed: int = 0):
    idxs = list(range(len(records)))
    random.seed(seed)
    random.shuffle(idxs)
    for start in range(0, len(idxs) - batch_size + 1, batch_size):
        yield [records[i] for i in idxs[start : start + batch_size]]


# ── Reference logprobs ─────────────────────────────────────────────────────────

def compute_ref_logprobs(
    ref_sampler: tinker.SamplingClient,
    full_tokens: list[int],
) -> list[float]:
    """
    Query reference model for per-token log-probs.

    compute_logprobs(from_ints(full_tokens)) returns a list of length N where
      result[i] = log P(token_i | token_0 .. token_{i-1}),  result[0] = None

    We return result[1:] (length N-1), which aligns with the policy logprobs
    produced by forward_backward_custom for a Datum built from full_tokens.
    """
    raw = ref_sampler.compute_logprobs(tt.ModelInput.from_ints(full_tokens)).result()
    return [lp if lp is not None else 0.0 for lp in raw][1:]


# ── DPO loss function ──────────────────────────────────────────────────────────

def make_dpo_loss_fn(
    ref_lp_all:   list[list[float]],   # len = 2*B, interleaved chosen/rejected
    weights_all:  list[list[float]],   # len = 2*B, loss masks
    dpo_beta:     float,
    pair_weights: list[float] | None = None,  # per-pair quality weights (len = B)
):
    """
    Returns a CustomLossFnV1 compatible with forward_backward_custom().

    Tinker calls:  loss, metrics = loss_fn(data, logprobs_list)
      data:          list[Datum]   – the original batch data
      logprobs_list: list[Tensor]  – per-target-token log-probs from current policy,
                                     each tensor has requires_grad=True

    Convention: even index = chosen, odd index = rejected.
    """
    def dpo_loss(data: list[tt.Datum], logprobs_list: list[torch.Tensor]):
        batch_losses     = []
        chosen_rewards   = []
        rejected_rewards = []
        accuracies       = []

        n_pairs = len(data) // 2
        for i in range(n_pairs):
            ci = 2 * i
            ri = 2 * i + 1

            pi_lp_c = logprobs_list[ci]   # current policy, chosen  (requires_grad=True)
            pi_lp_r = logprobs_list[ri]   # current policy, rejected

            ref_lp_c = torch.tensor(ref_lp_all[ci],  dtype=torch.float32)
            ref_lp_r = torch.tensor(ref_lp_all[ri],  dtype=torch.float32)

            w_c = torch.tensor(weights_all[ci], dtype=torch.float32)
            w_r = torch.tensor(weights_all[ri], dtype=torch.float32)

            # Guard against length mismatches (truncation edge cases)
            min_c = min(len(pi_lp_c), len(ref_lp_c), len(w_c))
            min_r = min(len(pi_lp_r), len(ref_lp_r), len(w_r))
            pi_lp_c  = pi_lp_c[:min_c];  ref_lp_c = ref_lp_c[:min_c];  w_c = w_c[:min_c]
            pi_lp_r  = pi_lp_r[:min_r];  ref_lp_r = ref_lp_r[:min_r];  w_r = w_r[:min_r]

            # Log-ratio: sum of (policy - reference) logprobs over response tokens
            chosen_log_ratio   = (pi_lp_c * w_c).sum() - (ref_lp_c * w_c).sum()
            rejected_log_ratio = (pi_lp_r * w_r).sum() - (ref_lp_r * w_r).sum()

            pw   = pair_weights[i] if pair_weights else 1.0
            loss = pw * (-F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio)))
            batch_losses.append(loss)

            cr = chosen_log_ratio.detach().item()
            rr = rejected_log_ratio.detach().item()
            chosen_rewards.append(cr)
            rejected_rewards.append(rr)
            accuracies.append(1.0 if cr > rr else 0.0)

        total_weight = sum(pair_weights) if pair_weights else n_pairs
        total_loss   = torch.stack(batch_losses).sum() / total_weight
        n = len(chosen_rewards)
        metrics = {
            "dpo_loss":        total_loss.detach().item(),
            "chosen_reward":   sum(chosen_rewards)  / n,
            "rejected_reward": sum(rejected_rewards) / n,
            "margin":         (sum(chosen_rewards) - sum(rejected_rewards)) / n,
            "accuracy":        sum(accuracies) / n,
        }
        return total_loss, metrics

    return dpo_loss


# ── Prepare one batch into Datums ──────────────────────────────────────────────

def prepare_batch(
    tokenizer,
    ref_sampler: tinker.SamplingClient,
    batch: list[dict],
    max_length: int,
):
    """
    Tokenize a batch and compute reference logprobs.

    Returns:
        data:        list[Datum]        – interleaved [chosen0, rejected0, chosen1, ...]
        ref_lp_all:  list[list[float]]  – reference logprobs aligned with data
        weights_all: list[list[float]]  – loss masks aligned with data
    """
    data:        list[tt.Datum]    = []
    full_toks:   list[list[int]]   = []
    weights_all: list[list[float]] = []

    for rec in batch:
        # Cross-instance pairs have separate prompts for chosen and rejected
        prompt_c = rec.get("prompt_chosen",   rec.get("prompt", ""))
        prompt_r = rec.get("prompt_rejected",  rec.get("prompt", ""))
        for prompt, response in ((prompt_c, rec["chosen"]), (prompt_r, rec["rejected"])):
            prompt_len = get_prompt_len(tokenizer, prompt)
            full_ids   = apply_chat_template(tokenizer, prompt, response)
            datum, toks = make_datum(full_ids, prompt_len, max_length)
            data.append(datum)
            full_toks.append(toks)
            weights_all.append(datum.loss_fn_inputs["weights"].data)

    # Compute reference logprobs concurrently (submit all, then collect)
    ref_futures = [
        ref_sampler.compute_logprobs(tt.ModelInput.from_ints(toks))
        for toks in full_toks
    ]
    ref_lp_all = []
    for future in ref_futures:
        raw = future.result()
        ref_lp_all.append([lp if lp is not None else 0.0 for lp in raw][1:])

    return data, ref_lp_all, weights_all


# ── Training step ──────────────────────────────────────────────────────────────

def train_step(
    tc:          tinker.TrainingClient,
    ref_sampler: tinker.SamplingClient,
    tokenizer,
    batch:       list[dict],
    adam_params: tt.AdamParams,
    dpo_beta:    float,
    max_length:  int,
) -> dict:
    """One full DPO training step: fwd-bwd + optim."""
    data, ref_lp_all, weights_all = prepare_batch(
        tokenizer, ref_sampler, batch, max_length
    )
    pair_weights = [rec.get("weight", 1.0) for rec in batch]
    loss_fn = make_dpo_loss_fn(ref_lp_all, weights_all, dpo_beta, pair_weights)

    # Submit fwd-bwd and optim concurrently (Tinker queues them in order)
    fb_future    = tc.forward_backward_custom(data, loss_fn)
    optim_future = tc.optim_step(adam_params)

    fb_result = fb_future.result()
    optim_future.result()
    return fb_result.metrics


def eval_step(
    tc:          tinker.TrainingClient,
    ref_sampler: tinker.SamplingClient,
    tokenizer,
    batch:       list[dict],
    dpo_beta:    float,
    max_length:  int,
) -> dict:
    """
    Eval: forward-only pass to compute DPO metrics without updating weights.
    Uses tc.forward() which does NOT accumulate gradients.
    """
    data, ref_lp_all, weights_all = prepare_batch(
        tokenizer, ref_sampler, batch, max_length
    )
    # forward (not forward_backward) → get logprobs without grad accumulation
    fwd_result = tc.forward(data, "cross_entropy").result()

    # Manually compute DPO metrics from logprobs
    chosen_rewards   = []
    rejected_rewards = []
    accuracies       = []

    n_pairs = len(data) // 2
    for i in range(n_pairs):
        ci = 2 * i
        ri = 2 * i + 1

        def weighted_sum(lp_data: list[float], w: list[float]) -> float:
            T = min(len(lp_data), len(w))
            return sum(lp_data[j] * w[j] for j in range(T))

        # Policy logprobs from forward result
        pi_lp_c = fwd_result.loss_fn_outputs[ci]["logprobs"].data
        pi_lp_r = fwd_result.loss_fn_outputs[ri]["logprobs"].data

        cr = weighted_sum(pi_lp_c, weights_all[ci]) - weighted_sum(ref_lp_all[ci], weights_all[ci])
        rr = weighted_sum(pi_lp_r, weights_all[ri]) - weighted_sum(ref_lp_all[ri], weights_all[ri])

        chosen_rewards.append(cr)
        rejected_rewards.append(rr)
        accuracies.append(1.0 if cr > rr else 0.0)

    n = len(chosen_rewards)
    return {
        "eval_chosen_reward":   sum(chosen_rewards)  / n,
        "eval_rejected_reward": sum(rejected_rewards) / n,
        "eval_margin":         (sum(chosen_rewards) - sum(rejected_rewards)) / n,
        "eval_accuracy":        sum(accuracies) / n,
    }


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _save_dpo_checkpoint(tc, ckpt_name: str, step: int, tag: str, run_dir: Path):
    """Save a checkpoint and record its tinker_path to dpo_log.jsonl."""
    result = tc.save_state(ckpt_name).result()
    tinker_path = getattr(result, "path", None) or ckpt_name
    log.info(f"Checkpoint saved: {ckpt_name}  →  {tinker_path}")
    entry = {"type": "checkpoint", "step": step, "tag": tag,
             "name": ckpt_name, "tinker_path": tinker_path}
    with open(run_dir / "dpo_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def train(args):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"dpo_{args.model.split('/')[-1]}_{int(time.time())}"
    run_dir  = RUNS_DIR / run_name
    run_dir.mkdir()

    log.info(f"Run dir: {run_dir}")
    log.info(f"Model={args.model}  lr={args.lr}  beta={args.dpo_beta}  batch={args.batch_size}")

    # ── Clients ──────────────────────────────────────────────────────────────
    log.info("Creating Tinker ServiceClient ...")
    service = tinker.ServiceClient()

    log.info(f"Creating LoRA TrainingClient (rank={args.lora_rank}) ...")
    tc = service.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
    )
    tokenizer = tc.get_tokenizer()
    log.info(f"Tokenizer: {tokenizer.__class__.__name__}")

    # Reference model = snapshot of initial weights (LoRA delta ≈ 0 at start)
    log.info("Snapshotting reference model weights ...")
    ref_sampler = tc.save_weights_and_get_sampling_client()
    log.info("Reference model ready.")

    adam_params = tt.AdamParams(
        learning_rate=args.lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    eval_records  = load_jsonl(DATA_DIR / "eval.jsonl")
    log.info(f"Train={len(train_records)}, Eval={len(eval_records)}")

    # ── Training loop ─────────────────────────────────────────────────────────
    metrics_log = []
    step  = 0
    epoch = 0
    overopt_streak = 0  # consecutive steps with margin > early_stop_margin

    # test for 3 epoch
    while epoch < 5:
    # while step < args.steps:
        epoch += 1
        log.info(f"=== Epoch {epoch} start (step {step}/{args.steps}) ===")
        for batch in iter_batches(train_records, args.batch_size, seed=epoch):
            if step >= args.steps:
                break

            step += 1
            t0 = time.time()

            metrics = train_step(
                tc=tc,
                ref_sampler=ref_sampler,
                tokenizer=tokenizer,
                batch=batch,
                adam_params=adam_params,
                dpo_beta=args.dpo_beta,
                max_length=args.max_length,
            )

            elapsed = time.time() - t0
            metrics.update({"step": step, "elapsed_s": round(elapsed, 1)})
            metrics_log.append(metrics)

            log.info(
                f"step={step:4d}/{args.steps} | "
                f"loss={metrics.get('dpo_loss', float('nan')):.4f} | "
                f"margin={metrics.get('margin', 0):.4f} | "
                f"acc={metrics.get('accuracy', 0):.2f} | "
                f"{elapsed:.1f}s"
            )

            # ── Early stopping (over-optimization guard) ──────────────────────
            if metrics.get('margin', 0) > args.early_stop_margin:
                overopt_streak += 1
            else:
                overopt_streak = 0
            if overopt_streak >= args.early_stop_patience:
                log.warning(
                    f"Early stop: margin > {args.early_stop_margin} for "
                    f"{args.early_stop_patience} consecutive steps. Saving and exiting."
                )
                ckpt_name = f"{run_name}--step-{step:04d}-early-stop"
                _save_dpo_checkpoint(tc, ckpt_name, step, f"step_{step:04d}_early_stop", run_dir)
                break

            # ── Checkpoint ───────────────────────────────────────────────────
            if step % args.save_every == 0:
                ckpt_name = f"{run_name}--step-{step:04d}"
                log.info(f"Saving checkpoint: {ckpt_name}")
                _save_dpo_checkpoint(tc, ckpt_name, step, f"step_{step:04d}", run_dir)
                log.info("Checkpoint saved.")

            # ── Eval ─────────────────────────────────────────────────────────
            if step % args.eval_every == 0 and eval_records:
                log.info("Running eval ...")
                eval_batch = random.sample(
                    eval_records, min(args.batch_size, len(eval_records))
                )
                try:
                    em = eval_step(
                        tc=tc,
                        ref_sampler=ref_sampler,
                        tokenizer=tokenizer,
                        batch=eval_batch,
                        dpo_beta=args.dpo_beta,
                        max_length=args.max_length,
                    )
                    log.info(
                        f"  [eval] margin={em.get('eval_margin', 0):.4f} | "
                        f"acc={em.get('eval_accuracy', 0):.2f}"
                    )
                    metrics_log[-1].update(em)
                except Exception as e:
                    log.warning(f"  [eval] skipped: {e}")

    # ── Final save ────────────────────────────────────────────────────────────
    log.info("Saving final checkpoint ...")
    _save_dpo_checkpoint(tc, f"{run_name}--final", step, "final", run_dir)

    metrics_path = run_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
    log.info(f"Metrics → {metrics_path}")
    log.info("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="Qwen/Qwen3-4B-Instruct-2507",
                   help="Base model available on the Tinker server")
    p.add_argument("--lr",         type=float, default=3e-6)
    p.add_argument("--dpo_beta",   type=float, default=0.3)
    p.add_argument("--early_stop_margin", type=float, default=50.0,
                   help="Stop if rolling avg margin exceeds this for --early_stop_patience steps")
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--batch_size", type=int,   default=4,
                   help="Pairs per step (actual Datum count = 2 * batch_size)")
    p.add_argument("--lora_rank",  type=int,   default=32)
    p.add_argument("--max_length", type=int,   default=8192,
                   help="Max token length (trajectory DPO: mean~6244, p90~7294)")
    p.add_argument("--steps",      type=int,   default=200)
    p.add_argument("--save_every", type=int,   default=50)
    p.add_argument("--eval_every", type=int,   default=20)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

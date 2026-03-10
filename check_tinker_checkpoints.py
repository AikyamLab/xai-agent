#!/usr/bin/env python3
"""
Check Tinker model checkpoints.

Usage:
  # Scan a run for all saved model checkpoints (tries each name from grpo_log):
  python check_tinker_checkpoints.py --scan-run <run_id> [--grpo-log path/to/grpo_log.jsonl]

  # Verify specific tinker:// paths directly:
  python check_tinker_checkpoints.py tinker://abc:train:0/weights/checkpoint_step_50

  # List all training runs (compact):
  python check_tinker_checkpoints.py --list-runs
"""
import argparse
import json
import os
import sys


DEFAULT_GRPO_LOG = "checkpoints/grpo_tabular_qwen4b_3/grpo_log.jsonl"
WEIGHTS_PATH_FMT = "tinker://{run_id}/weights/{name}"


def get_rest():
    import tinker
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("ERROR: TINKER_API_KEY not set")
        sys.exit(1)
    return tinker.ServiceClient().create_rest_client()


def verify_path(rest, tinker_path):
    try:
        result = rest.get_weights_info_by_tinker_path(tinker_path).result()
        info = vars(result) if hasattr(result, "__dict__") else str(result)
        print(f"  OK    {tinker_path}")
        print(f"        {info}")
        return True
    except Exception as e:
        print(f"  MISS  {tinker_path}  → {e}")
        return False


def read_checkpoint_names(grpo_log_path):
    """Read checkpoint tag names from grpo_log.jsonl."""
    names = []
    try:
        with open(grpo_log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "checkpoint" and "name" in rec:
                    names.append(rec["name"])
    except FileNotFoundError:
        print(f"  [warn] grpo_log not found: {grpo_log_path}")
    return names


def scan_run(rest, run_id, grpo_log_path):
    """Verify all checkpoint paths for a given run_id using names from grpo_log."""
    names = read_checkpoint_names(grpo_log_path)
    if not names:
        # Fallback: try common names
        names = [f"checkpoint_step_{n}" for n in range(10, 200, 10)] + ["checkpoint_final"]
        print(f"  [info] No grpo_log found, trying common names ({len(names)} candidates)")
    else:
        print(f"  [info] Found {len(names)} checkpoint entries in grpo_log: {names}")

    print(f"\n── Scanning run {run_id} ──")
    found = []
    for name in names:
        path = WEIGHTS_PATH_FMT.format(run_id=run_id, name=name)
        if verify_path(rest, path):
            found.append(path)

    print(f"\nFound {len(found)}/{len(names)} checkpoints.")
    return found


def list_runs(rest, limit=50):
    """List most recent training runs compactly."""
    print(f"── Recent training runs (limit={limit}) ──")
    result = rest.list_training_runs(limit=limit, offset=0).result()
    runs = getattr(result, "training_runs", None) or result
    if not runs:
        print("  (none)")
        return
    print(f"  Total returned: {len(runs)}")
    for run in runs:
        last_ckpt = run.last_checkpoint
        ckpt_path = getattr(last_ckpt, "tinker_path", None) if last_ckpt else None
        marker = "* " if ckpt_path else "  "
        print(f"{marker}run_id={run.training_run_id}  "
              f"last_ckpt={ckpt_path or 'None'}")


def main():
    parser = argparse.ArgumentParser(description="Check Tinker model checkpoints")
    parser.add_argument("paths", nargs="*", help="tinker:// paths to verify directly")
    parser.add_argument("--scan-run", metavar="RUN_ID",
                        help="Scan a run for all model checkpoints")
    parser.add_argument("--grpo-log", default=DEFAULT_GRPO_LOG,
                        help=f"Path to grpo_log.jsonl (default: {DEFAULT_GRPO_LOG})")
    parser.add_argument("--list-runs", action="store_true",
                        help="List recent training runs")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max runs to show with --list-runs (default: 50)")
    args = parser.parse_args()

    rest = get_rest()

    if args.list_runs:
        list_runs(rest, limit=args.limit)

    if args.scan_run:
        scan_run(rest, args.scan_run, args.grpo_log)

    if args.paths:
        print("── Verifying paths ──")
        for p in args.paths:
            verify_path(rest, p)

    if not args.list_runs and not args.scan_run and not args.paths:
        parser.print_help()


if __name__ == "__main__":
    main()

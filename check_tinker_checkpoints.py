#!/usr/bin/env python3
"""
Query Tinker API to list saved checkpoints and verify they exist.
Uses RestClient which has list_training_runs and list_user_checkpoints.
"""
import os
import sys

def main():
    import tinker

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("ERROR: TINKER_API_KEY not set")
        sys.exit(1)
    print(f"TINKER_API_KEY: set ({api_key[:8]}...)\n")

    service_client = tinker.ServiceClient()
    rest = service_client.create_rest_client()

    # ── 1. List all checkpoints belonging to this user ─────────────────────────
    print("── User checkpoints (list_user_checkpoints) ──")
    try:
        result = rest.list_user_checkpoints().result()
        items = getattr(result, "checkpoints", None) or getattr(result, "items", None) or result
        if not items:
            print("  (none)")
        else:
            for ckpt in items:
                print(f"  [{ckpt.checkpoint_type}]  "
                      f"id={ckpt.checkpoint_id}  "
                      f"path={ckpt.tinker_path}  "
                      f"size={ckpt.size_bytes}  "
                      f"time={ckpt.time}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 2. List all training runs for this user ────────────────────────────────
    print("\n── Training runs (list_training_runs) ──")
    try:
        result = rest.list_training_runs().result()
        runs = getattr(result, "training_runs", None) or result
        if not runs:
            print("  (none)")
        else:
            for run in runs:
                print(f"  run_id={run.training_run_id}  "
                      f"model={run.base_model}  "
                      f"last_ckpt={run.last_checkpoint}  "
                      f"last_sampler={run.last_sampler_checkpoint}")

            # ── 3. List checkpoints for each run ──────────────────────────────
            print("\n── Checkpoints per training run ──")
            for run in runs:
                run_id = run.training_run_id
                try:
                    ckpt_result = rest.list_checkpoints(run_id).result()
                    ckpts = getattr(ckpt_result, "checkpoints", None) or ckpt_result
                    if not ckpts:
                        print(f"  [{run_id}] (no checkpoints)")
                    else:
                        for ckpt in ckpts:
                            print(f"  [{run_id}] [{ckpt.checkpoint_type}]  "
                                  f"id={ckpt.checkpoint_id}  "
                                  f"path={ckpt.tinker_path}  "
                                  f"time={ckpt.time}")
                except Exception as e:
                    print(f"  [{run_id}] ERROR listing checkpoints: {e}")
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()

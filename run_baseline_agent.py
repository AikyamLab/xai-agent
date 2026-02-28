#!/usr/bin/env python3
"""
Baseline Agent Wrapper

Thin wrapper around xai_pipeline_v2.py that injects baseline reasoning
preambles (CoT / ReAct / ToT) into the prompt builders before running
the standard 3-agent pipeline.

How it works:
    1. Reads BASELINE_TYPE env var (set by run_baseline_agents.sh)
    2. Applies monkey-patches to prepend reasoning preambles to all
       PromptBuilder subclasses (Q1–Q10)
    3. Delegates to xai_pipeline_v2.main() with all CLI args passed through

For "naive": no patches are applied — the pipeline runs as-is.
For "cot"/"react"/"tot": reasoning preambles are prepended to every
proposer and actor prompt.

Usage:
    # Called by run_baseline_agent_batch.py (via subprocess), not directly
    BASELINE_TYPE=cot python run_baseline_agent.py --dataset ... --question_id ...

    # Or directly
    BASELINE_TYPE=tot python run_baseline_agent.py \\
        --dataset dataset/test/tabular/adult_tabnn_q1.json \\
        --question_id 0 --model_url tabular/adult_tabnn.pth \\
        --vlm tinker/Qwen3-VL-30B-A3B-Instruct --no-improvement --no-sf
"""

import os
import sys

# ---------------------------------------------------------------------------
# Step 1: Read baseline type from environment
# ---------------------------------------------------------------------------
BASELINE_TYPE = os.environ.get("BASELINE_TYPE", "naive").lower().strip()
print(f"\n{'='*70}")
print(f"BASELINE AGENT WRAPPER")
print(f"  Baseline type: {BASELINE_TYPE}")
print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# Step 2: Apply prompt patches BEFORE importing the pipeline
# ---------------------------------------------------------------------------
from baselines.patch_prompts import apply_baseline_patches
apply_baseline_patches(BASELINE_TYPE)

# ---------------------------------------------------------------------------
# Step 3: Run the standard pipeline (all CLI args passed through)
# ---------------------------------------------------------------------------
from xai_pipeline_v2 import main

if __name__ == "__main__":
    main()

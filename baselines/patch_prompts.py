"""
Prompt Monkey-Patching for Baseline Agents

Patches the PromptBuilder base class methods so that every question-specific
prompt builder (Q1–Q10) automatically prepends the baseline-type preamble
to its proposer and actor prompts.

Usage (from run_baseline_agent.py):

    from baselines.patch_prompts import apply_baseline_patches
    apply_baseline_patches("cot")    # injects CoT preamble
    # then run the pipeline — all prompts will have the preamble prepended

This works because:
- All Q1–Q10 prompt builders inherit from PromptBuilder (base_prompt.py)
- Each overrides build_proposer_prompt / build_actor_prompt
- We wrap those methods at the BASE CLASS level so every subclass picks it up
- The pipeline code is untouched
"""

from __future__ import annotations

import os
from typing import Any, Dict

_PATCHED = False  # guard against double-patching


def apply_baseline_patches(baseline_type: str) -> None:
    """
    Monkey-patch PromptBuilder so all prompt builders prepend the
    reasoning-mode preamble for the given baseline type.

    Args:
        baseline_type: One of "naive", "cot", "react", "tot".
                       "naive" is a no-op (no preamble).
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    baseline_type = baseline_type.lower().strip()
    if baseline_type == "naive":
        # Naive: no preamble, pipeline runs as-is
        print(f"[patch_prompts] Baseline type 'naive' — no prompt patches applied.")
        return

    from baselines.prompt_preambles import PROPOSER_PREAMBLES, ACTOR_PREAMBLES

    proposer_preamble = PROPOSER_PREAMBLES.get(baseline_type, "")
    actor_preamble = ACTOR_PREAMBLES.get(baseline_type, "")

    if not proposer_preamble and not actor_preamble:
        print(f"[patch_prompts] Unknown baseline type '{baseline_type}' — no patches applied.")
        return

    print(f"[patch_prompts] Applying '{baseline_type}' preamble to all prompt builders...")

    # Import ALL question-specific prompt builder modules so their classes
    # are registered before we patch.  Each module defines a subclass of
    # PromptBuilder that overrides build_proposer_prompt / build_actor_prompt.
    from prompts.base_prompt import PromptBuilder
    import prompts.q1_most_responsible
    import prompts.q2_least_responsible
    import prompts.q3_distinctive
    import prompts.q4_contrastive_instances
    import prompts.q5_mask_prediction
    import prompts.q6_flip_prediction
    import prompts.q7_change_prediction
    import prompts.q8_irrelevant_parts
    import prompts.q9_shared_feature
    import prompts.q10_similar_different

    # Collect all concrete subclasses
    def _all_subclasses(cls):
        result = set()
        for sub in cls.__subclasses__():
            result.add(sub)
            result.update(_all_subclasses(sub))
        return result

    subclasses = _all_subclasses(PromptBuilder)

    patched_count = 0
    for klass in subclasses:
        # --- Patch build_proposer_prompt ---
        if proposer_preamble and hasattr(klass, "build_proposer_prompt"):
            original_proposer = klass.build_proposer_prompt

            # Avoid re-wrapping already-patched methods
            if getattr(original_proposer, "_baseline_patched", False):
                continue

            def _make_proposer_wrapper(orig, preamble):
                def wrapper(self, context, *args, **kwargs):
                    prompt = orig(self, context, *args, **kwargs)
                    return preamble + prompt
                wrapper._baseline_patched = True
                return wrapper

            klass.build_proposer_prompt = _make_proposer_wrapper(
                original_proposer, proposer_preamble
            )

        # --- Patch build_actor_prompt ---
        if actor_preamble and hasattr(klass, "build_actor_prompt"):
            original_actor = klass.build_actor_prompt

            if getattr(original_actor, "_baseline_patched", False):
                continue

            def _make_actor_wrapper(orig, preamble):
                def wrapper(self, context, strategy, results, *args, **kwargs):
                    prompt = orig(self, context, strategy, results, *args, **kwargs)
                    return preamble + prompt
                wrapper._baseline_patched = True
                return wrapper

            klass.build_actor_prompt = _make_actor_wrapper(
                original_actor, actor_preamble
            )

        # --- Patch build_proposer_prompt_multi (for Q4, Q9, Q10) ---
        if proposer_preamble and hasattr(klass, "build_proposer_prompt_multi"):
            original_multi = klass.build_proposer_prompt_multi

            if getattr(original_multi, "_baseline_patched", False):
                continue

            def _make_multi_wrapper(orig, preamble):
                def wrapper(self, context, instances, *args, **kwargs):
                    prompt = orig(self, context, instances, *args, **kwargs)
                    return preamble + prompt
                wrapper._baseline_patched = True
                return wrapper

            klass.build_proposer_prompt_multi = _make_multi_wrapper(
                original_multi, proposer_preamble
            )

        # --- Patch build_actor_prompt_multi (for Q4, Q9, Q10) ---
        if actor_preamble and hasattr(klass, "build_actor_prompt_multi"):
            original_actor_multi = klass.build_actor_prompt_multi

            if getattr(original_actor_multi, "_baseline_patched", False):
                continue

            def _make_actor_multi_wrapper(orig, preamble):
                def wrapper(self, *args, **kwargs):
                    prompt = orig(self, *args, **kwargs)
                    return preamble + prompt
                wrapper._baseline_patched = True
                return wrapper

            klass.build_actor_prompt_multi = _make_actor_multi_wrapper(
                original_actor_multi, actor_preamble
            )

        patched_count += 1

    print(f"[patch_prompts] Patched {patched_count} prompt builder classes with '{baseline_type}' preamble.")

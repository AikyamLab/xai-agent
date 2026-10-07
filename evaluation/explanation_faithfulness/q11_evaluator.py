"""
Q11 Evaluator: Which Concept Was Most Responsible (CUB-only)

Unlike every other evaluator in this package, Q11 does not mask anything in
input space (no evaluation/masking_utils involved at all). The Actor names a
concept (grounded against a candidate list of attributes actually present in
the image -- see agents/actor_agent.py::_build_context and
concept_level/build_q11_dataset.py); this evaluator:

1. Matches that name against the candidate list (hallucination check).
2. Loads the matched concept's precomputed Concept Activation Vector (CAV) --
   see concept_level/train_cavs.py.
3. Captures the penultimate-layer activation for `original_input` via a
   forward pre-hook on the model's classifier head (same architecture-
   agnostic tap point as concept_level/cub_model_utils.py).
4. Orthogonally projects the CAV direction out of that activation and
   re-runs just the classifier head (not the whole model) on the result.

Metric: max(0, P_original - P_modified), where P_modified is the
probability of the original predicted class after the concept's
representation-space direction has been removed.
"""

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult

threshold = 0.5

CAV_DIR = os.path.join("concept_level", "cavs")

# Which attribute holds the classifier head for each CUB model -- see
# concept_level/cub_model_utils.py::MODEL_CONFIG (kept in sync manually,
# these are fixed architectural facts about the two checkpoints, not
# something that varies per run).
MODEL_HEAD_ATTR = {
    "cub_resnet": "fc",
    "cub_densenet": "classifier",
}


def _normalize_concept_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip('"\'')


class Q11Evaluator(BaseEvaluator):
    """Evaluator for Q11: which concept was most responsible (CUB-only)"""

    @property
    def question_type(self) -> int:
        return 11

    @property
    def metric_name(self) -> str:
        return "Probability Drop (representation-space CAV ablation)"

    @property
    def metric_formula(self) -> str:
        return "max(0, P_original - P_modified)"

    def _match_concept(
        self, concept_name: str, candidate_concepts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        target = _normalize_concept_name(concept_name)
        for c in candidate_concepts:
            if _normalize_concept_name(c["concept_name"]) == target:
                return c
        return None

    def _get_dataset_name(self, dataset_base_name: Optional[str]) -> Optional[str]:
        """Derive the cub_resnet/cub_densenet dataset name from dataset_base_name
        (e.g. 'cub_resnet_q11' -> 'cub_resnet'), same regex convention used
        throughout the pipeline (see agents/critic_agent.py)."""
        if not dataset_base_name:
            return None
        match = re.match(r"(.+?)_(q\d+)(?:_.*)?$", dataset_base_name)
        return match.group(1) if match else dataset_base_name

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        output_data = agent_output.get("output", {})
        concept_name = output_data.get("concept_name")
        if not concept_name or not isinstance(concept_name, str):
            return EvaluationResult(
                score=0.0, passed=False,
                metric_name=self.metric_name, metric_formula=self.metric_formula,
                errors=["Missing or non-string 'output.concept_name' in agent output"]
            )

        candidate_concepts = kwargs.get("candidate_concepts") or []
        matched = self._match_concept(concept_name, candidate_concepts)
        if matched is None:
            return EvaluationResult(
                score=0.0, passed=False,
                metric_name=self.metric_name, metric_formula=self.metric_formula,
                errors=[f"concept_name {concept_name!r} not found in candidate list "
                        f"(hallucinated concept, not actually present in this image)"]
            )
        attribute_id = matched["attribute_id"]

        dataset_name = self._get_dataset_name(kwargs.get("dataset_base_name"))
        head_attr = MODEL_HEAD_ATTR.get(dataset_name)
        if head_attr is None:
            return EvaluationResult(
                score=0.0, passed=False,
                metric_name=self.metric_name, metric_formula=self.metric_formula,
                errors=[f"Unknown CUB model {dataset_name!r}, cannot locate classifier head "
                        f"(expected one of {list(MODEL_HEAD_ATTR.keys())})"]
            )
        head = getattr(model, head_attr)

        cav_path = os.path.join(CAV_DIR, f"{dataset_name}_{attribute_id}.npy")
        if not os.path.exists(cav_path):
            return EvaluationResult(
                score=0.0, passed=False,
                metric_name=self.metric_name, metric_formula=self.metric_formula,
                errors=[f"No trained CAV found at {cav_path} for concept "
                        f"{matched['concept_name']!r} (attribute_id={attribute_id})"]
            )
        cav = np.load(cav_path)
        cav = cav / np.linalg.norm(cav)

        original_class = original_prediction.get('predicted_class_idx', 0)
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        processor = kwargs.get('processor')

        # Capture the penultimate activation via a forward pre-hook while
        # running the model's normal prediction path (reuses get_prediction's
        # existing preprocessing instead of duplicating it).
        captured: Dict[str, torch.Tensor] = {}

        def _hook(module, inputs):
            captured["activation"] = inputs[0].detach()

        handle = head.fc0.register_forward_pre_hook(_hook)
        try:
            prediction = self.get_prediction(model, original_input, processor, device)
        finally:
            handle.remove()

        if "activation" not in captured:
            return EvaluationResult(
                score=0.0, passed=False,
                metric_name=self.metric_name, metric_formula=self.metric_formula,
                errors=["Failed to capture penultimate activation via forward hook"]
            )

        original_probs = self._normalize_probs(prediction.get('probabilities'))
        if original_probs is None:
            return EvaluationResult(
                score=0.0, passed=False,
                metric_name=self.metric_name, metric_formula=self.metric_formula,
                errors=["Original probabilities not available"]
            )
        if isinstance(original_probs, dict):
            original_probs = [original_probs[k] for k in sorted(original_probs.keys())]
        p_original = float(original_probs[original_class])

        h = captured["activation"]  # [1, D]
        v = torch.tensor(cav, dtype=h.dtype, device=h.device)
        proj = (h @ v).unsqueeze(-1) * v.unsqueeze(0)  # [1, D]
        h_ablated = h - proj

        with torch.no_grad():
            head.eval()
            logits = head(h_ablated)
            modified_probs = torch.softmax(logits, dim=-1).squeeze(0)
        p_modified = float(modified_probs[original_class])
        modified_class = int(torch.argmax(modified_probs).item())

        raw_drop = p_original - p_modified
        score = max(0.0, raw_drop)
        class_changed = modified_class != original_class
        passed = (raw_drop > 0.5) or class_changed

        return EvaluationResult(
            score=score,
            passed=passed,
            metric_name=self.metric_name,
            metric_formula=self.metric_formula,
            p_original=p_original,
            p_modified=p_modified,
            original_class=str(original_class),
            modified_class=str(modified_class),
            details={
                "concept_name": matched["concept_name"],
                "attribute_id": attribute_id,
                "raw_drop": raw_drop,
                "class_changed": class_changed,
                "threshold": threshold,
                "n_candidate_concepts": len(candidate_concepts),
                "interpretation": (
                    "score = max(0, P_orig - P_mod); P_mod computed by orthogonally "
                    "projecting the concept's CAV direction out of the penultimate "
                    "activation and re-running only the classifier head"
                ),
            }
        )

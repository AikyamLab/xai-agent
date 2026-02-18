"""
Base Evaluator classes for XAI Agent Framework

Defines the abstract interface for explanation faithfulness evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class EvaluationResult:
    """
    Result of an explanation faithfulness evaluation.
    """
    # Core metrics
    score: float  # The primary faithfulness metric
    passed: bool  # Binary success indicator

    # Metric details
    metric_name: str = ""
    metric_formula: str = ""

    # Intermediate values for debugging
    p_original: Optional[float] = None  # Original probability
    p_modified: Optional[float] = None  # Probability after modification
    original_class: Optional[str] = None
    modified_class: Optional[str] = None

    # Additional details
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "score": self.score,
            "passed": self.passed,
            "metric_name": self.metric_name,
            "metric_formula": self.metric_formula,
            "p_original": self.p_original,
            "p_modified": self.p_modified,
            "original_class": self.original_class,
            "modified_class": self.modified_class,
            "details": self.details,
            "errors": self.errors
        }


class BaseEvaluator(ABC):
    """
    Abstract base class for explanation faithfulness evaluators.

    Each question type (Q1-Q10) has its own evaluator that:
    1. Parses the agent's output to extract identified regions/features
    2. Applies masking or modification to the input
    3. Runs the model on original and modified inputs
    4. Computes the faithfulness metric
    """

    def __init__(self, modality: str = "vision"):
        """
        Initialize evaluator.

        Args:
            modality: Data modality ("vision", "text", "tabular")
        """
        self.modality = modality

    @property
    @abstractmethod
    def question_type(self) -> int:
        """Return the question type this evaluator handles (1-10)"""
        pass

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the name of the evaluation metric"""
        pass

    @property
    @abstractmethod
    def metric_formula(self) -> str:
        """Return the formula for the evaluation metric"""
        pass

    @abstractmethod
    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate the faithfulness of an agent's explanation.

        Args:
            agent_output: Parsed output from the agent (must match output schema)
            original_input: Original input data (image tensor, text, or tabular data)
            model: The target model being explained
            original_prediction: Original prediction results
            **kwargs: Additional arguments (e.g., processor, tokenizer, ground_truth)

        Returns:
            EvaluationResult with score, passed status, and details
        """
        pass

    def compute_region_ratio(
        self,
        region: Dict[str, Any],
        original_input: Any
    ) -> float:
        """
        Compute the fraction of the input covered by the masked region.

        Returns:
            Float in [0, 1]. 0 = nothing masked, 1 = entire input masked.
        """
        try:
            if self.modality == "vision":
                bbox = region.get("bounding_box")
                if bbox is None:
                    return 0.0
                x_min, y_min, x_max, y_max = bbox
                region_area = max(0, x_max - x_min) * max(0, y_max - y_min)
                # Determine image dimensions
                if PIL_AVAILABLE and isinstance(original_input, Image.Image):
                    w, h = original_input.size
                elif isinstance(original_input, torch.Tensor):
                    if original_input.dim() == 4:
                        h, w = original_input.shape[2], original_input.shape[3]
                    elif original_input.dim() == 3:
                        h, w = original_input.shape[1], original_input.shape[2]
                    else:
                        return 0.0
                elif isinstance(original_input, np.ndarray):
                    h, w = original_input.shape[:2]
                else:
                    return 0.0
                image_area = w * h
                return region_area / image_area if image_area > 0 else 0.0

            elif self.modality == "text":
                start = region.get("start_index", 0)
                end = region.get("end_index", 0)
                span_len = max(0, end - start)
                if isinstance(original_input, dict) and 'premise' in original_input:
                    # NLI: region indices refer to premise text
                    text_len = len(original_input['premise'])
                elif isinstance(original_input, str):
                    text_len = len(original_input)
                elif isinstance(original_input, (list, tuple)):
                    text_len = len(original_input)
                else:
                    return 0.0
                return span_len / text_len if text_len > 0 else 0.0

            elif self.modality == "tabular":
                # One feature masked at a time
                if isinstance(original_input, dict):
                    total = len(original_input)
                elif isinstance(original_input, torch.Tensor):
                    total = original_input.shape[-1]
                elif isinstance(original_input, np.ndarray):
                    total = original_input.shape[-1]
                else:
                    return 0.0
                return 1.0 / total if total > 0 else 0.0

        except Exception:
            return 0.0
        return 0.0

    def extract_region(self, agent_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the identified region from agent output.

        Args:
            agent_output: Agent output dictionary

        Returns:
            Region dictionary with modality-specific fields, or None
        """
        output_data = agent_output.get("output", {})

        if self.modality == "vision":
            bbox = output_data.get("bounding_box")
            if bbox:
                return {"bounding_box": bbox}
        elif self.modality == "text":
            start = output_data.get("start_index")
            end = output_data.get("end_index")
            if start is not None and end is not None:
                return {"start_index": start, "end_index": end}
        elif self.modality == "tabular":
            key = output_data.get("feature_key")
            if key:
                return {"feature_key": key}

        return None

    def get_prediction(
        self,
        model: nn.Module,
        input_data: Any,
        processor: Any = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Run model prediction on input data.

        Args:
            model: PyTorch model
            input_data: Input tensor or data (image tensor, text string, dict for NLI, etc.)
            processor: Optional preprocessor (image transform, tokenizer, etc.)
            device: Device to run on

        Returns:
            Prediction dictionary with class and probabilities
        """
        model.eval()
        with torch.no_grad():
            # Handle NLI dict input: {'premise': '...', 'hypothesis': '...'}
            if isinstance(input_data, dict) and 'premise' in input_data and 'hypothesis' in input_data:
                if processor is None:
                    raise ValueError("Processor (tokenizer) required for NLI text input")
                premise_ids = processor(input_data['premise'])
                hypothesis_ids = processor(input_data['hypothesis'])
                premise_tensor = torch.tensor([premise_ids], dtype=torch.long).to(device)
                hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long).to(device)
                outputs = model(premise_tensor, hypothesis_tensor)
            else:
                # If input is already a tensor, use directly (skip processor)
                # This handles tabular data where features are pre-processed tensors
                if isinstance(input_data, torch.Tensor):
                    input_tensor = input_data
                elif processor is not None:
                    input_tensor = processor(input_data)
                else:
                    raise ValueError("Must provide processor or tensor input")

                # Convert list (e.g. from tokenizer) to tensor
                if isinstance(input_tensor, list):
                    input_tensor = torch.tensor([input_tensor], dtype=torch.long)

                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                elif input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)

                input_tensor = input_tensor.to(device)
                outputs = model(input_tensor)

            # Handle binary classification (single logit output)
            if outputs.shape[-1] == 1:
                # Check if model already applies sigmoid (by inspecting final layers)
                has_sigmoid = self._model_has_final_sigmoid(model)
                if has_sigmoid:
                    prob_pos = outputs.item()
                else:
                    prob_pos = torch.sigmoid(outputs).item()
                probs = torch.tensor([1.0 - prob_pos, prob_pos])
            else:
                probs = torch.softmax(outputs, dim=-1).squeeze()

            top_prob, top_class = probs.max(dim=-1)

            return {
                "predicted_class_idx": top_class.item(),
                "confidence": top_prob.item(),
                "probabilities": probs.cpu().numpy(),
                "logits": outputs.squeeze().cpu().numpy()
            }

    @staticmethod
    def _model_has_final_sigmoid(model: nn.Module) -> bool:
        """Check if model applies sigmoid/softmax as its final activation."""
        modules = list(model.modules())
        for m in reversed(modules):
            if isinstance(m, (nn.Sigmoid, nn.Softmax)):
                return True
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                return False
        return False

    @staticmethod
    def _normalize_probs(probs):
        """Convert dict probabilities to list, sorted by key for consistent indexing."""
        if isinstance(probs, dict):
            return [probs[k] for k in sorted(probs.keys())]
        return probs

    def compute_probability_difference(
        self,
        original_probs: Any,
        modified_probs: Any,
        target_class: int
    ) -> float:
        """
        Compute probability difference for a target class.

        Args:
            original_probs: Original probability distribution
            modified_probs: Modified probability distribution
            target_class: Class index to compare

        Returns:
            P_original[target_class] - P_modified[target_class]
        """
        if hasattr(original_probs, '__getitem__'):
            p_orig = float(original_probs[target_class])
            p_mod = float(modified_probs[target_class])
        else:
            p_orig = float(original_probs)
            p_mod = float(modified_probs)

        return p_orig - p_mod


class MultiInstanceEvaluator(BaseEvaluator):
    """
    Base class for evaluators handling multiple instances (Q4, Q9, Q10).
    """

    @abstractmethod
    def evaluate_multi(
        self,
        agent_output: Dict[str, Any],
        inputs: List[Any],
        model: nn.Module,
        predictions: List[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate with multiple instances.

        Args:
            agent_output: Agent output with multi-instance regions
            inputs: List of input data
            model: Target model
            predictions: List of prediction results
            **kwargs: Additional arguments

        Returns:
            EvaluationResult
        """
        pass

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Default evaluate wraps single input as list and calls evaluate_multi.
        """
        # For multi-instance evaluators, inputs should be passed via kwargs
        inputs = kwargs.pop("inputs", [original_input])
        predictions = kwargs.pop("predictions", [original_prediction])
        return self.evaluate_multi(agent_output, inputs, model, predictions, **kwargs)

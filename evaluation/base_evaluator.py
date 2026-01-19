"""
Base Evaluator classes for XAI Agent Framework

Defines the abstract interface for explanation faithfulness evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


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
            input_data: Input tensor or data
            processor: Optional preprocessor
            device: Device to run on

        Returns:
            Prediction dictionary with class and probabilities
        """
        model.eval()
        with torch.no_grad():
            if processor is not None:
                input_tensor = processor(input_data)
            elif isinstance(input_data, torch.Tensor):
                input_tensor = input_data
            else:
                raise ValueError("Must provide processor or tensor input")

            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(device)

            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=-1)

            top_prob, top_class = probs.max(dim=-1)

            return {
                "predicted_class_idx": top_class.item(),
                "confidence": top_prob.item(),
                "probabilities": probs.squeeze().cpu().numpy(),
                "logits": outputs.squeeze().cpu().numpy()
            }

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
        inputs = kwargs.get("inputs", [original_input])
        predictions = kwargs.get("predictions", [original_prediction])
        return self.evaluate_multi(agent_output, inputs, model, predictions, **kwargs)

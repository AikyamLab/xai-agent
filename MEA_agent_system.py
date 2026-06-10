"""
Three Agent System - New Modular Implementation

This module provides a unified interface to the refactored three-agent architecture:
- ProposerAgent: Strategy planning and tool selection
- ActorAgent: XAI tool execution and explanation generation
- CriticAgent: Explanation faithfulness evaluation

Usage:
    from MEA_agent_system import (
        create_three_agent_system,
        ProposerAgent,
        ActorAgent,
        CriticAgent
    )

    # Create system with VLM
    proposer, actor, critic = create_three_agent_system(vlm, model)

    # Run pipeline
    strategy = proposer.run(question, template, model_info, image_path, prediction)
    results = actor.run(strategy, question, template, image_path, model_info, prediction)
    evaluation = critic.run(results, question, image_tensor, prediction)
"""

from typing import Any, Dict, List, Optional, Tuple

# Import from modular agents
from agents.base_agent import BaseAgent
from agents.proposer_agent import ProposerAgent
from agents.actor_agent import ActorAgent
from agents.critic_agent import CriticAgent

# Import evaluation utilities
from evaluation import get_evaluator, get_masker, EvaluationResult

# Import prompt utilities
from question_templates_new import (
    get_question_template,
    get_prompt_builder,
    QuestionTemplate,
)


def create_three_agent_system(
    vlm: Any,
    model: Any = None,
    data_model_loader: Any = None,
    output_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    output_size_config=None
) -> Tuple[ProposerAgent, ActorAgent, CriticAgent]:
    """
    Create the three-agent system with shared VLM.

    Args:
        vlm: VisionLanguageModel instance
        model: Target model for explanation (optional, for CriticAgent)
        data_model_loader: DataModelLoader instance (optional)
        output_dir: Output directory path
        models_dir: Models directory path

    Returns:
        Tuple of (ProposerAgent, ActorAgent, CriticAgent)
    """
    proposer = ProposerAgent(
        vlm=vlm,
        data_model_loader=data_model_loader,
        models_dir=models_dir,
        output_dir=output_dir
    )

    actor = ActorAgent(
        vlm=vlm,
        output_dir=output_dir,
        output_size_config=output_size_config
    )

    critic = CriticAgent(
        vlm=vlm,
        model=model,
        output_dir=output_dir
    )

    return proposer, actor, critic


class ThreeAgentPipeline:
    """
    Unified pipeline for running all three agents.

    Provides a simplified interface for:
    1. Strategy proposal
    2. Explanation generation
    3. Faithfulness evaluation
    """

    def __init__(
        self,
        vlm: Any,
        model: Any = None,
        data_model_loader: Any = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the pipeline.

        Args:
            vlm: VisionLanguageModel instance
            model: Target model for explanation
            data_model_loader: DataModelLoader instance
            output_dir: Output directory
        """
        self.vlm = vlm
        self.model = model
        self.data_model_loader = data_model_loader

        self.proposer, self.actor, self.critic = create_three_agent_system(
            vlm=vlm,
            model=model,
            data_model_loader=data_model_loader,
            output_dir=output_dir
        )

    def initialize_actor_tools(
        self,
        model: Any,
        model_type: str,
        processor: Any,
        modality: str = "vision"
    ):
        """Initialize XAI tools for the Actor Agent"""
        self.actor.initialize_tools(model, model_type, processor, modality)

    def run(
        self,
        question: Dict[str, Any],
        input_data: Any,
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Any] = None,
        evaluate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the full three-agent pipeline.

        Args:
            question: Question dictionary with q_type, modality, question text
            input_data: Original input data (tensor/image/text/dict)
            input_path: Path to input file (for vision)
            model_info: Model metadata
            prediction: Model prediction results
            ground_truth: Ground truth label (for spurious feature questions)
            evaluate: Whether to run faithfulness evaluation
            **kwargs: Additional arguments

        Returns:
            Dictionary with strategy, results, and evaluation
        """
        q_type = question.get('q_type', 1)
        modality = question.get('modality', 'vision')

        # Get question template
        template = get_question_template(q_type, modality)

        # Step 1: Proposer - Generate strategy
        print("\n" + "=" * 70)
        print(f"Running Q{q_type}: {template.template}")
        print("=" * 70)

        strategy = self.proposer.run(
            question=question,
            question_template=template,
            model_info=model_info,
            input_path=input_path,
            prediction=prediction
        )

        # Step 2: Actor - Execute and explain
        results = self.actor.run(
            strategy=strategy,
            question=question,
            question_template=template,
            input_path=input_path,
            model_info=model_info,
            prediction=prediction
        )

        # Validate output
        is_valid, errors = template.validate_output(results)
        if not is_valid:
            print(f"Warning: Output validation failed: {errors}")

        # Step 3: Critic - Evaluate (if requested)
        evaluation = None
        if evaluate and self.model is not None:
            evaluation = self.critic.run(
                results=results,
                question=question,
                original_input=input_data,
                original_prediction=prediction,
                ground_truth=ground_truth,
                **kwargs
            )

        return {
            "question": question,
            "strategy": strategy,
            "results": results,
            "evaluation": evaluation,
            "output_valid": is_valid,
            "validation_errors": errors if not is_valid else []
        }

    def run_batch(
        self,
        questions: List[Dict[str, Any]],
        inputs: List[Any],
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline on a batch of questions.

        Args:
            questions: List of question dictionaries
            inputs: List of input data
            input_paths: List of input file paths
            predictions: List of predictions

        Returns:
            List of pipeline results
        """
        results = []

        for i, question in enumerate(questions):
            input_data = inputs[i] if inputs else None
            input_path = input_paths[i] if input_paths else None
            prediction = predictions[i] if predictions else None

            result = self.run(
                question=question,
                input_data=input_data,
                input_path=input_path,
                prediction=prediction,
                **kwargs
            )
            results.append(result)

        return results


# Export all components
__all__ = [
    # Agent classes
    "BaseAgent",
    "ProposerAgent",
    "ActorAgent",
    "CriticAgent",
    # Factory function
    "create_three_agent_system",
    # Pipeline
    "ThreeAgentPipeline",
    # Utilities
    "get_question_template",
    "get_prompt_builder",
    "get_evaluator",
    "get_masker",
    "EvaluationResult",
    "QuestionTemplate",
]

"""
XAI Pipeline V2 - Using New Modular Architecture

Complete end-to-end XAI pipeline using:
- Modular prompts (prompts/)
- Modular agents (agents/)
- Modular evaluation (evaluation/)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# New modular imports
from question_templates_new import (
    get_question_template,
    get_prompt_builder,
    QuestionTemplate,
    QUESTION_METADATA,
    Modality,
)
from three_agent_system_new import (
    create_three_agent_system,
    ThreeAgentPipeline,
    ProposerAgent,
    ActorAgent,
    CriticAgent,
)
from evaluation import get_evaluator, EvaluationResult

# Existing imports
from vlm_wrapper import VisionLanguageModel
from DataModelLoader import DataModelLoader


class XAIPipelineV2:
    """
    XAI Pipeline V2 with New Modular Architecture

    Features:
    - Per-question type prompt templates (Q1-Q10)
    - Standardized output formats
    - Faithfulness evaluation pipeline
    - Support for vision, text, and tabular modalities
    """

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        output_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        models_dir: Optional[str] = None
    ):
        """
        Initialize XAI Pipeline V2.

        Args:
            vlm_model_id: VLM model ID
            output_dir: Output directory
            dataset_dir: Directory containing datasets
            models_dir: Directory containing models
        """
        # Set default directories
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")
        if dataset_dir is None:
            dataset_dir = os.path.join(os.getcwd(), "dataset")
        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), "models_to_read")

        self.output_dir = Path(output_dir).resolve()
        self.dataset_dir = Path(dataset_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("INITIALIZING XAI PIPELINE V2 (New Architecture)")
        print("=" * 70)
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Models directory: {self.models_dir}")
        print(f"Output directory: {self.output_dir}")

        # Initialize VLM
        print(f"\nLoading VLM: {vlm_model_id}")
        self.vlm = VisionLanguageModel(model_id=vlm_model_id)

        # Initialize data/model loader
        self.data_model_loader = DataModelLoader(
            cache_dir=str(self.output_dir / "model_cache")
        )

        # Initialize three agents using new modular system
        print("\nInitializing Agents (New Architecture)...")
        self.proposer, self.actor, self.critic = create_three_agent_system(
            vlm=self.vlm,
            model=None,  # Will be set after loading target model
            data_model_loader=self.data_model_loader,
            output_dir=str(self.output_dir),
            models_dir=str(self.models_dir)
        )

        print("\nXAI Pipeline V2 initialized successfully!")

    def run(
        self,
        question_dataset_path: str,
        question_id: str,
        target_model_url: Optional[str] = None,
        image_path: Optional[str] = None,
        evaluate_faithfulness: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete XAI pipeline for a single question.

        Args:
            question_dataset_path: Path to question dataset JSON
            question_id: ID of question to process
            target_model_url: URL/path of target model
            image_path: Path to image for vision modality
            evaluate_faithfulness: Whether to run faithfulness evaluation

        Returns:
            Complete results dict
        """
        print("\n" + "=" * 70)
        print(f"RUNNING XAI PIPELINE V2 FOR QUESTION: {question_id}")
        print("=" * 70)

        # Step 1: Load question from dataset
        question, template = self._load_question(question_dataset_path, question_id)

        # Step 2: Load target model and data
        model_info, prediction, data_path, input_tensor = self._load_model_and_data(
            question, target_model_url, image_path=image_path
        )

        # Step 3: Initialize Actor's XAI tools
        if model_info and model_info.get('model'):
            print("\n=== Initializing XAI Tools ===")
            self.actor.initialize_tools(
                model=model_info['model'],
                model_type=model_info.get('model_type', 'local_pth'),
                processor=model_info.get('processor'),
                modality=question.get('modality', 'vision')
            )
            # Set model for critic
            self.critic.set_model(model_info['model'])

        # Step 4: Proposer generates strategy
        print("\n=== Step 3: Proposer Agent ===")
        strategy = self.proposer.run(
            question=question,
            question_template=template,
            model_info=model_info,
            input_path=data_path,
            prediction=prediction
        )

        # Step 5: Actor executes and explains
        print("\n=== Step 4: Actor Agent ===")
        results = self.actor.run(
            strategy=strategy,
            question=question,
            question_template=template,
            input_path=data_path,
            model_info=model_info,
            prediction=prediction
        )

        # Step 6: Validate output format
        print("\n=== Step 5: Output Validation ===")
        is_valid, validation_errors = template.validate_output(results)
        if is_valid:
            print("  Output format: VALID")
        else:
            print(f"  Output format: INVALID - {validation_errors}")

        # Step 7: Critic evaluates (if model available)
        print("\n=== Step 6: Critic Agent ===")
        evaluation = None
        if evaluate_faithfulness and model_info and model_info.get('model'):
            evaluation = self.critic.run(
                results=results,
                question=question,
                original_input=input_tensor,
                original_prediction=prediction,
                ground_truth=question.get('ground_truth'),
                processor=model_info.get('processor'),
                device=model_info.get('device', 'cuda')
            )
        else:
            print("  Skipping faithfulness evaluation (no model or disabled)")
            evaluation = {"status": "skipped"}

        # Combine all results
        complete_results = {
            "question_id": question_id,
            "question": question,
            "template": {
                "q_type": template.q_type,
                "category": template.category.value,
                "template_text": template.template
            },
            "strategy": strategy,
            "results": results,
            "output_validation": {
                "is_valid": is_valid,
                "errors": validation_errors
            },
            "evaluation": evaluation
        }

        # Save complete results
        self._save_complete_results(complete_results)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        return complete_results

    def _load_question(
        self,
        dataset_path: str,
        question_id: str
    ) -> Tuple[Dict[str, Any], QuestionTemplate]:
        """Load question and create template using new system."""
        print("\n=== Step 1: Loading Question ===")

        # Load dataset JSON
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # Handle both list and dict formats
        if isinstance(dataset, list):
            questions = dataset
        elif isinstance(dataset, dict):
            questions = dataset.get('questions', [dataset] if 'q_type' in dataset else [])
        else:
            questions = []
        print(f"Loaded dataset with {len(questions)} questions")

        # Find question by ID or index
        question = None
        try:
            idx = int(question_id)
            if 0 <= idx < len(questions):
                question = questions[idx]
                print(f"Using question at index {idx}")
        except ValueError:
            # Try as string ID
            for q in questions:
                if q.get('question_id') == question_id:
                    question = q
                    break

        if question is None:
            raise ValueError(f"Question '{question_id}' not found")

        # Normalize question fields
        if 'question' not in question:
            # Use 'q' or 'example' field as question
            question['question'] = question.get('example', question.get('q', ''))
        if 'modality' not in question:
            # Infer modality from features
            features = question.get('features', {})
            if features.get('modality') == 'image':
                question['modality'] = 'vision'
            else:
                question['modality'] = features.get('modality', 'vision')
        if 'question_id' not in question:
            question['question_id'] = f"q{question_id}"

        print(f"Question: {question.get('question', '')[:80]}...")

        # Get question type and modality
        q_type = question.get('q_type', 1)
        modality = question.get('modality', 'vision')

        # Create template using new system
        template = get_question_template(q_type, modality)
        print(f"Template: Q{q_type} - {template.template[:60]}...")
        print(f"  Modality: {modality}")
        print(f"  Category: {template.category.value}")

        return question, template

    def _load_model_and_data(
        self,
        question: Dict[str, Any],
        model_url: Optional[str],
        image_path: Optional[str] = None
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[str], Any]:
        """Load target model and input data."""
        print("\n=== Step 2: Loading Model and Data ===")

        modality = question.get('modality', 'vision')
        dataset_name = question.get('dataset', '')

        model_info = None
        prediction = None
        loaded_data_path = None
        input_tensor = None

        try:
            # Load data based on modality
            if modality == 'vision':
                final_image_path = None

                if image_path:
                    if not Path(image_path).exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    final_image_path = image_path
                else:
                    # Fallback: try to find image
                    print("Warning: --image_path not provided, using fallback logic")
                    data_path_from_question = question.get('data_path')

                    if data_path_from_question and Path(data_path_from_question).exists():
                        final_image_path = data_path_from_question
                    else:
                        # Search for image
                        features = question.get('features', {})
                        image_index = features.get(
                            'image_index',
                            question.get('metadata', {}).get('row_no', 0)
                        )
                        possible_paths = [
                            self.dataset_dir / f"{dataset_name}_{image_index}.png",
                            self.dataset_dir / f"{dataset_name}_{image_index}.jpg",
                            self.dataset_dir / dataset_name / f"{image_index}.png",
                            self.dataset_dir / dataset_name / f"{image_index}.jpg",
                        ]
                        for path in possible_paths:
                            if path.exists():
                                final_image_path = str(path)
                                break

                if final_image_path:
                    print(f"Loading image: {final_image_path}")
                    self.data_model_loader.load_image(final_image_path)
                    loaded_data_path = final_image_path
                    print("Image loaded")
                else:
                    raise FileNotFoundError(
                        f"Could not find image for dataset={dataset_name}. "
                        "Use --image_path argument."
                    )

            elif modality == 'text':
                text_input = question.get('text_input', '')
                if text_input:
                    print(f"Loading text data (length: {len(text_input)})")
                    self.data_model_loader.load_text_data(text_input)
                    print("Text data loaded")
                else:
                    print("Warning: No text_input in question")

            elif modality == 'tabular':
                features = question.get('features', {})
                if features:
                    print(f"Loading tabular data ({len(features)} features)")
                    self.data_model_loader.load_tabular_data(features)
                    print("Tabular data loaded")
                else:
                    print("Warning: No features in question")

            # Load model if URL provided
            if model_url:
                # Resolve relative path
                if not os.path.isabs(model_url):
                    local_model_path = self.models_dir / model_url
                    if local_model_path.exists():
                        model_url = str(local_model_path)

                print(f"Loading model: {model_url}")
                if model_url.startswith("https"):
                    model_info = self.data_model_loader.load_model_from_url(model_url)
                else:
                    model_info = self.data_model_loader.load_model_from_path(model_url)

                if model_info and model_info.get('success'):
                    print(f"Model loaded: {model_info.get('model_type', 'unknown')}")

                    # Add model summary
                    model_summary = self.data_model_loader.get_model_summary()
                    model_info.update({
                        "architecture": model_summary.get("architecture"),
                        "num_classes": model_summary.get("num_classes"),
                        "num_parameters": model_summary.get("num_parameters"),
                    })

                    # Make prediction
                    if self.data_model_loader.model is not None:
                        print("Making prediction...")
                        prediction = self.data_model_loader.predict()
                        if prediction and prediction.get('success'):
                            print(f"Prediction: class {prediction.get('predicted_class_idx')} "
                                  f"(confidence: {prediction.get('confidence', 0.0):.4f})")

                            # Get input data for evaluation
                            if modality == 'vision':
                                input_tensor = self.data_model_loader.current_image
                            elif modality == 'text':
                                input_tensor = self.data_model_loader.current_text_data
                            elif modality == 'tabular':
                                input_tensor = self.data_model_loader.current_tabular_data
                        else:
                            print(f"Warning: Prediction failed: {prediction.get('error')}")
                else:
                    print(f"Warning: Model loading failed: {model_info.get('error') if model_info else 'Unknown'}")

        except Exception as e:
            print(f"Warning: Failed to load model/data: {e}")
            import traceback
            traceback.print_exc()

        return model_info, prediction, loaded_data_path, input_tensor

    def _save_complete_results(self, results: Dict[str, Any]):
        """Save complete pipeline results."""
        results_file = self.output_dir / f"complete_results_v2_{results['question_id']}.json"

        # Clean up for JSON serialization
        clean_results = {
            "question_id": results["question_id"],
            "question": results["question"],
            "template": results["template"],
            "strategy": results["strategy"],
            "results": {k: v for k, v in results["results"].items()
                       if k not in ['raw_output', 'model']},
            "output_validation": results["output_validation"],
            "evaluation": results["evaluation"]
        }

        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)

        print(f"\nComplete results saved to: {results_file}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run XAI Pipeline V2 with New Modular Architecture"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to question dataset JSON file"
    )
    parser.add_argument(
        "--question_id",
        type=str,
        required=True,
        help="Question ID to process (integer index or string ID)"
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        help="URL or path to target model"
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="VLM model ID (default: Qwen/Qwen3-VL-8B-Instruct)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./outputs)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory containing datasets (default: ./dataset)"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Directory containing models (default: ./models_to_read)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to image for vision modality questions"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip faithfulness evaluation"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = XAIPipelineV2(
        vlm_model_id=args.vlm,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir
    )

    # Run pipeline
    results = pipeline.run(
        question_dataset_path=args.dataset,
        question_id=args.question_id,
        target_model_url=args.model_url,
        image_path=args.image_path,
        evaluate_faithfulness=not args.no_eval
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (V2)")
    print("=" * 70)
    print(f"Question ID: {results['question_id']}")
    print(f"Question Type: Q{results['template']['q_type']}")
    print(f"Category: {results['template']['category']}")
    print(f"Strategy: {results['strategy'].get('strategy_type', 'unknown')}")
    print(f"Output Valid: {results['output_validation']['is_valid']}")

    if results.get('evaluation') and results['evaluation'].get('faithfulness'):
        faith = results['evaluation']['faithfulness']
        print(f"Faithfulness Score: {faith.get('score', 'N/A')}")
        print(f"Faithfulness Passed: {faith.get('passed', 'N/A')}")

    explanation = results.get('results', {}).get('explanation', 'N/A')
    if isinstance(explanation, str) and len(explanation) > 100:
        explanation = explanation[:100] + "..."
    print(f"Explanation: {explanation}")


if __name__ == "__main__":
    main()

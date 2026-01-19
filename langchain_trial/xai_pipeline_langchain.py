"""
XAI Pipeline - LangChain Implementation

Complete end-to-end XAI pipeline using LangChain agents and tools.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from three_agent_system_langchain import (
    ProposerAgentLangChain,
    ActorAgentLangChain,
    CriticAgentLangChain
)
from vlm_langchain_wrapper import VisionLanguageModel
from question_templates import QuestionDataset, TemplateRegistry, Modality
from DataModelLoader import DataModelLoader

import os


class XAIPipelineLangChain:
    """
    Complete XAI Pipeline with LangChain Three Agent System
    """

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        output_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        models_dir: Optional[str] = None
    ):
        """
        Initialize XAI Pipeline with LangChain agents.

        Args:
            vlm_model_id: VLM model ID
            output_dir: Output directory (absolute or relative path)
                       Defaults to "./outputs" relative to current working directory
            dataset_dir: Directory containing datasets (for resolving data paths)
                        Defaults to "./dataset" relative to current working directory
            models_dir: Directory containing models
                       Defaults to "./models_to_read" relative to current working directory
        """
        import os

        # Set default directories if not provided
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

        print("\n" + "="*70)
        print("INITIALIZING XAI PIPELINE (LangChain)")
        print("="*70)
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

        # Initialize three agents (LangChain versions)
        print("\nInitializing LangChain Agents...")
        self.proposer = ProposerAgentLangChain(self.vlm, self.data_model_loader, models_dir=str(self.models_dir), output_dir=str(self.output_dir))
        self.actor = ActorAgentLangChain(self.vlm, output_dir=str(self.output_dir))
        self.critic = CriticAgentLangChain(self.vlm, output_dir=str(self.output_dir))

        print("\n✓ XAI Pipeline (LangChain) initialized successfully!")

    def run(
        self,
        question_dataset_path: str,
        question_id: str,
        target_model_url: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete XAI pipeline for a single question.

        Args:
            question_dataset_path: Path to question dataset JSON
            question_id: ID of question to process
            target_model_url: URL of target model (optional)
            image_path: Path to image for vision modality (optional)

        Returns:
            Complete results dict with strategy, explanation, and evaluation
        """
        print("\n" + "="*70)
        print(f"RUNNING XAI PIPELINE (LangChain) FOR QUESTION: {question_id}")
        print("="*70)

        # Step 1: Load question from dataset
        question, template = self._load_question(question_dataset_path, question_id)

        # Step 2: Load target model and data
        model_info, prediction, data_path = self._load_model_and_data(
            question, target_model_url, image_path=image_path
        )

        # Step 3: Initialize Actor's XAI tools if model is available
        if model_info and model_info.get('model'):
            print("\n=== Initializing XAI Tools for Actor ===")
            self.actor.initialize_tools(
                model=model_info['model'],
                model_type=model_info.get('model_type', 'local_pth'),
                processor=model_info.get('processor')
            )

        # Step 4: Proposer generates strategy (using LangChain Agent)
        print("\n=== Step 3: Proposer Agent (LangChain) ===")
        strategy, saved_model_metadata_path = self.proposer.propose_strategy(
            question=question,
            question_template=template,
            model_info=model_info,
            image_path=data_path if template.modality == Modality.VISION else None,
            prediction=prediction
        )


        # Step 5: Actor executes and explains (using LangChain Agent with Tools)
        print("\n=== Step 4: Actor Agent (LangChain) ===")
        results = self.actor.execute_and_explain(
            strategy=strategy,
            question=question,
            question_template=template,
            image_path=data_path if template.modality == Modality.VISION else None,
            model_info=model_info,
            prediction=prediction
        )

        # Step 6: Critic evaluates (using LangChain Agent)
        print("\n=== Step 5: Critic Agent (LangChain) ===")
        evaluation = self.critic.evaluate(
            results=results,
            question=question,
            ground_truth=question.get('ground_truth')
        )

        # Combine all results
        complete_results = {
            "question_id": question_id,
            "question": question,
            "strategy": strategy,
            "results": results,
            "evaluation": evaluation
        }

        # Save complete results
        self._save_complete_results(complete_results)

        print("\n" + "="*70)
        print("PIPELINE COMPLETE (LangChain)")
        print("="*70)

        return complete_results

    def _load_question(
        self,
        dataset_path: str,
        question_id: str
    ) -> tuple:
        """Load question and its template."""
        print("\n=== Step 1: Loading Question ===")

        # Load dataset
        dataset = QuestionDataset(dataset_path)
        print(f"✓ Loaded dataset with {len(dataset)} questions")

        # Try to parse question_id as an index first
        try:
            idx = int(question_id)
            if 0 <= idx < len(dataset):
                question = dataset[idx]
                print(f"✓ Using question at index {idx}")
            else:
                raise ValueError(f"Question index {idx} out of range (dataset has {len(dataset)} questions)")
        except ValueError:
            # If not a valid integer, try as a string ID
            question = dataset.get_question(question_id)
            if question is None:
                raise ValueError(f"Question '{question_id}' not found in dataset")

        print(f"✓ Question: {question['question'][:80]}...")

        # Get template
        template = dataset.get_template_for_question(question)
        if template is None:
            raise ValueError(f"No template found for question type: {question.get('q_type')}")

        print(f"✓ Template: {template.template[:80]}...")
        print(f"  Modality: {template.modality}")
        print(f"  Category: {template.category}")

        return question, template

    def _load_model_and_data(
        self,
        question: Dict[str, Any],
        model_url: Optional[str],
        image_path: Optional[str] = None
    ) -> tuple:
        """Load target model and input data."""
        print("\n=== Step 2: Loading Model and Data ===")

        modality = question.get('modality', 'vision')
        dataset_name = question.get('dataset', '')
        data_path = question.get('data_path')
        print(f"DEBUG: data_path from question: {data_path}")

        # Load model (if model_url provided)
        model_info = None
        prediction = None
        loaded_data_path = None

        try:
            # Load data based on modality
            if modality == 'vision':
                final_image_path = None
                if image_path:
                    # If --image_path is provided, it takes precedence and must be valid.
                    if not Path(image_path).exists():
                        raise FileNotFoundError(f"Image file not found at provided --image_path: {image_path}")
                    final_image_path = image_path
                else:
                    # If --image_path is not provided, use fallback logic.
                    print("Warning: --image_path not provided for vision question. Falling back to automatic discovery.")
                    data_path_from_question = question.get('data_path')
                    if data_path_from_question and Path(data_path_from_question).exists():
                        final_image_path = data_path_from_question
                    else:
                        # ... (search logic)
                        features = question.get('features', {})
                        image_index = features.get('image_index', question.get('metadata', {}).get('row_no', 0))
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
                    print(f"Loading image from: {final_image_path}")
                    self.data_model_loader.load_image(final_image_path)
                    loaded_data_path = final_image_path
                    print("✓ Image loaded")
                else:
                    raise FileNotFoundError(
                        f"Could not find image file for dataset={dataset_name}. "
                        "Please specify the path using the --image_path argument."
                    )
            
            elif modality == 'text':
                # For text, load from features
                text_input = question.get('text_input', '')
                if text_input:
                    print(f"Loading text data (length: {len(text_input)})")
                    self.data_model_loader.load_text_data(text_input)
                    print("✓ Text data loaded")
                else:
                    print("Warning: No text_input in question")

            elif modality == 'tabular':
                # For tabular, load from features
                features = question.get('features', {})
                if features:
                    print(f"Loading tabular data ({len(features)} features)")
                    self.data_model_loader.load_tabular_data(features)
                    print("✓ Tabular data loaded")
                else:
                    print("Warning: No features in question")

            # Load model if URL provided
            if model_url:
                # Check if model_url is a local path
                if not os.path.isabs(model_url):
                    # Try to resolve relative to models_dir
                    local_model_path = self.models_dir / model_url
                    if local_model_path.exists():
                        model_url = str(local_model_path)

                print(f"Loading model from: {model_url}")
                model_info = self.data_model_loader.load_model_from_path(model_url)

                if model_info and model_info.get('success'):
                    print(f"✓ Model loaded: {model_info.get('model_type', 'unknown')}")

                    # Add model summary details
                    model_summary = self.data_model_loader.get_model_summary()
                    model_info.update({
                        "architecture": model_summary.get("architecture"),
                        "num_classes": model_summary.get("num_classes"),
                        "num_parameters": model_summary.get("num_parameters"),
                        "num_trainable_parameters": model_summary.get("num_trainable_parameters"),
                        "total_layers": model_summary.get("total_layers")
                    })

                    # Make prediction if we have both model and data
                    if self.data_model_loader.model is not None:
                        print("Making prediction...")
                        prediction = self.data_model_loader.predict()
                        if prediction and prediction.get('success'):
                            print(f"✓ Prediction: class {prediction.get('predicted_class_idx')} "
                                  f"(confidence: {prediction.get('confidence', 0.0):.4f})")
                        else:
                            print(f"⚠ Prediction failed: {prediction.get('error', 'Unknown error')}")
                else:
                    print(f"⚠ Model loading failed: {model_info.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"⚠ Failed to load model/data: {e}")
            import traceback
            traceback.print_exc()
            model_info = None
            prediction = None

        return model_info, prediction, loaded_data_path

    def _save_complete_results(self, results: Dict[str, Any]):
        """Save complete pipeline results."""
        import json

        results_file = self.output_dir / f"complete_results_{results['question_id']}.json"

        # Clean up for JSON serialization
        clean_results = {
            "question_id": results["question_id"],
            "question": results["question"],
            "strategy": results["strategy"],
            "results": {k: v for k, v in results["results"].items() if k != 'raw_output'},
            "evaluation": results["evaluation"]
        }

        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2)

        print(f"\n✓ Complete results saved to: {results_file}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run XAI Pipeline with LangChain Three Agent System"
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
        help="Question ID to process (can be an integer index like '0' or a string ID)"
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        help="URL or path to target model (optional). Can be relative to models_dir or absolute path."
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="VLM model ID (default: Qwen/Qwen2-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./outputs relative to current working directory)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory containing datasets (default: ./dataset relative to current working directory)"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Directory containing models (default: ./models_to_read relative to current working directory)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to image for vision modality questions. If provided, it overrides the path from the dataset."
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = XAIPipelineLangChain(
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
        image_path=args.image_path
    )

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Question ID: {results['question_id']}")
    print(f"Strategy: {results['strategy'].get('strategy_type', 'unknown')}")
    print(f"Explanation: {results['results'].get('explanation', 'N/A')}...")
    print(f"Evaluation: {results['evaluation'].get('overall_rating', 'N/A')}")


if __name__ == "__main__":
    main()

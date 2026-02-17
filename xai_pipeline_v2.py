"""
XAI Pipeline V2 - Using New Modular Architecture

Complete end-to-end XAI pipeline using:
- Modular prompts (prompts/)
- Modular agents (agents/)
- Modular evaluation (evaluation/)
- Strategy faithfulness evaluation with tool attribution
"""

import argparse
import json
import os
import importlib.util
from datetime import datetime
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
from evaluation import get_evaluator, EvaluationResult, ToolAttributionEvaluator, set_masking_output_dir

# Existing imports
from vlm_wrapper import VisionLanguageModel, create_vlm
from DataModelLoader import DataModelLoader


def load_model_loader_module(model_path: str, models_dir: str):
    """
    Dynamically load the appropriate loader module for a given model.

    Args:
        model_path: Path to the model file (e.g., 'vision/stl10_resnet_head.pth')
        models_dir: Base directory for models

    Returns:
        Loaded module with load_model, load_data, predict functions
    """
    model_path = Path(model_path)

    # Get the model name stem (e.g., 'stl10_resnet_head' from 'stl10_resnet_head.pth')
    model_name_stem = model_path.stem

    # Construct loader module path
    # e.g., models_to_read/vision/load_stl10_resnet_head.py
    loader_module_name = f"load_{model_name_stem}.py"

    # Check if model_path is absolute or relative
    if model_path.is_absolute():
        loader_path = model_path.parent / loader_module_name
    else:
        loader_path = Path(models_dir) / model_path.parent / loader_module_name

    if not loader_path.exists():
        raise FileNotFoundError(f"Loader module not found: {loader_path}")

    # Dynamically import the loader module
    spec = importlib.util.spec_from_file_location(loader_path.stem, loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    # Verify required functions exist
    required_funcs = ['load_model', 'load_data', 'predict']
    for func_name in required_funcs:
        if not hasattr(loader_module, func_name):
            raise AttributeError(f"Loader module {loader_path} missing required function: {func_name}")

    return loader_module


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

        # Set output directory for masking utilities (feature_mean_cache, masked_inputs)
        set_masking_output_dir(str(self.output_dir))

        print("\n" + "=" * 70)
        print("INITIALIZING XAI PIPELINE V2")
        print("=" * 70)
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Models directory: {self.models_dir}")
        print(f"Output directory: {self.output_dir}")

        # Initialize VLM
        print("\nInitializing VLM...")
        self.vlm = create_vlm(model_id=vlm_model_id)

        # Initialize three agents using new modular system
        print("\nInitializing Agents (New Architecture)...")
        self.proposer, self.actor, self.critic = create_three_agent_system(
            vlm=self.vlm,
            model=None,  # Will be set after loading target model
            output_dir=str(self.output_dir),
            models_dir=str(self.models_dir)
        )

        # Strategy faithfulness evaluator (lazy initialization)
        self.tool_attribution_evaluator: Optional[ToolAttributionEvaluator] = None

        # Create directories for strategy faithfulness and training data
        self.sf_dir = self.output_dir / "strategy_faithfulness"
        self.sf_dir.mkdir(parents=True, exist_ok=True)

        self.training_data_dir = self.output_dir / "training_data"
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        self.sf_cache_dir = self.output_dir / "strategy_faithfulness_cache"
        self.sf_cache_dir.mkdir(parents=True, exist_ok=True)

        print("\nXAI Pipeline V2 initialized successfully!")

    # =========================================================================
    # Image Path Helpers (for auto-constructing paths from root + dataset + index)
    # =========================================================================

    def _extract_dataset_name_from_path(self, json_path: str) -> str:
        """
        Extract dataset name from JSON file path.

        Examples:
            'stl10_resnet_q4.json' -> 'stl10'
            'stl10_resnet_q1_test.json' -> 'stl10'
            'cub_resnet_q4.json' -> 'cubs'
            'cub_densenet_q1.json' -> 'cubs'

        Args:
            json_path: Path to the question JSON file

        Returns:
            Dataset directory name (e.g., 'stl10', 'cubs')
        """
        import re
        basename = Path(json_path).stem

        # Pattern: {dataset}_{model}_{qtype}[_suffix]
        # Extract first part before underscore+model pattern
        match = re.match(r'^([a-zA-Z0-9]+)_', basename)
        if match:
            dataset = match.group(1).lower()
            # Handle known mappings (dataset name -> directory name)
            dataset_map = {
                'cub': 'cubs',
                'stl10': 'stl10',
                'imdb': 'imdb',
                'snli': 'snli',
                'adult': 'adult',
                'cancer': 'cancer'
            }
            return dataset_map.get(dataset, dataset)
        return 'unknown'

    def _build_image_path(self, image_root: str, dataset_name: str, image_index: int) -> str:
        """
        Auto-construct image path from components.

        Args:
            image_root: Root path, e.g., '/path/to/dataset/image'
            dataset_name: Dataset directory name, e.g., 'stl10'
            image_index: Image index from JSON

        Returns:
            Full path, e.g., '/path/to/dataset/image/stl10/test_2812.png'
        """
        return f"{image_root}/{dataset_name}/test_{image_index}.png"

    # =========================================================================
    # Q4 Specific Methods (instance_A / instance_B format)
    # =========================================================================

    def _load_question_q4(
        self,
        dataset_path: str,
        question_id: str
    ) -> Tuple[Dict[str, Any], "QuestionTemplate"]:
        """
        Load Q4 question with instance_A/instance_B format.

        Q4 JSON format:
        {
            "pair_id": 0,
            "instance_A": {
                "features": {"image_index": 4540, ...},
                "target": {"value": 3, "name": "cat"},
                "prediction": {"value": 3, "name": "cat"}
            },
            "instance_B": {
                "features": {"image_index": 3146, ...},
                "target": {"value": 8, "name": "ship"},
                "prediction": {"value": 8, "name": "ship"}
            },
            "q_type": 4
        }
        """
        print("\n=== Step 1: Loading Q4 Question ===")

        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # Handle both list and dict formats
        if isinstance(dataset, list):
            questions = dataset
        elif isinstance(dataset, dict):
            questions = dataset.get('questions', [dataset])
        else:
            questions = []

        # Find question
        question = None
        try:
            idx = int(question_id)
            if 0 <= idx < len(questions):
                question = questions[idx]
        except ValueError:
            for q in questions:
                if q.get('pair_id') == question_id or q.get('question_id') == question_id:
                    question = q
                    break

        if question is None:
            raise ValueError(f"Q4 question '{question_id}' not found")

        # Validate Q4 format
        if 'instance_A' not in question or 'instance_B' not in question:
            raise ValueError("Q4 question must have instance_A and instance_B")

        # Extract instance information
        instance_a = question['instance_A']
        instance_b = question['instance_B']

        # Normalize to unified structure for agents
        question['is_multi_instance'] = True
        question['is_q4_format'] = True  # Flag for Q4-specific handling
        question['num_instances'] = 2

        # Extract image indices
        image_index_a = instance_a['features']['image_index']
        image_index_b = instance_b['features']['image_index']
        question['image_indices'] = [image_index_a, image_index_b]

        # Use pair_id as row_no for file naming
        question['row_no'] = question.get('pair_id', question_id)

        # Normalize question text
        question['question'] = question.get('example', question.get('q', ''))

        # Set modality
        modality_map = {'image': 'vision', 'text': 'text', 'tabular': 'tabular'}
        question['modality'] = modality_map.get(question.get('modality', 'image'), 'vision')

        question['question_id'] = f"q4_{question.get('pair_id', question_id)}"

        print(f"Q4 Question loaded:")
        print(f"  Instance A: image_index={image_index_a}, pred={instance_a['prediction']['name']}")
        print(f"  Instance B: image_index={image_index_b}, pred={instance_b['prediction']['name']}")

        # Get Q4 template
        q_type = question.get('q_type', 4)
        template = get_question_template(q_type, question['modality'])

        return question, template

    def _load_model_and_data_q4(
        self,
        question: Dict[str, Any],
        model_url: Optional[str],
        image_root: Optional[str] = None
    ) -> Tuple[Optional[Dict], Dict[str, Dict], Dict[str, str], Dict[str, Any]]:
        """
        Load model and data for Q4 with instance_A/instance_B format.

        Returns:
            Tuple of:
            - model_info: Model information dict
            - predictions: {'A': pred_dict, 'B': pred_dict}
            - data_paths: {'A': path, 'B': path}
            - input_tensors: {'A': PIL.Image, 'B': PIL.Image}
        """
        print("\n=== Step 2: Loading Model and Data for Q4 ===")

        instance_a = question['instance_A']
        instance_b = question['instance_B']
        modality = question.get('modality', 'vision')

        model_info = None
        predictions = {'A': None, 'B': None}
        data_paths = {'A': '', 'B': ''}
        input_tensors = {'A': None, 'B': None}

        try:
            # Load model (once)
            if model_url:
                if not os.path.isabs(model_url):
                    local_model_path = self.models_dir / model_url
                    if local_model_path.exists():
                        model_url = str(local_model_path)

                print(f"Loading model: {model_url}")
                loader_module = load_model_loader_module(model_url, str(self.models_dir))
                model, processor = loader_module.load_model(model_url)

                if hasattr(loader_module, 'get_model_info'):
                    extra_info = loader_module.get_model_info(model)
                else:
                    extra_info = {}

                model_info = {
                    "success": True,
                    "model": model,
                    "processor": processor,
                    "model_name": Path(model_url).name,
                    "model_type": "local_pth",
                    "model_path": model_url,
                    "device": str(loader_module.DEVICE if hasattr(loader_module, 'DEVICE') else 'cpu'),
                    "architecture": model.__class__.__name__,
                    "num_classes": extra_info.get('num_classes'),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "label_map": extra_info.get('label_map', {}),
                    "feature_names": extra_info.get('feature_names', [])
                }

                # Initialize DataModelLoader
                self.data_model_loader = DataModelLoader(
                    model_name=Path(model_url).stem,
                    modality=modality
                )

                print("\n=== Initializing XAI Tools ===")
                self.actor.initialize_tools(data_model_loader=self.data_model_loader)
                self.critic.set_model(self.data_model_loader.get_model())

            # Load both instances based on modality
            if modality == 'vision':
                if image_root is None:
                    image_root = str(self.dataset_dir / "image")

                # Get dataset name from question
                dataset_name = question.get('dataset', '').lower().replace('-', '')
                if not dataset_name:
                    dataset_name = self._extract_dataset_name_from_path(
                        question.get('dataset_base_name', '')
                    )
                # Map dataset names
                dataset_map = {'stl10': 'stl10', 'stl-10': 'stl10', 'cubs': 'cubs', 'cub': 'cubs'}
                dataset_name = dataset_map.get(dataset_name, dataset_name)

                split = question.get('split', 'test')

                # Load Instance A
                img_idx_a = instance_a['features']['image_index']
                print(f"\n  Loading Instance A: image_index={img_idx_a}")

                img_path_a = self._build_image_path(image_root, dataset_name, img_idx_a)
                data_paths['A'] = img_path_a
                print(f"    Path: {img_path_a}")

                data_a = self.data_model_loader.load_sample(index=img_idx_a, split=split)
                input_tensors['A'] = data_a.get('image')

                # Make prediction for A
                pred_a = self.data_model_loader.predict(input_tensors['A'])
                if pred_a and pred_a.get('success'):
                    pred_a['ground_truth_idx'] = instance_a['target']['value']
                    pred_a['ground_truth_name'] = instance_a['target']['name']
                    pred_a['expected_prediction'] = instance_a['prediction']
                    print(f"    Prediction: {pred_a.get('predicted_class_name')} "
                          f"(expected: {instance_a['prediction']['name']})")
                predictions['A'] = pred_a

                # Load Instance B
                img_idx_b = instance_b['features']['image_index']
                print(f"\n  Loading Instance B: image_index={img_idx_b}")

                img_path_b = self._build_image_path(image_root, dataset_name, img_idx_b)
                data_paths['B'] = img_path_b
                print(f"    Path: {img_path_b}")

                data_b = self.data_model_loader.load_sample(index=img_idx_b, split=split)
                input_tensors['B'] = data_b.get('image')

                # Make prediction for B
                pred_b = self.data_model_loader.predict(input_tensors['B'])
                if pred_b and pred_b.get('success'):
                    pred_b['ground_truth_idx'] = instance_b['target']['value']
                    pred_b['ground_truth_name'] = instance_b['target']['name']
                    pred_b['expected_prediction'] = instance_b['prediction']
                    print(f"    Prediction: {pred_b.get('predicted_class_name')} "
                          f"(expected: {instance_b['prediction']['name']})")
                predictions['B'] = pred_b

            elif modality == 'text':
                loader_module = self.data_model_loader.loader_module
                model = self.data_model_loader.get_model()
                processor = self.data_model_loader.get_processor()

                for label, instance in [('A', instance_a), ('B', instance_b)]:
                    feat = instance['features']
                    row_no = instance.get('row_no', 'unknown')
                    print(f"\n  Loading Instance {label}: row_no={row_no}")

                    # Determine text format (IMDB vs NLI)
                    if 'premise' in feat and 'hypothesis' in feat:
                        text_input = feat  # NLI dict
                    elif 'text' in feat:
                        text_input = feat.get('text', '')
                    else:
                        text_input = feat

                    data = loader_module.load_data(text_input)
                    self.data_model_loader.current_sample_data = data

                    data_paths[label] = f"text_row_{row_no}"

                    # Store text for evaluation
                    if 'premise' in feat:
                        input_tensors[label] = {'premise': feat['premise'], 'hypothesis': feat['hypothesis']}
                    else:
                        input_tensors[label] = data.get('text', text_input)

                    # Make prediction
                    if model is not None:
                        pred = loader_module.predict(model=model, text_input=data, tokenizer=processor)
                        if pred and pred.get('success'):
                            pred['ground_truth_idx'] = instance['target']['value']
                            pred['ground_truth_name'] = instance['target'].get('label', instance['target'].get('name'))
                            pred['expected_prediction'] = instance['prediction']
                            print(f"    Prediction: {pred.get('predicted_class_name')} "
                                  f"(expected: {instance['prediction'].get('label', instance['prediction'].get('name'))})")
                        predictions[label] = pred

            elif modality == 'tabular':
                split = question.get('split', 'test')
                # Q4 uses full-dataset indices for some datasets (e.g. breast cancer)
                q_type = question.get('q_type', 4)
                if q_type in (4, 8, 9, 10):
                    split = 'full'

                for label, instance in [('A', instance_a), ('B', instance_b)]:
                    row_no = instance.get('row_no', 0)
                    print(f"\n  Loading Instance {label}: row_no={row_no}")

                    data = self.data_model_loader.load_sample(index=row_no, split=split)
                    data_paths[label] = f"tabular_row_{row_no}"
                    input_tensors[label] = data.get('features')

                    # Make prediction
                    loader_module = self.data_model_loader.loader_module
                    model = self.data_model_loader.get_model()
                    processor = self.data_model_loader.get_processor()
                    if model is not None:
                        pred = loader_module.predict(
                            model=model,
                            input_data=data.get('features'),
                            preprocessor=processor
                        )
                        if pred and pred.get('success'):
                            pred['ground_truth_idx'] = instance['target']['value']
                            pred['ground_truth_name'] = instance['target'].get('label', instance['target'].get('name'))
                            pred['expected_prediction'] = instance['prediction']
                            print(f"    Prediction: {pred.get('predicted_class_name')} "
                                  f"(expected: {instance['prediction'].get('label', instance['prediction'].get('name'))})")
                        predictions[label] = pred

        except Exception as e:
            print(f"Warning: Failed to load model/data for Q4: {e}")
            import traceback
            traceback.print_exc()

        return model_info, predictions, data_paths, input_tensors

    def run_q4(
        self,
        question_dataset_path: str,
        question_id: str,
        target_model_url: Optional[str] = None,
        image_root: Optional[str] = None,
        evaluate_faithfulness: bool = True,
        faithfulness_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run XAI pipeline for Q4 (contrastive instances).

        This method handles the Q4-specific instance_A/instance_B format.

        Args:
            question_dataset_path: Path to Q4 question dataset JSON
            question_id: Question ID (pair_id) to process
            target_model_url: URL/path of target model
            image_root: Root directory for images
            evaluate_faithfulness: Whether to run faithfulness evaluation
            faithfulness_threshold: Threshold for passing faithfulness

        Returns:
            Complete results dict
        """
        print("\n" + "=" * 70)
        print("RUNNING XAI PIPELINE V2 FOR Q4 (CONTRASTIVE INSTANCES)")
        print("=" * 70)

        dataset_base_name = Path(question_dataset_path).stem

        # Step 1: Load Q4 question
        question, template = self._load_question_q4(question_dataset_path, question_id)
        question['dataset_base_name'] = dataset_base_name

        # Step 2: Load model and data for both instances
        model_info, predictions, data_paths, input_tensors = self._load_model_and_data_q4(
            question, target_model_url, image_root
        )

        # Build instances list for prompts
        # Include index so actor can reload the correct sample before tool execution
        # Vision uses features.image_index; text/tabular uses instance.row_no
        instance_a = question['instance_A']
        instance_b = question['instance_B']
        modality = question.get('modality', 'vision')

        def _get_instance_index(inst):
            if modality == 'vision':
                return inst['features']['image_index']
            return inst.get('row_no')

        instances = [
            {
                'prediction': predictions['A'],
                'path': data_paths['A'],
                'label': 'A',
                'image_index': _get_instance_index(instance_a),
                'features': instance_a.get('features', {}),
                'split': question.get('split', 'test')
            },
            {
                'prediction': predictions['B'],
                'path': data_paths['B'],
                'label': 'B',
                'image_index': _get_instance_index(instance_b),
                'features': instance_b.get('features', {}),
                'split': question.get('split', 'test')
            }
        ]

        # Step 3: Proposer generates strategy
        print("\n=== Step 3: Proposer Agent (Q4) ===")
        strategy = self.proposer.run_q4(
            question=question,
            question_template=template,
            model_info=model_info,
            instances=instances
        )

        # Step 4: Actor executes and explains
        print("\n=== Step 4: Actor Agent (Q4) ===")
        results = self.actor.run_q4(
            strategy=strategy,
            question=question,
            question_template=template,
            instances=instances,
            model_info=model_info
        )

        # Step 5: Validate output format
        print("\n=== Step 5: Output Validation ===")
        is_valid, validation_errors = template.validate_output(results)
        if is_valid:
            print("  Output format: VALID")
        else:
            print(f"  Output format: INVALID - {validation_errors}")

        # Step 6: Critic evaluates (Q4-specific)
        print("\n=== Step 6: Critic Agent (Q4 Faithfulness) ===")
        evaluation = None
        if evaluate_faithfulness and model_info and model_info.get('model'):
            evaluation = self.critic.run_q4(
                results=results,
                question=question,
                inputs={'A': input_tensors['A'], 'B': input_tensors['B']},
                predictions={'A': predictions['A'], 'B': predictions['B']},
                processor=model_info.get('processor'),
                device=model_info.get('device', 'cuda')
            )
        else:
            print("  Skipping faithfulness evaluation")
            evaluation = {"status": "skipped"}

        # Combine results
        complete_results = {
            "question_id": question['question_id'],
            "question_type": "Q4",
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

        # Save results
        self._save_complete_results(complete_results)

        print("\n" + "=" * 70)
        print("Q4 PIPELINE COMPLETE")
        print("=" * 70)

        return complete_results

    # =========================================================================
    # Q9/Q10 Multi-Instance Methods
    # =========================================================================

    def run_multi_instance(
        self,
        question: Dict[str, Any],
        template: "QuestionTemplate",
        target_model_url: Optional[str] = None,
        evaluate_faithfulness: bool = True,
        faithfulness_threshold: float = 0.1,
        enable_improvement: bool = True,
        enable_sf: bool = True,
        sf_max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run XAI pipeline for Q9/Q10 multi-instance questions.

        These questions use unified array format with image_indices[], target[], predicted[].
        Each instance is loaded and predicted separately, then passed together
        to the proposer and actor for comparative analysis.
        """
        q_type = question.get('q_type')
        print("\n" + "=" * 70)
        print(f"RUNNING XAI PIPELINE V2 FOR Q{q_type} (MULTI-INSTANCE)")
        print("=" * 70)

        # Step 2: Load model and data for all instances
        model_info, predictions_list, data_paths, input_tensors = self._load_model_and_data_multi(
            question, target_model_url
        )

        # Build instances list for prompts
        image_indices = question.get('image_indices', question.get('row_no', []))
        split = question.get('split', 'test')
        instances = [
            {
                'prediction': pred,
                'path': path,
                'label': str(i),
                'image_index': image_indices[i],
                'split': split
            }
            for i, (pred, path) in enumerate(zip(predictions_list, data_paths))
        ]

        # Step 3: Proposer generates strategy
        print(f"\n=== Step 3: Proposer Agent (Q{q_type}) ===")
        strategy = self.proposer.run(
            question=question,
            question_template=template,
            model_info=model_info,
            input_paths=data_paths,
            predictions=predictions_list
        )

        # Step 4: Actor executes and explains
        print(f"\n=== Step 4: Actor Agent (Q{q_type}) ===")
        results = self.actor.run(
            strategy=strategy,
            question=question,
            question_template=template,
            model_info=model_info,
            input_paths=data_paths,
            predictions=predictions_list
        )

        # Step 5: Validate output format
        print("\n=== Step 5: Output Validation ===")
        is_valid, validation_errors = template.validate_output(results)
        if is_valid:
            print("  Output format: VALID")
        else:
            print(f"  Output format: INVALID - {validation_errors}")

        # Step 6: Critic evaluates explanation faithfulness
        print(f"\n=== Step 6: Critic Agent (Q{q_type} Faithfulness) ===")
        evaluation = None
        if evaluate_faithfulness and model_info and model_info.get('model'):
            # Build ground_truths list from question targets
            targets = question.get('targets', question.get('target', []))
            ground_truths = []
            for t in targets:
                if isinstance(t, dict):
                    ground_truths.append(t.get('value', t.get('label')))
                else:
                    ground_truths.append(t)

            evaluation = self.critic.run(
                results=results,
                question=question,
                original_input=input_tensors[0] if input_tensors else None,
                original_prediction=predictions_list[0] if predictions_list else None,
                ground_truth=question.get('ground_truth'),
                processor=model_info.get('processor'),
                device=model_info.get('device', 'cuda'),
                class_names=model_info.get('label_map', {}),
                feature_names=model_info.get('feature_names', []),
                inputs=input_tensors,
                predictions=predictions_list,
                ground_truths=ground_truths
            )
        else:
            print("  Skipping faithfulness evaluation")
            evaluation = {"status": "skipped"}

        # Combine initial results
        complete_results = {
            "question_id": question.get('question_id', f"q{q_type}_{question.get('dataset_base_name', '')}"),
            "question_type": f"Q{q_type}",
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

        # Step 7: Check explanation faithfulness and decide next steps
        sf_result = None
        if evaluate_faithfulness and model_info and model_info.get('model'):
            faithfulness_result = evaluation.get('faithfulness', {})
            faithfulness_score = faithfulness_result.get('score')
            if faithfulness_score is None:
                faithfulness_score = 0.0

            faithfulness_passed = faithfulness_result.get('passed', faithfulness_score >= faithfulness_threshold)
            evaluator_threshold = faithfulness_result.get('details', {}).get('threshold', faithfulness_threshold)

            if faithfulness_passed:
                print("\n=== Step 7: Explanation Faithfulness Check ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) passed (threshold: {evaluator_threshold})")
                print("  Saving training datapoint for passed sample.")

                self._save_training_datapoint_passed(
                    question=question,
                    original_strategy=strategy,
                    original_results=results,
                    original_evaluation=evaluation
                )
            else:
                print("\n=== Step 7: Strategy Faithfulness Evaluation ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) failed (threshold: {evaluator_threshold})")

                if enable_sf:
                    print("  Running strategy faithfulness evaluation...")

                    if self.tool_attribution_evaluator is None:
                        self.tool_attribution_evaluator = ToolAttributionEvaluator(
                            cache_dir=str(self.sf_cache_dir),
                            output_dir=str(self.sf_dir)
                        )
                        self.tool_attribution_evaluator.set_agents(self.actor, self.critic)

                    original_tool_results = {
                        'tool_results': results.get('tool_results', {}),
                        'visualization_paths': results.get('visualization_paths', []),
                        'tool_results_summary': results.get('tool_results_summary', '')
                    }

                    sf_result = self.tool_attribution_evaluator.compute_tool_importance(
                        original_strategy=strategy,
                        original_faithfulness=faithfulness_score,
                        original_tool_results=original_tool_results,
                        question=question,
                        question_template=template,
                        input_path=data_paths[0] if data_paths else "",
                        model_info=model_info,
                        prediction=predictions_list[0] if predictions_list else {},
                        input_tensor=input_tensors[0] if input_tensors else None,
                        faithfulness_threshold=faithfulness_threshold,
                        processor=model_info.get('processor'),
                        device=model_info.get('device', 'cuda'),
                        max_samples=sf_max_samples,
                        input_paths=data_paths,
                        predictions=predictions_list,
                        input_tensors=input_tensors,
                        ground_truths=ground_truths
                    )

                    self.tool_attribution_evaluator.save_result(sf_result, question)
                    complete_results['strategy_faithfulness'] = sf_result.to_dict()
                else:
                    print("  Strategy faithfulness evaluation skipped (--no-sf).")

                # Step 8: Improvement Phase
                if enable_improvement:
                    print("\n=== Step 8: Improvement Phase ===")

                    print("  Generating Critic reflections...")
                    proposer_reflection, actor_reflection = self.critic.generate_reflections(
                        strategy=strategy,
                        results=results,
                        question=question,
                        faithfulness_result=evaluation.get('faithfulness', {}),
                        tool_importance_scores=sf_result.tool_importance_scores if sf_result else {},
                        threshold=evaluator_threshold
                    )

                    self.critic.save_reflections(proposer_reflection, actor_reflection, question)

                    # Proposer generates improved strategy
                    print("\n  Proposer generating improved strategy...")
                    improved_strategy = self.proposer.run_with_reflection(
                        question=question,
                        question_template=template,
                        model_info=model_info,
                        input_path=data_paths[0] if data_paths else None,
                        prediction=predictions_list[0] if predictions_list else None,
                        proposer_reflection=proposer_reflection,
                        original_strategy=strategy,
                        input_paths=data_paths,
                        predictions=predictions_list
                    )

                    # Actor generates improved explanation
                    print("\n  Actor generating improved explanation...")
                    improved_results = self.actor.run_with_reflection(
                        strategy=improved_strategy,
                        question=question,
                        question_template=template,
                        input_path=data_paths[0] if data_paths else None,
                        model_info=model_info,
                        prediction=predictions_list[0] if predictions_list else None,
                        actor_reflection=actor_reflection,
                        original_results=results,
                        input_paths=data_paths,
                        predictions=predictions_list
                    )

                    # Evaluate improved explanation
                    print("\n  Evaluating improved explanation...")
                    improved_evaluation = self.critic.run(
                        results=improved_results,
                        question=question,
                        original_input=input_tensors[0] if input_tensors else None,
                        original_prediction=predictions_list[0] if predictions_list else None,
                        ground_truth=question.get('ground_truth'),
                        processor=model_info.get('processor'),
                        device=model_info.get('device', 'cuda'),
                        class_names=model_info.get('label_map', {}),
                        feature_names=model_info.get('feature_names', []),
                        inputs=input_tensors,
                        predictions=predictions_list,
                        ground_truths=ground_truths,
                        suffix="_improved"
                    )

                    improved_faithfulness = improved_evaluation.get('faithfulness', {}).get('score', 0.0)
                    if improved_faithfulness is None:
                        improved_faithfulness = 0.0

                    improvement_delta = improved_faithfulness - faithfulness_score

                    complete_results['reflections'] = {
                        'proposer_reflection': proposer_reflection,
                        'actor_reflection': actor_reflection
                    }
                    complete_results['improved'] = {
                        'strategy': improved_strategy,
                        'results': {k: v for k, v in improved_results.items()
                                  if k not in ['raw_output', 'model']},
                        'evaluation': improved_evaluation
                    }
                    complete_results['improvement_metrics'] = {
                        'original_faithfulness': faithfulness_score,
                        'improved_faithfulness': improved_faithfulness,
                        'delta': improvement_delta,
                        'improved': improvement_delta > 0
                    }

                    # Save training datapoint
                    self._save_training_datapoint(
                        question=question,
                        original_strategy=strategy,
                        original_results=results,
                        original_evaluation=evaluation,
                        strategy_faithfulness=sf_result,
                        proposer_reflection=proposer_reflection,
                        actor_reflection=actor_reflection,
                        improved_strategy=improved_strategy,
                        improved_results=improved_results,
                        improved_evaluation=improved_evaluation
                    )

                    print(f"\n  Improvement complete:")
                    print(f"    Original faithfulness: {faithfulness_score:.4f}")
                    print(f"    Improved faithfulness: {improved_faithfulness:.4f}")
                    print(f"    Delta: {improvement_delta:+.4f}")
                else:
                    print("\n  Improvement disabled. Skipping improvement phase.")

        # Save complete results
        self._save_complete_results(complete_results)

        print("\n" + "=" * 70)
        print(f"Q{q_type} MULTI-INSTANCE PIPELINE COMPLETE")
        print("=" * 70)

        return complete_results

    def run(
        self,
        question_dataset_path: str,
        question_id: str,
        target_model_url: Optional[str] = None,
        image_path: Optional[str] = None,
        evaluate_faithfulness: bool = True,
        faithfulness_threshold: float = 0.1,
        enable_improvement: bool = True,
        enable_sf: bool = True,
        sf_max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete XAI pipeline for a single question.

        Includes strategy faithfulness evaluation and improvement
        loop when explanation faithfulness is below threshold.

        Args:
            question_dataset_path: Path to question dataset JSON
            question_id: ID of question to process
            target_model_url: URL/path of target model
            image_path: Path to image for vision modality
            evaluate_faithfulness: Whether to run faithfulness evaluation
            faithfulness_threshold: Threshold for passing faithfulness (default: 0.1)
            enable_improvement: Whether to run improvement when below threshold
            enable_sf: Whether to run strategy faithfulness evaluation

        Returns:
            Complete results dict
        """
        print("\n" + "=" * 70)
        print(f"RUNNING XAI PIPELINE V2 FOR QUESTION: {question_id}")
        print("=" * 70)

        # Extract dataset base name from path (e.g., stl10_resnet_q1 from stl10_resnet_q1.json)
        dataset_base_name = Path(question_dataset_path).stem

        # Step 1: Load question from dataset
        question, template = self._load_question(question_dataset_path, question_id)

        # Add dataset_base_name to question for consistent naming across all outputs
        question['dataset_base_name'] = dataset_base_name

        # Auto-detect Q4 format and route to run_q4
        if question.get('is_q4_format') or question.get('q_type') == 4:
            print("  Detected Q4 format, routing to run_q4...")
            return self.run_q4(
                question_dataset_path=question_dataset_path,
                question_id=question_id,
                target_model_url=target_model_url,
                evaluate_faithfulness=evaluate_faithfulness,
                faithfulness_threshold=faithfulness_threshold
            )

        # Auto-detect Q9/Q10 multi-instance format and route to run_multi_instance
        if question.get('is_multi_instance') and question.get('q_type') in [9, 10]:
            print(f"  Detected Q{question['q_type']} multi-instance format, routing to run_multi_instance...")
            return self.run_multi_instance(
                question=question,
                template=template,
                target_model_url=target_model_url,
                evaluate_faithfulness=evaluate_faithfulness,
                faithfulness_threshold=faithfulness_threshold,
                enable_improvement=enable_improvement,
                enable_sf=enable_sf,
                sf_max_samples=sf_max_samples
            )

        # Step 2: Load target model and data
        model_info, prediction, data_path, input_tensor = self._load_model_and_data(
            question, target_model_url, image_path=image_path
        )

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

        # Step 7: Critic evaluates explanation faithfulness (if model available)
        print("\n=== Step 6: Critic Agent (Explanation Faithfulness) ===")
        evaluation = None
        if evaluate_faithfulness and model_info and model_info.get('model'):
            evaluation = self.critic.run(
                results=results,
                question=question,
                original_input=input_tensor,
                original_prediction=prediction,
                ground_truth=question.get('ground_truth'),
                processor=model_info.get('processor'),
                device=model_info.get('device', 'cuda'),
                class_names=model_info.get('label_map', {}),
                feature_names=model_info.get('feature_names', [])
            )
        else:
            print("  Skipping faithfulness evaluation (no model or disabled)")
            evaluation = {"status": "skipped"}

        # Combine initial results
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

        # Step 7: Check explanation faithfulness and decide next steps
        sf_result = None
        if evaluate_faithfulness and model_info and model_info.get('model'):
            faithfulness_result = evaluation.get('faithfulness', {})
            faithfulness_score = faithfulness_result.get('score')
            if faithfulness_score is None:
                faithfulness_score = 0.0

            # Use evaluator's passed field (respects question-specific threshold)
            # Fall back to generic threshold if passed field is not available
            faithfulness_passed = faithfulness_result.get('passed', faithfulness_score >= faithfulness_threshold)

            # Get threshold from evaluator for logging (if available)
            evaluator_threshold = faithfulness_result.get('details', {}).get('threshold', faithfulness_threshold)

            if faithfulness_passed:
                # Explanation faithfulness is good enough - skip strategy faithfulness and reflection
                print("\n=== Step 7: Explanation Faithfulness Check ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) passed (threshold: {evaluator_threshold})")
                print("  Explanation is good enough. Skipping strategy faithfulness evaluation and reflection.")
                print("  Saving training datapoint for passed sample.")

                # Save training datapoint for passed samples
                self._save_training_datapoint_passed(
                    question=question,
                    original_strategy=strategy,
                    original_results=results,
                    original_evaluation=evaluation
                )
            else:
                # Explanation faithfulness is below threshold
                print("\n=== Step 7: Strategy Faithfulness Evaluation ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) failed (threshold: {evaluator_threshold})")

                if enable_sf:
                    print("  Running strategy faithfulness evaluation...")

                    # Initialize tool attribution evaluator if needed
                    if self.tool_attribution_evaluator is None:
                        self.tool_attribution_evaluator = ToolAttributionEvaluator(
                            cache_dir=str(self.sf_cache_dir),
                            output_dir=str(self.sf_dir)
                        )
                        self.tool_attribution_evaluator.set_agents(self.actor, self.critic)

                    # Build original_tool_results from Actor's results
                    original_tool_results = {
                        'tool_results': results.get('tool_results', {}),
                        'visualization_paths': results.get('visualization_paths', []),
                        'tool_results_summary': "; ".join([
                            f"{k}: {'success' if v.get('success') else 'failed'}"
                            for k, v in results.get('tool_results', {}).items()
                            if isinstance(v, dict)
                        ])
                    }

                    # Compute tool importance scores (reuses tool results, only re-runs feature extraction)
                    sf_result = self.tool_attribution_evaluator.compute_tool_importance(
                        original_strategy=strategy,
                        original_faithfulness=faithfulness_score,
                        original_tool_results=original_tool_results,
                        question=question,
                        question_template=template,
                        input_path=data_path,
                        model_info=model_info,
                        prediction=prediction,
                        input_tensor=input_tensor,
                        faithfulness_threshold=faithfulness_threshold,
                        processor=model_info.get('processor'),
                        device=model_info.get('device', 'cuda'),
                        max_samples=sf_max_samples
                    )

                    # Save strategy faithfulness result
                    self.tool_attribution_evaluator.save_result(sf_result, question)
                    complete_results['strategy_faithfulness'] = sf_result.to_dict()
                else:
                    print("  Strategy faithfulness evaluation skipped (--no-sf).")

                # Step 8: Improvement Phase (only if enabled and faithfulness below threshold)
                if enable_improvement:
                    print("\n=== Step 8: Improvement Phase ===")

                    # Generate reflections for both agents
                    print("  Generating Critic reflections...")
                    proposer_reflection, actor_reflection = self.critic.generate_reflections(
                        strategy=strategy,
                        results=results,
                        question=question,
                        faithfulness_result=evaluation.get('faithfulness', {}),
                        tool_importance_scores=sf_result.tool_importance_scores if sf_result else {},
                        threshold=evaluator_threshold
                    )

                    # Save reflections
                    self.critic.save_reflections(proposer_reflection, actor_reflection, question)

                    # Proposer generates improved strategy
                    print("\n  Proposer generating improved strategy...")
                    improved_strategy = self.proposer.run_with_reflection(
                        question=question,
                        question_template=template,
                        model_info=model_info,
                        input_path=data_path,
                        prediction=prediction,
                        proposer_reflection=proposer_reflection,
                        original_strategy=strategy
                    )

                    # Actor generates improved explanation
                    print("\n  Actor generating improved explanation...")
                    improved_results = self.actor.run_with_reflection(
                        strategy=improved_strategy,
                        question=question,
                        question_template=template,
                        input_path=data_path,
                        model_info=model_info,
                        prediction=prediction,
                        actor_reflection=actor_reflection,
                        original_results=results
                    )

                    # Evaluate improved explanation
                    print("\n  Evaluating improved explanation...")
                    improved_evaluation = self.critic.run(
                        results=improved_results,
                        question=question,
                        original_input=input_tensor,
                        original_prediction=prediction,
                        ground_truth=question.get('ground_truth'),
                        processor=model_info.get('processor'),
                        device=model_info.get('device', 'cuda'),
                        class_names=model_info.get('label_map', {}),
                        feature_names=model_info.get('feature_names', []),
                        suffix="_improved"
                    )

                    # Calculate improvement metrics
                    improved_faithfulness = improved_evaluation.get('faithfulness', {}).get('score', 0.0)
                    if improved_faithfulness is None:
                        improved_faithfulness = 0.0

                    improvement_delta = improved_faithfulness - faithfulness_score

                    # Add improvement results to complete_results
                    complete_results['reflections'] = {
                        'proposer_reflection': proposer_reflection,
                        'actor_reflection': actor_reflection
                    }
                    complete_results['improved'] = {
                        'strategy': improved_strategy,
                        'results': {k: v for k, v in improved_results.items()
                                  if k not in ['raw_output', 'model']},
                        'evaluation': improved_evaluation
                    }
                    complete_results['improvement_metrics'] = {
                        'original_faithfulness': faithfulness_score,
                        'improved_faithfulness': improved_faithfulness,
                        'delta': improvement_delta,
                        'improved': improvement_delta > 0
                    }

                    # Save training datapoint
                    self._save_training_datapoint(
                        question=question,
                        original_strategy=strategy,
                        original_results=results,
                        original_evaluation=evaluation,
                        strategy_faithfulness=sf_result,
                        proposer_reflection=proposer_reflection,
                        actor_reflection=actor_reflection,
                        improved_strategy=improved_strategy,
                        improved_results=improved_results,
                        improved_evaluation=improved_evaluation
                    )

                    print(f"\n  Improvement complete:")
                    print(f"    Original faithfulness: {faithfulness_score:.4f}")
                    print(f"    Improved faithfulness: {improved_faithfulness:.4f}")
                    print(f"    Delta: {improvement_delta:+.4f}")
                else:
                    print("\n  Improvement disabled. Skipping improvement phase.")

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
        question['question'] = question.get('example', question.get('q', ''))
        # Infer modality from dataset
        modality = question.get('modality')
        
        question['question_id'] = f"q{question_id}"

        print(f"Question: {question.get('question', '')}")

        # Get question type and modality
        q_type = question.get('q_type')
        question['modality'] = modality

        # Detect multi-instance questions (Q4, Q9, Q10 or row_no is array)
        # Unified format uses: row_no (array), image_indices (array), target (array), predicted (array)
        # Q4 special format uses: instance_A/instance_B with nested features.image_index
        row_no = question.get('row_no')
        has_instance_ab = 'instance_A' in question and 'instance_B' in question
        is_multi_instance = (
            q_type in [4, 9, 10] or
            isinstance(row_no, list) or
            has_instance_ab
        )

        if is_multi_instance:
            question['is_multi_instance'] = True

            # Handle Q4 instance_A/instance_B format
            if has_instance_ab:
                instance_a = question['instance_A']
                instance_b = question['instance_B']

                # Extract image indices from nested structure
                image_index_a = instance_a.get('features', {}).get('image_index')
                image_index_b = instance_b.get('features', {}).get('image_index')
                image_indices = [image_index_a, image_index_b]

                # Extract targets and predictions from instance format
                targets = [
                    instance_a.get('target', {}),
                    instance_b.get('target', {})
                ]
                predictions = [
                    instance_a.get('prediction', {}),
                    instance_b.get('prediction', {})
                ]

                question['is_q4_format'] = True
            else:
                # Extract image indices from unified format
                # Q9/Q10 benchmarks may use 'row_idx', 'row_no', or 'image_indices'
                image_indices = question.get('image_indices',
                                    question.get('row_idx',
                                        question.get('row_no', [])))
                if not isinstance(image_indices, list):
                    image_indices = [image_indices]

                # Extract targets and predictions (arrays)
                targets = question.get('target', [])
                predictions = question.get('predicted', [])
                if not isinstance(targets, list):
                    targets = [targets]
                if not isinstance(predictions, list):
                    predictions = [predictions]

            question['image_indices'] = image_indices
            question['num_instances'] = len(image_indices)
            question['targets'] = targets
            question['predictions'] = predictions

            print(f"  Multi-instance question (Q{q_type}): {question['num_instances']} instances")
            print(f"    Image indices: {image_indices}")
            pred_labels = []
            for p in predictions:
                if isinstance(p, dict):
                    pred_labels.append(p.get('name', p.get('label', p.get('value', 'unknown'))))
                else:
                    pred_labels.append(str(p))
            print(f"    Predictions: {pred_labels}")
        else:
            question['is_multi_instance'] = False
            question['num_instances'] = 1

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
        """
        Load target model and input data using modular loaders.

        Uses the loader module from models_to_read/{modality}/load_{model_name}.py
        which provides load_model(), load_data(), and predict() functions.
        """
        print("\n=== Step 2: Loading Model and Data (Modular Loader) ===")

        modality = question.get('modality')
        dataset_name = question.get('dataset', '')

        model_info = None
        prediction = None
        loaded_data_path = None
        input_tensor = None
        loader_module = None

        try:
            # Load model using modular loader
            if model_url:
                # Resolve relative path
                if not os.path.isabs(model_url):
                    local_model_path = self.models_dir / model_url
                    if local_model_path.exists():
                        model_url = str(local_model_path)

                print(f"Loading model: {model_url}")

                # Load the appropriate loader module
                loader_module = load_model_loader_module(model_url, str(self.models_dir))
                print(f"Using loader module: {loader_module.__name__}")

                # Load model using the loader
                model, processor = loader_module.load_model(model_url)

                # Get model info
                if hasattr(loader_module, 'get_model_info'):
                    extra_info = loader_module.get_model_info(model)
                else:
                    extra_info = {}

                model_info = {
                    "success": True,
                    "model": model,
                    "processor": processor,
                    "model_name": Path(model_url).name,
                    "model_type": "local_pth",
                    "model_path": model_url,
                    "device": str(loader_module.DEVICE if hasattr(loader_module, 'DEVICE') else 'cpu'),
                    "architecture": model.__class__.__name__,
                    "num_classes": extra_info.get('num_classes'),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "label_map": extra_info.get('label_map', {}),
                    "feature_names": extra_info.get('feature_names', [])
                }

                print(f"Model loaded: {model_info['architecture']}")

                # Instantiate the new DataModelLoader
                self.data_model_loader = DataModelLoader(
                    model_name=Path(model_url).stem,
                    modality=modality
                )
                
                print("\n=== Initializing XAI Tools ===")
                self.actor.initialize_tools(data_model_loader=self.data_model_loader)
                # Set model for critic
                self.critic.set_model(self.data_model_loader.get_model())

            # Load data based on modality using loader module
            if modality == 'vision' and loader_module:
                # Check if image_path is provided directly
                if image_path and os.path.exists(image_path):
                    print(f"Loading vision data from image_path: {image_path}")
                    from PIL import Image
                    input_tensor = Image.open(image_path).convert('RGB')
                    loaded_data_path = image_path
                    print(f"Data loaded: {loaded_data_path}")

                    # Store in data_model_loader for consistency
                    self.data_model_loader.current_sample_data = {
                        'image': input_tensor,
                        'image_path': image_path
                    }
                else:
                    # Get sample index from question
                    features = question.get('features', {})
                    sample_index = features.get(
                        'image_index',
                        question.get('metadata', {}).get('row_no', question.get('row_no', 0))
                    )
                    split = question.get('split', 'test')

                    print(f"Loading vision data: index={sample_index}, split={split}")

                    # Use the new DataModelLoader to load the sample
                    data = self.data_model_loader.load_sample(index=sample_index, split=split)
                    input_tensor = data.get('image') # This is the PIL image

                    loaded_data_path = f"dataset_index_{sample_index}"
                    print(f"Data loaded: {loaded_data_path}")
                    print(f"  Ground truth: {data.get('label_name')} (class {data.get('label')})")

                # Make prediction using the loader's predict method
                if self.data_model_loader:
                    print("Making prediction...")
                    prediction = self.data_model_loader.predict(input_tensor)
                    if prediction and prediction.get('success'):
                        print(f"Prediction: {prediction.get('predicted_class_name')} "
                              f"(class {prediction.get('predicted_class_idx')}, "
                              f"confidence: {prediction.get('confidence', 0.0):.4f})")

                        # Add ground truth to prediction for comparison (only if loaded from dataset)
                        if not image_path:
                            prediction['ground_truth_idx'] = data.get('label')
                            prediction['ground_truth_name'] = data.get('label_name')
                            prediction['correct'] = prediction.get('predicted_class_idx') == data.get('label')
                    else:
                        print(f"Warning: Prediction failed: {prediction.get('error') if prediction else 'Unknown'}")

            elif modality == 'text' and loader_module:
                # Get text data from question
                features = question.get('features', {})

                if 'premise' in features and 'hypothesis' in features:
                    # SNLI format
                    text_input = features
                elif 'review_text' in features:
                    # IMDB format
                    text_input = features.get('review_text', '')
                else:
                    text_input = features

                print(f"Loading text data")

                # Use loader's load_data function
                data = loader_module.load_data(text_input)

                # Store in current_sample_data so tools and evaluators can access it
                self.data_model_loader.current_sample_data = data

                # Store for evaluation
                if 'premise' in features:
                    input_tensor = {'premise': features.get('premise'), 'hypothesis': features.get('hypothesis')}
                    self.data_model_loader.current_text_data = input_tensor
                else:
                    input_tensor = data.get('text', text_input)
                    self.data_model_loader.current_text_data = input_tensor

                self.data_model_loader.current_data_type = 'text'
                loaded_data_path = "text_input"
                print("Text data loaded")

                # Make prediction
                if model_info and model_info.get('model'):
                    print("Making prediction...")
                    prediction = loader_module.predict(
                        model=model_info['model'],
                        text_input=data,
                        tokenizer=model_info.get('processor')
                    )
                    if prediction and prediction.get('success'):
                        print(f"Prediction: {prediction.get('predicted_class_name')} "
                              f"(confidence: {prediction.get('confidence', 0.0):.4f})")
                    else:
                        print(f"Warning: Prediction failed: {prediction.get('error') if prediction else 'Unknown'}")

            elif modality == 'tabular' and loader_module:
                # Tabular data handling: load by row_no from dataset
                row_no = question.get('row_no', question.get('metadata', {}).get('row_no', 0))
                split = question.get('split', 'test')
                # Q8-Q10 (spurious features) use full-dataset indices for some datasets
                q_type = question.get('q_type', 1)
                if q_type in (8, 9, 10):
                    split = 'full'

                print(f"Loading tabular data: row_no={row_no}, split={split}")
                data = self.data_model_loader.load_sample(index=row_no, split=split)
                input_tensor = data.get('features')  # Preprocessed feature tensor
                loaded_data_path = f"tabular_index_{row_no}"
                print(f"Tabular data loaded: {len(data.get('feature_names', []))} features")
                print(f"  Ground truth: {data.get('label_name')} (class {data.get('label')})")

                # Make prediction
                if model_info and model_info.get('model'):
                    print("Making prediction...")
                    prediction = loader_module.predict(
                        model=model_info['model'],
                        input_data=input_tensor,
                        preprocessor=model_info.get('processor')
                    )
                    if prediction and prediction.get('success'):
                        print(f"Prediction: {prediction.get('predicted_class_name')} "
                              f"(confidence: {prediction.get('confidence', 0.0):.4f})")
                        # Add ground truth
                        prediction['ground_truth_idx'] = data.get('label')
                        prediction['ground_truth_name'] = data.get('label_name')
                        prediction['correct'] = prediction.get('predicted_class_idx') == data.get('label')
                    else:
                        print(f"Warning: Prediction failed: {prediction.get('error') if prediction else 'Unknown'}")

        except Exception as e:
            print(f"Warning: Failed to load model/data: {e}")
            import traceback
            traceback.print_exc()

        return model_info, prediction, loaded_data_path, input_tensor

    def _load_model_and_data_multi(
        self,
        question: Dict[str, Any],
        model_url: Optional[str],
        image_root: Optional[str] = None
    ) -> Tuple[Optional[Dict], List[Dict], List[str], List[Any]]:
        """
        Load target model and ALL input instances for multi-instance questions (Q4, Q9, Q10).

        Works with unified format:
        - image_indices: [idx1, idx2, ...] - array of image indices
        - targets: [{value, label}, ...] - array of ground truth
        - predictions: [{value, label}, ...] - array of predictions

        Args:
            question: Question dict with image_indices, targets, predictions arrays
            model_url: URL/path of target model
            image_root: Root directory for images (optional, defaults to dataset_dir/image)

        Returns:
            Tuple of:
            - model_info: Model information dict
            - predictions: List of prediction dicts for each instance
            - data_paths: List of image paths for each instance
            - input_tensors: List of PIL Images for each instance
        """
        num_instances = question.get('num_instances', 1)
        q_type = question.get('q_type', 1)
        print(f"\n=== Step 2: Loading Model and Data for Multi-Instance (Q{q_type}, {num_instances} instances) ===")

        modality = question.get('modality', 'vision')
        model_info = None
        predictions = []
        data_paths = []
        input_tensors = []

        # Step 1: Load model and initialize tools (must succeed for tools to work)
        if model_url:
            try:
                if not os.path.isabs(model_url):
                    local_model_path = self.models_dir / model_url
                    if local_model_path.exists():
                        model_url = str(local_model_path)

                print(f"Loading model: {model_url}")
                loader_module = load_model_loader_module(model_url, str(self.models_dir))
                model, processor = loader_module.load_model(model_url)

                if hasattr(loader_module, 'get_model_info'):
                    extra_info = loader_module.get_model_info(model)
                else:
                    extra_info = {}

                model_info = {
                    "success": True,
                    "model": model,
                    "processor": processor,
                    "model_name": Path(model_url).name,
                    "model_type": "local_pth",
                    "model_path": model_url,
                    "device": str(loader_module.DEVICE if hasattr(loader_module, 'DEVICE') else 'cpu'),
                    "architecture": model.__class__.__name__,
                    "num_classes": extra_info.get('num_classes'),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "label_map": extra_info.get('label_map', {}),
                    "feature_names": extra_info.get('feature_names', [])
                }

                print(f"Model loaded: {model_info['architecture']}")

                # Initialize DataModelLoader
                self.data_model_loader = DataModelLoader(
                    model_name=Path(model_url).stem,
                    modality=modality
                )

                print("\n=== Initializing XAI Tools ===")
                self.actor.initialize_tools(data_model_loader=self.data_model_loader)
                self.critic.set_model(self.data_model_loader.get_model())

            except Exception as e:
                print(f"ERROR: Failed to load model and initialize tools: {e}")
                import traceback
                traceback.print_exc()
                # Without model+tools, pipeline cannot produce meaningful results
                raise RuntimeError(f"Model/tool initialization failed for multi-instance: {e}") from e

        # Step 2: Load ALL instances (errors here don't block tool initialization)
        try:
            if modality == 'vision':
                # Get image root path
                if image_root is None:
                    image_root = str(self.dataset_dir / "image")

                # Extract dataset name from question path
                dataset_name = self._extract_dataset_name_from_path(
                    question.get('dataset_base_name', '')
                )

                # Get image indices from unified format
                image_indices = question.get('image_indices', [])
                targets = question.get('targets', [])
                split = question.get('split', 'test')

                print(f"Loading {len(image_indices)} instances...")

                # Load each instance
                for i, img_idx in enumerate(image_indices):
                    print(f"  Loading Instance {i}: index={img_idx}")

                    # Build image path
                    img_path = self._build_image_path(image_root, dataset_name, img_idx)
                    data_paths.append(img_path)
                    print(f"    Path: {img_path}")

                    # Load sample
                    data = self.data_model_loader.load_sample(index=img_idx, split=split)
                    image = data.get('image')  # PIL Image
                    input_tensors.append(image)

                    # Make prediction
                    pred = self.data_model_loader.predict(image)
                    if pred and pred.get('success'):
                        pred['ground_truth_idx'] = data.get('label')
                        pred['ground_truth_name'] = data.get('label_name')
                        # Add target from question if available
                        if i < len(targets):
                            pred['target_from_question'] = targets[i]
                        print(f"    Prediction: {pred.get('predicted_class_name')} "
                              f"(confidence: {pred.get('confidence', 0.0):.4f})")

                    predictions.append(pred)

            elif modality == 'text':
                image_indices = question.get('image_indices', [])
                targets = question.get('targets', [])
                split = question.get('split', 'test')
                features_list = question.get('features', [])

                print(f"Loading {len(image_indices)} text instances...")

                for i, img_idx in enumerate(image_indices):
                    print(f"  Loading Instance {i}: index={img_idx}")

                    # Get text from features array in benchmark JSON
                    if isinstance(features_list, list) and i < len(features_list):
                        feat = features_list[i]
                    else:
                        feat = {}

                    # Determine if NLI or standard text
                    if 'premise' in feat and 'hypothesis' in feat:
                        text_input = feat  # NLI format
                    elif 'text' in feat:
                        text_input = feat.get('text', '')
                    elif 'review_text' in feat:
                        text_input = feat.get('review_text', '')
                    else:
                        text_input = feat

                    # Use loader module's load_data to tokenize
                    loader_module = self.data_model_loader.loader_module
                    data = loader_module.load_data(text_input)
                    self.data_model_loader.current_sample_data = data

                    data_paths.append(f"text_index_{img_idx}")

                    # Store text data for evaluation
                    if 'premise' in feat:
                        input_tensor_i = {'premise': feat.get('premise'), 'hypothesis': feat.get('hypothesis')}
                    else:
                        input_tensor_i = data.get('text', text_input)
                    input_tensors.append(input_tensor_i)

                    # Make prediction
                    model = self.data_model_loader.get_model()
                    processor = self.data_model_loader.get_processor()
                    if model is not None:
                        if 'premise' in feat:
                            pred = loader_module.predict(model=model, text_input=data, tokenizer=processor)
                        else:
                            pred = loader_module.predict(model=model, text_input=data, tokenizer=processor)
                        if pred and pred.get('success'):
                            if i < len(targets):
                                pred['target_from_question'] = targets[i]
                            print(f"    Prediction: {pred.get('predicted_class_name')} "
                                  f"(confidence: {pred.get('confidence', 0.0):.4f})")
                        predictions.append(pred)
                    else:
                        predictions.append({})

            elif modality == 'tabular':
                image_indices = question.get('image_indices', [])
                targets = question.get('targets', [])
                split = question.get('split', 'test')
                # Multi-instance tabular (Q9/Q10) uses full-dataset indices for some datasets
                q_type = question.get('q_type', 1)
                if q_type in (8, 9, 10):
                    split = 'full'

                print(f"Loading {len(image_indices)} tabular instances...")

                for i, row_idx in enumerate(image_indices):
                    print(f"  Loading Instance {i}: index={row_idx}")

                    data = self.data_model_loader.load_sample(index=row_idx, split=split)
                    data_paths.append(f"tabular_index_{row_idx}")
                    input_tensors.append(data.get('features'))

                    # Make prediction
                    loader_module = self.data_model_loader.loader_module
                    model = self.data_model_loader.get_model()
                    processor = self.data_model_loader.get_processor()
                    if model is not None:
                        pred = loader_module.predict(
                            model=model,
                            input_data=data.get('features'),
                            preprocessor=processor
                        )
                        if pred and pred.get('success'):
                            pred['ground_truth_idx'] = data.get('label')
                            pred['ground_truth_name'] = data.get('label_name')
                            if i < len(targets):
                                pred['target_from_question'] = targets[i]
                            print(f"    Prediction: {pred.get('predicted_class_name')} "
                                  f"(confidence: {pred.get('confidence', 0.0):.4f})")
                        predictions.append(pred)
                    else:
                        predictions.append({})

        except Exception as e:
            print(f"Warning: Failed to load instance data for multi-instance: {e}")
            import traceback
            traceback.print_exc()

        return model_info, predictions, data_paths, input_tensors

    def _save_complete_results(self, results: Dict[str, Any]):
        """Save complete pipeline results."""
        import re
        # Extract naming components from question
        question = results.get('question', {})
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', results.get('question_id', 0))
        modality = question.get('modality', 'vision')

        # Extract dataset_name and q_type from dataset_base_name
        # e.g., "stl10_resnet_q1_test" -> dataset_name="stl10_resnet", q_type="q1"
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)  # e.g., "stl10_resnet"
            q_type_str = match.group(2)    # e.g., "q1"
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        # Directory structure: /complete_results/{modality}/{dataset_name}/{q_type}/{question_id}/
        complete_results_dir = self.output_dir / "complete_results" / modality / dataset_name / q_type_str / str(row_no)
        complete_results_dir.mkdir(parents=True, exist_ok=True)

        results_file = complete_results_dir / "complete_results.json"

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

        # Add strategy faithfulness if present
        if "strategy_faithfulness" in results:
            clean_results["strategy_faithfulness"] = results["strategy_faithfulness"]

        # Add improvement results if present
        if "reflections" in results:
            clean_results["reflections"] = results["reflections"]
        if "improved" in results:
            clean_results["improved"] = results["improved"]
        if "improvement_metrics" in results:
            clean_results["improvement_metrics"] = results["improvement_metrics"]

        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)

        print(f"\nComplete results saved to: {results_file}")

    def _save_training_datapoint(
        self,
        question: Dict[str, Any],
        original_strategy: Dict[str, Any],
        original_results: Dict[str, Any],
        original_evaluation: Dict[str, Any],
        strategy_faithfulness: Any,
        proposer_reflection: str,
        actor_reflection: str,
        improved_strategy: Dict[str, Any],
        improved_results: Dict[str, Any],
        improved_evaluation: Dict[str, Any]
    ):
        """
        Save a training datapoint containing all improvement data.

        This datapoint can be used for training or fine-tuning agents.

        Args:
            question: Original question
            original_strategy: Original strategy from Proposer
            original_results: Original results from Actor
            original_evaluation: Original evaluation from Critic
            strategy_faithfulness: Strategy faithfulness evaluation result
            proposer_reflection: Critic's feedback for Proposer (JSON string)
            actor_reflection: Critic's feedback for Actor (JSON string)
            improved_strategy: Improved strategy after reflection
            improved_results: Improved results after reflection
            improved_evaluation: Evaluation of improved results
        """
        modality = question.get('modality', 'vision')
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))

        # Calculate improvement metrics
        original_faith = original_evaluation.get('faithfulness', {}).get('score', 0.0)
        improved_faith = improved_evaluation.get('faithfulness', {}).get('score', 0.0)

        if original_faith is None:
            original_faith = 0.0
        if improved_faith is None:
            improved_faith = 0.0

        # Determine tools changed
        original_tools = [t.get('tool_name') for t in original_strategy.get('selected_tools', [])]
        improved_tools = [t.get('tool_name') for t in improved_strategy.get('selected_tools', [])]

        tools_added = [t for t in improved_tools if t not in original_tools]
        tools_removed = [t for t in original_tools if t not in improved_tools]

        # Build training datapoint
        datapoint = {
            # Metadata
            "datapoint_id": f"{dataset_base_name}_{row_no}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "question_id": question.get('question_id', 'unknown'),
            "question": question,
            "modality": modality,
            "timestamp": datetime.now().isoformat(),

            # Original results
            "original": {
                "strategy": original_strategy,
                "explanation": {k: v for k, v in original_results.items()
                              if k not in ['raw_output', 'model', 'tool_results']},
                "explanation_faithfulness": original_evaluation.get('faithfulness', {})
            },

            # Strategy faithfulness
            "strategy_faithfulness": strategy_faithfulness.to_dict() if hasattr(strategy_faithfulness, 'to_dict') else strategy_faithfulness,

            # Reflections (raw JSON strings for agents to consume)
            "proposer_reflection": proposer_reflection,
            "actor_reflection": actor_reflection,

            # Improved results
            "improved": {
                "strategy": improved_strategy,
                "explanation": {k: v for k, v in improved_results.items()
                              if k not in ['raw_output', 'model', 'tool_results']},
                "explanation_faithfulness": improved_evaluation.get('faithfulness', {})
            },

            # Improvement metrics
            "improvement_metrics": {
                "original_faithfulness": original_faith,
                "improved_faithfulness": improved_faith,
                "faithfulness_delta": improved_faith - original_faith,
                "improvement_percentage": ((improved_faith - original_faith) / max(original_faith, 0.001)) * 100,
                "strategy_changed": original_tools != improved_tools,
                "tools_added": tools_added,
                "tools_removed": tools_removed,
                "original_tool_count": len(original_tools),
                "improved_tool_count": len(improved_tools)
            }
        }

        # Save to training_data directory with nested structure
        import re
        # Extract dataset_name and q_type from dataset_base_name
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        # Directory structure: /training_data/{modality}/{dataset_name}/{q_type}/{question_id}/
        training_dir = self.training_data_dir / modality / dataset_name / q_type_str / str(row_no)
        training_dir.mkdir(parents=True, exist_ok=True)

        filename = "training_datapoint.json"
        filepath = training_dir / filename

        with open(filepath, 'w') as f:
            json.dump(datapoint, f, indent=2, default=str)

        print(f"  Training datapoint saved to: {filepath}")

    def _save_training_datapoint_passed(
        self,
        question: Dict[str, Any],
        original_strategy: Dict[str, Any],
        original_results: Dict[str, Any],
        original_evaluation: Dict[str, Any]
    ):
        """
        Save a simplified training datapoint for samples that passed faithfulness.

        These samples don't go through strategy faithfulness or improvement,
        but are still valuable as positive examples for training.

        Args:
            question: Original question
            original_strategy: Strategy from Proposer
            original_results: Results from Actor
            original_evaluation: Evaluation from Critic
        """
        modality = question.get('modality', 'vision')
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))

        datapoint = {
            "datapoint_id": f"{dataset_base_name}_{row_no}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "question_id": question.get('question_id', 'unknown'),
            "question": question,
            "modality": modality,
            "timestamp": datetime.now().isoformat(),
            "status": "passed",
            "original": {
                "strategy": original_strategy,
                "explanation": {k: v for k, v in original_results.items()
                              if k not in ['raw_output', 'model', 'tool_results']},
                "explanation_faithfulness": original_evaluation.get('faithfulness', {})
            }
        }

        # Save to training_data directory with nested structure
        import re
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        training_dir = self.training_data_dir / modality / dataset_name / q_type_str / str(row_no)
        training_dir.mkdir(parents=True, exist_ok=True)

        filename = "training_datapoint.json"
        filepath = training_dir / filename

        with open(filepath, 'w') as f:
            json.dump(datapoint, f, indent=2, default=str)

        print(f"  Training datapoint (passed) saved to: {filepath}")


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
        help="VLM model ID. Local: 'Qwen/Qwen3-VL-8B-Instruct'. Tinker: 'tinker/Qwen/Qwen3-VL-30B-A3B-Instruct'. API: 'gemini-2.5-pro', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001'"
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
        help="[Deprecated] Path to image - loads directly from dataset"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip faithfulness evaluation"
    )
    parser.add_argument(
        "--faithfulness_threshold",
        type=float,
        default=0.1,
        help="Threshold for explanation faithfulness (default: 0.1)"
    )
    parser.add_argument(
        "--no-improvement",
        action="store_true",
        help="Skip improvement phase even if faithfulness is below threshold"
    )
    parser.add_argument(
        "--no-sf",
        action="store_true",
        help="Skip strategy faithfulness evaluation"
    )
    parser.add_argument(
        "--sf_max_samples",
        type=int,
        default=None,
        help="Max number of tool configs to sample for strategy faithfulness (default: None = full 2^N enumeration)"
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
        evaluate_faithfulness=not args.no_eval,
        faithfulness_threshold=args.faithfulness_threshold,
        enable_improvement=not args.no_improvement,
        enable_sf=not args.no_sf,
        sf_max_samples=args.sf_max_samples
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

    # Print strategy faithfulness if available
    if results.get('strategy_faithfulness'):
        sf = results['strategy_faithfulness']
        print(f"\nStrategy Faithfulness:")
        print(f"  Tool Importance Scores: {sf.get('tool_importance_scores', {})}")

    # Print improvement metrics if available
    if results.get('improvement_metrics'):
        im = results['improvement_metrics']
        print(f"\nImprovement Metrics:")
        print(f"  Original Faithfulness: {im.get('original_faithfulness', 'N/A'):.4f}")
        print(f"  Improved Faithfulness: {im.get('improved_faithfulness', 'N/A'):.4f}")
        print(f"  Delta: {im.get('delta', 'N/A'):+.4f}")

    explanation = results.get('results', {}).get('explanation', 'N/A')
    if isinstance(explanation, str) and len(explanation) > 100:
        explanation = explanation[:100] + "..."
    print(f"\nExplanation: {explanation}")


if __name__ == "__main__":
    main()

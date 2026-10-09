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
from MEA_agent_system import (
    create_three_agent_system,
    ThreeAgentPipeline,
    ProposerAgent,
    ActorAgent,
    CriticAgent,
)
from evaluation import get_evaluator, EvaluationResult, set_masking_output_dir

# Existing imports
from vlm_wrapper import VisionLanguageModel, create_vlm
from DataModelLoader import DataModelLoader
from prompts.output_size_config import OutputSizeConfig


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

    # Dynamically import the loader module.
    # Register under the canonical package name so that DataModelLoader's
    # importlib.import_module("models_to_read.{modality}.load_{name}") returns
    # the SAME object and therefore the SAME _cache.  Without this, the two
    # import paths produce separate module objects with independent _cache dicts,
    # causing get_feature_modes() to see an empty cache after load_model() has
    # already populated it via the spec-loaded copy.
    canonical_name = f"models_to_read.{model_path.parent.name}.{loader_path.stem}"
    import sys

    # Return the cached module only if exec completed successfully (indicated
    # by the presence of `load_model`); otherwise a partially initialised module
    # left in sys.modules by a failed exec_module would be returned.
    if canonical_name in sys.modules:
        cached = sys.modules[canonical_name]
        if hasattr(cached, 'load_model'):
            return cached
        # Stale incomplete entry — remove so we retry cleanly below.
        del sys.modules[canonical_name]

    spec = importlib.util.spec_from_file_location(canonical_name, loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = loader_module   # register BEFORE exec (circular import safety)
    try:
        spec.loader.exec_module(loader_module)
    except Exception:
        # Clean up so the next call retries from scratch instead of returning
        # the incomplete module object.
        sys.modules.pop(canonical_name, None)
        raise

    # Verify required functions exist
    required_funcs = ['load_model', 'load_data', 'predict']
    for func_name in required_funcs:
        if not hasattr(loader_module, func_name):
            raise AttributeError(f"Loader module {loader_path} missing required function: {func_name}")

    return loader_module


class MEAPipeline:
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
        models_dir: Optional[str] = None,
        mode: str = "test",
        tinker_checkpoint: Optional[str] = None,
        tinker_lora_rank: int = 16,
        vlm: Optional[Any] = None,
        output_size_config=None,
        temperature: float = 0.0,
        ablation_mode: Optional[str] = None,
    ):
        """
        Initialize XAI Pipeline V2.

        Args:
            vlm_model_id: VLM model ID (ignored when vlm is provided)
            output_dir: Output directory
            dataset_dir: Directory containing datasets
            models_dir: Directory containing models
            mode: Dataset split to use ('train' or 'test'); benchmark JSONs are
                  loaded from ``dataset_dir/{mode}/{modality}/``.
            tinker_checkpoint: Tinker checkpoint to load for evaluation, e.g.
                  'tinker/dpo_Qwen3-VL-30B-A3B-Instruct_1771865187--step-0500'.
                  Only applied when mode='test'. Requires vlm_model_id to be a
                  tinker/* model (used as the LoRA base model).
            tinker_lora_rank: LoRA rank used during DPO/LoRA training (default: 16).
            vlm: Optional pre-built VLM instance. When provided, vlm_model_id and
                 tinker_checkpoint are ignored. Useful for RL training where the VLM
                 is a custom RLSamplingVLM that records token trajectories.
            ablation_mode: One of None, 'tool_only', 'autonomous_only'. When set, the
                 Proposer's prompt hides the disallowed strategy branch and the
                 generated strategy is validated to only use the allowed branch --
                 see ProposerAgent._enforce_ablation_mode().
        """
        self.mode = mode
        self.output_size_config = output_size_config

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
        print(f"Mode: {self.mode}")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Models directory: {self.models_dir}")
        print(f"Output directory: {self.output_dir}")

        if vlm is not None:
            # Use the provided VLM directly (e.g. RLSamplingVLM for GRPO training)
            print("\nUsing provided VLM instance (skipping create_vlm).")
            self.vlm = vlm
        else:
            # If a tinker_checkpoint is provided but the caller did not explicitly select a
            # tinker/* base model (i.e. the default local model is still set), auto-derive
            # the base model from the checkpoint name so we go through the Tinker path
            # instead of trying to load an 8B / 30B model locally.
            if tinker_checkpoint is not None and not vlm_model_id.startswith("tinker/"):
                derived = self._derive_tinker_base_model(tinker_checkpoint)
                if derived:
                    print(f"\nAuto-deriving VLM base model from tinker_checkpoint: tinker/{derived}")
                    vlm_model_id = f"tinker/{derived}"

            # Initialize VLM
            print("\nInitializing VLM...")
            self.vlm = create_vlm(model_id=vlm_model_id, temperature=temperature)

        # Initialize three agents using new modular system
        print("\nInitializing Agents (New Architecture)...")
        self.proposer, self.actor, self.critic = create_three_agent_system(
            vlm=self.vlm,
            model=None,  # Will be set after loading target model
            output_dir=str(self.output_dir),
            models_dir=str(self.models_dir),
            output_size_config=output_size_config,
            ablation_mode=ablation_mode
        )

        # Load Tinker LoRA/DPO checkpoint for evaluation (test mode only)
        if tinker_checkpoint is not None:
            if mode == "test":
                self._load_tinker_checkpoint(tinker_checkpoint, tinker_lora_rank)
            else:
                print(f"\nWarning: tinker_checkpoint is ignored in mode='{mode}' "
                      f"(only applied when mode='test').")

        self.training_data_dir = self.output_dir / "training_data"
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        print("\nXAI Pipeline V2 initialized successfully!")

    # =========================================================================
    # Tinker Checkpoint Loading (test-mode evaluation of fine-tuned models)
    # =========================================================================

    @staticmethod
    def _derive_tinker_base_model(checkpoint: str) -> Optional[str]:
        """Extract the base model name from a tinker checkpoint path.

        Expected format: 'tinker/{algo}_{ModelName}_{timestamp}--{step}'
        e.g. 'tinker/dpo_Qwen3-VL-30B-A3B-Instruct_1771865187--step-0500'
             -> 'Qwen3-VL-30B-A3B-Instruct'

        Returns the ModelName portion, or None if parsing fails.
        """
        name = checkpoint.removeprefix("tinker/")
        run_id = name.split("--")[0]          # strip '--step-...' suffix
        parts = run_id.split("_")             # ['dpo', 'Qwen3', 'VL', '30B', ..., '1771865187']
        if len(parts) < 3:
            return None
        # Drop the algorithm prefix (first part) and the numeric timestamp (last part)
        inner = parts[1:-1] if parts[-1].isdigit() else parts[1:]
        if not inner:
            return None
        return "-".join(inner)                # 'Qwen3-VL-30B-A3B-Instruct'

    def _load_tinker_checkpoint(self, checkpoint: str, lora_rank: int = 16) -> None:
        """
        Load a fine-tuned LoRA/DPO checkpoint from Tinker and replace the VLM's
        sampling client.  Only called when mode='test'.

        The checkpoint path format is 'tinker/<run_id>--<step>', e.g.:
            'tinker/dpo_Qwen3-VL-30B-A3B-Instruct_1771865187--step-0500'

        This is converted to 'tinker://<run_id>/<step>' for load_state(), then
        save_weights_for_sampler() prepares it for inference.

        Args:
            checkpoint:  Tinker checkpoint path as passed via --tinker_checkpoint.
            lora_rank:   LoRA rank used during training (must match the original job).
        """
        try:
            import tinker
        except ImportError:
            raise ImportError(
                "tinker package not available. Install with: pip install tinker"
            )

        from vlm_wrapper import TinkerVisionLanguageModel
        if not isinstance(self.vlm, TinkerVisionLanguageModel):
            raise RuntimeError(
                f"tinker_checkpoint requires a Tinker-backed VLM (--vlm tinker/...), "
                f"but current VLM is {type(self.vlm).__name__}."
            )

        # Accept two formats:
        #   1. Native tinker path:  tinker://run_id/step  (from training logs)
        #   2. Legacy dash format:  tinker/run_id--step   (old convention)
        if checkpoint.startswith("tinker://"):
            # Already a valid tinker URL — use as-is
            tinker_path = checkpoint
        elif "--" in checkpoint:
            checkpoint_name = checkpoint.removeprefix("tinker/")
            run_id, step = checkpoint_name.split("--", 1)
            tinker_path = f"tinker://{run_id}/{step}"
        else:
            raise ValueError(
                f"Cannot parse tinker_checkpoint format: {repr(checkpoint)}\n"
                f"Expected either 'tinker://<run_id>/<step>' or 'tinker/<run_id>--<step>'."
            )

        base_model = self.vlm.model_id  # e.g. "Qwen/Qwen3-VL-30B-A3B-Instruct"

        print(f"\nLoading Tinker LoRA/DPO checkpoint for evaluation...")
        print(f"  Checkpoint tinker path : {tinker_path}")
        print(f"  Base model             : {base_model}")
        print(f"  LoRA rank              : {lora_rank}")

        import time
        max_retries = 5
        base_delay = 10  # seconds; doubles each attempt

        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    delay = base_delay * (2 ** (attempt - 2))  # 10, 20, 40, 80 s
                    print(f"  Retry {attempt}/{max_retries} after {delay}s ...")
                    time.sleep(delay)

                service_client = tinker.ServiceClient()
                training_client = service_client.create_lora_training_client(
                    base_model=base_model,
                    rank=lora_rank,
                )
                training_client.load_state(tinker_path)

                sampling_path = training_client.save_weights_for_sampler(name="eval").result().path
                print(f"  Sampler weights path   : {sampling_path}")

                # Replace the sampling client in-place so the existing VLM object is reused
                self.vlm.sampling_client = service_client.create_sampling_client(
                    model_path=sampling_path
                )
                break  # success

            except Exception as exc:
                last_exc = exc
                print(f"  Attempt {attempt}/{max_retries} failed: {exc}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to load Tinker checkpoint after {max_retries} attempts: {exc}"
                    ) from exc

        # Propagate updated VLM to all three agents
        self.proposer.vlm = self.vlm
        self.actor.vlm = self.vlm
        self.critic.vlm = self.vlm

        print("Tinker checkpoint loaded successfully.")

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

        # Set modality first so index extraction can use it
        modality_map = {'image': 'vision', 'text': 'text', 'tabular': 'tabular', 'vision': 'vision'}
        _raw_mod = question.get('modality')  # may be None even if key exists
        if _raw_mod:
            question['modality'] = modality_map.get(_raw_mod, 'vision')
        else:
            # Infer from dataset path when field is absent or explicitly None
            _pl = dataset_path.lower()
            if '/tabular/' in _pl or '_tabular' in _pl:
                question['modality'] = 'tabular'
            elif '/text/' in _pl or '_text' in _pl:
                question['modality'] = 'text'
            else:
                question['modality'] = 'vision'

        # Extract instance indices: prefer features.image_index, fall back to features.row_no, then inst.row_no
        def _get_image_index(inst):
            feats = inst.get('features', {})
            idx = feats.get('image_index')
            if idx is None:
                idx = feats.get('row_no')
            if idx is None:
                idx = inst.get('row_no')
            return idx

        if question['modality'] == 'vision':
            image_index_a = _get_image_index(instance_a)
            image_index_b = _get_image_index(instance_b)
        else:
            image_index_a = instance_a.get('row_no', instance_a.get('features', {}).get('image_index'))
            image_index_b = instance_b.get('row_no', instance_b.get('features', {}).get('image_index'))
        question['image_indices'] = [image_index_a, image_index_b]

        # Use pair_id as row_no for file naming
        question['row_no'] = question.get('pair_id', question_id)

        # Normalize question text
        question['question'] = question.get('example', question.get('q', ''))

        question['question_id'] = f"q4_{question.get('pair_id', question_id)}"

        print(f"Q4 Question loaded:")
        _pred_key = 'name' if 'name' in instance_a['prediction'] else 'label'
        print(f"  Instance A: index={image_index_a}, pred={instance_a['prediction'].get(_pred_key)}")
        print(f"  Instance B: index={image_index_b}, pred={instance_b['prediction'].get(_pred_key)}")

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

                # Release old model from CUDA before creating new DataModelLoader
                if hasattr(self, 'data_model_loader') and self.data_model_loader is not None:
                    del self.data_model_loader
                    self.data_model_loader = None
                    torch.cuda.empty_cache()

                # Single model load via DataModelLoader
                self.data_model_loader = DataModelLoader(
                    model_name=Path(model_url).stem,
                    modality=modality,
                    data_path=str(self.dataset_dir),
                    model_path=model_url,
                )
                model = self.data_model_loader.get_model()
                processor = self.data_model_loader.get_processor()
                loader_module = self.data_model_loader.loader_module

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
                    "feature_names": extra_info.get('feature_names') or []
                }

                # Tabular Q4: feature_names from instance_A's features dict
                if modality == 'tabular':
                    inst_a_feats = question.get('instance_A', {}).get('features', {})
                    if inst_a_feats:
                        model_info['feature_names'] = list(inst_a_feats.keys())

                print("\n=== Initializing XAI Tools ===")
                self.actor.initialize_tools(data_model_loader=self.data_model_loader)
                self.proposer.set_tool_registry(self.actor.tool_registry)
                self.critic.set_model(model)

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

                def _get_vis_index(inst):
                    feats = inst.get('features', {})
                    idx = feats.get('image_index')
                    if idx is None:
                        idx = feats.get('row_no')
                    if idx is None:
                        idx = inst.get('row_no')
                    return idx

                def _get_vis_path(inst, idx):
                    """Use features.image_path if available, else build from index."""
                    rel = inst.get('features', {}).get('image_path')
                    if rel:
                        return image_root + rel
                    return self._build_image_path(image_root, dataset_name, idx)

                temp_img_dir = self.output_dir / "temp_images"
                temp_img_dir.mkdir(parents=True, exist_ok=True)
                _q4_rid = question.get('rollout_id', 0)

                def _resolve_img_path(pil_img, path, idx, label):
                    """Return path; save to temp file if path doesn't exist on disk."""
                    if path and os.path.exists(path):
                        return path
                    temp_path = str(temp_img_dir / f"q4_{label}_{idx}_{split}_r{_q4_rid}.png")
                    pil_img.save(temp_path)
                    return temp_path

                # Load Instance A
                img_idx_a = _get_vis_index(instance_a)
                print(f"\n  Loading Instance A: image_index={img_idx_a}")

                data_a = self.data_model_loader.load_sample(index=img_idx_a, split=split)
                input_tensors['A'] = data_a.get('image')
                img_path_a = _resolve_img_path(input_tensors['A'],
                                               _get_vis_path(instance_a, img_idx_a),
                                               img_idx_a, 'A')
                data_paths['A'] = img_path_a
                print(f"    Path: {img_path_a}")

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
                img_idx_b = _get_vis_index(instance_b)
                print(f"\n  Loading Instance B: image_index={img_idx_b}")

                data_b = self.data_model_loader.load_sample(index=img_idx_b, split=split)
                input_tensors['B'] = data_b.get('image')
                img_path_b = _resolve_img_path(input_tensors['B'],
                                               _get_vis_path(instance_b, img_idx_b),
                                               img_idx_b, 'B')
                data_paths['B'] = img_path_b
                print(f"    Path: {img_path_b}")

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
                    if model_info and data.get('feature_names'):
                        model_info['feature_names'] = data['feature_names']
                        question['feature_names'] = data['feature_names']

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
            raise RuntimeError(f"Failed to load model/data for Q4: {e}") from e

        return model_info, predictions, data_paths, input_tensors

    def run_q4(
        self,
        question_dataset_path: str,
        question_id: str,
        target_model_url: Optional[str] = None,
        image_root: Optional[str] = None,
        evaluate_faithfulness: bool = True,
        enable_improvement: bool = True,
        rollout_id: Optional[int] = None,
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
        if rollout_id is not None:
            question['rollout_id'] = rollout_id

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
                feats = inst.get('features', {})
                return feats.get('image_index', feats.get('row_no'))
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
                device=model_info.get('device', 'cuda'),
                original_features=question.get('features', {}),
                feature_modes=self.data_model_loader.get_feature_modes() if self.data_model_loader else None
            )
        else:
            print("  Skipping faithfulness evaluation")
            evaluation = {"status": "skipped"}

        # Step 7: Check faithfulness and run improvement loop (mirrors run_multi_instance logic)
        sf_result = None
        if evaluate_faithfulness and model_info and model_info.get('model') and evaluation:
            faithfulness_result = evaluation.get('faithfulness', {})
            faithfulness_score = faithfulness_result.get('score')
            if faithfulness_score is None:
                faithfulness_score = 0.0
            faithfulness_passed = faithfulness_result.get('passed', faithfulness_score >= 0.1)
            evaluator_threshold = faithfulness_result.get('details', {}).get('threshold', 0.1)

            if faithfulness_passed:
                print("\n=== Step 7: Explanation Faithfulness Check (Q4) ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) passed (threshold: {evaluator_threshold})")
                print("  Saving training datapoint for passed sample.")
                self._save_training_datapoint_passed(
                    question=question,
                    original_strategy=strategy,
                    original_results=results,
                    original_evaluation=evaluation
                )
            else:
                print(f"\n=== Step 7: Faithfulness Check ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) failed (threshold: {evaluator_threshold})")

                # Step 8: Improvement Phase
                if enable_improvement:
                    print("\n=== Step 8: Improvement Phase (Q4) ===")

                    print("  Generating Critic reflections...")
                    proposer_reflection, actor_reflection = self.critic.generate_reflections(
                        strategy=strategy,
                        results=results,
                        question=question,
                        faithfulness_result=evaluation.get('faithfulness', {}),
                        tool_importance_scores={},
                        threshold=evaluator_threshold
                    )
                    self.critic.save_reflections(proposer_reflection, actor_reflection, question)

                    # Proposer generates improved Q4 strategy (reuses run_q4 with reflection params)
                    print("\n  Proposer generating improved Q4 strategy...")
                    improved_strategy = self.proposer.run_q4(
                        question=question,
                        question_template=template,
                        model_info=model_info,
                        instances=instances,
                        proposer_reflection=proposer_reflection,
                        original_strategy=strategy
                    )

                    # Actor re-executes Q4 with improved strategy (same method, better strategy)
                    print("\n  Actor re-executing Q4 with improved strategy...")
                    improved_results = self.actor.run_q4(
                        strategy=improved_strategy,
                        question=question,
                        question_template=template,
                        instances=instances,
                        model_info=model_info
                    )

                    # Critic evaluates improved Q4 explanation
                    print("\n  Evaluating improved Q4 explanation...")
                    improved_evaluation = self.critic.run_q4(
                        results=improved_results,
                        question=question,
                        inputs={'A': input_tensors['A'], 'B': input_tensors['B']},
                        predictions={'A': predictions['A'], 'B': predictions['B']},
                        processor=model_info.get('processor'),
                        device=model_info.get('device', 'cuda'),
                        suffix="_improved",
                        original_features=question.get('features', {}),
                        feature_modes=self.data_model_loader.get_feature_modes() if self.data_model_loader else None
                    )

                    improved_faithfulness = improved_evaluation.get('faithfulness', {}).get('score', 0.0)
                    if improved_faithfulness is None:
                        improved_faithfulness = 0.0
                    improvement_delta = improved_faithfulness - faithfulness_score

                    self._save_training_datapoint(
                        question=question,
                        original_strategy=strategy,
                        original_results=results,
                        original_evaluation=evaluation,
                        proposer_reflection=proposer_reflection,
                        actor_reflection=actor_reflection,
                        improved_strategy=improved_strategy,
                        improved_results=improved_results,
                        improved_evaluation=improved_evaluation
                    )

                    print(f"\n  Q4 Improvement complete:")
                    print(f"    Original faithfulness: {faithfulness_score:.4f}")
                    print(f"    Improved faithfulness: {improved_faithfulness:.4f}")
                    print(f"    Delta: {improvement_delta:+.4f}")

                    # Update evaluation and results to the improved version for complete_results
                    evaluation = improved_evaluation
                    results = improved_results
                    strategy = improved_strategy
                else:
                    print("\n  Improvement disabled. Skipping improvement phase.")

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
        enable_improvement: bool = True,
        rollout_id: Optional[int] = None,
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

        if rollout_id is not None:
            question['rollout_id'] = rollout_id

        # Step 2: Load model and data for all instances
        # Serialize explainee model load + predict across concurrent rollout
        # threads (see GPU_TOOL_LOCK in xai_tools.py).
        from xai_tools import GPU_TOOL_LOCK
        with GPU_TOOL_LOCK:
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

            _pred0 = predictions_list[0] if predictions_list else None
            _gt0 = question.get('ground_truth')
            if _gt0 is None and _pred0:
                _gt0 = _pred0.get('ground_truth_idx')
            if _gt0 is None:
                _target0 = question.get('target')
                if isinstance(_target0, list):
                    _target0 = _target0[0] if _target0 else {}
                if isinstance(_target0, dict):
                    _gt0 = _target0.get('value')
            evaluation = self.critic.run(
                results=results,
                question=question,
                original_input=input_tensors[0] if input_tensors else None,
                original_prediction=_pred0,
                ground_truth=_gt0,
                processor=model_info.get('processor'),
                device=model_info.get('device', 'cuda'),
                class_names=model_info.get('label_map', {}),
                feature_names=model_info.get('feature_names', []),
                inputs=input_tensors,
                predictions=predictions_list,
                ground_truths=ground_truths,
                original_features=question.get('features', {}),
                feature_modes=self.data_model_loader.get_feature_modes() if self.data_model_loader else None
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

            faithfulness_passed = faithfulness_result.get('passed', faithfulness_score >= 0.1)
            evaluator_threshold = faithfulness_result.get('details', {}).get('threshold', 0.1)

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
                print(f"\n=== Step 7: Faithfulness Check ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) failed (threshold: {evaluator_threshold})")

                # Step 8: Improvement Phase
                if enable_improvement:
                    print("\n=== Step 8: Improvement Phase ===")

                    print("  Generating Critic reflections...")
                    proposer_reflection, actor_reflection = self.critic.generate_reflections(
                        strategy=strategy,
                        results=results,
                        question=question,
                        faithfulness_result=evaluation.get('faithfulness', {}),
                        tool_importance_scores={},
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
                        original_prediction=_pred0,
                        ground_truth=_gt0,
                        processor=model_info.get('processor'),
                        device=model_info.get('device', 'cuda'),
                        class_names=model_info.get('label_map', {}),
                        feature_names=model_info.get('feature_names', []),
                        inputs=input_tensors,
                        predictions=predictions_list,
                        ground_truths=ground_truths,
                        suffix="_improved",
                        original_features=question.get('features', {}),
                        feature_modes=self.data_model_loader.get_feature_modes() if self.data_model_loader else None
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

                    self._save_training_datapoint(
                        question=question,
                        original_strategy=strategy,
                        original_results=results,
                        original_evaluation=evaluation,
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
        enable_improvement: bool = True,
        rollout_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run complete XAI pipeline for a single question.

        Args:
            question_dataset_path: Path to question dataset JSON
            question_id: ID of question to process
            target_model_url: URL/path of target model
            image_path: Path to image for vision modality
            evaluate_faithfulness: Whether to run faithfulness evaluation
            enable_improvement: Whether to run improvement when below threshold

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
        if rollout_id is not None:
            question['rollout_id'] = rollout_id

        # Auto-detect Q4 format and route to run_q4
        if question.get('is_q4_format') or question.get('q_type') == 4:
            print("  Detected Q4 format, routing to run_q4...")
            return self.run_q4(
                question_dataset_path=question_dataset_path,
                question_id=question_id,
                target_model_url=target_model_url,
                evaluate_faithfulness=evaluate_faithfulness,
                enable_improvement=enable_improvement,
                rollout_id=rollout_id,
            )

        # Auto-detect Q9/Q10 multi-instance format and route to run_multi_instance
        if question.get('is_multi_instance') and question.get('q_type') in [9, 10]:
            print(f"  Detected Q{question['q_type']} multi-instance format, routing to run_multi_instance...")
            return self.run_multi_instance(
                question=question,
                template=template,
                target_model_url=target_model_url,
                evaluate_faithfulness=evaluate_faithfulness,
                enable_improvement=enable_improvement,
                rollout_id=rollout_id,
            )

        # Step 2: Load target model and data
        # GPU_TOOL_LOCK: see the identical note above _load_model_and_data_multi's
        # call site in this file.
        from xai_tools import GPU_TOOL_LOCK
        with GPU_TOOL_LOCK:
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
            # Derive ground_truth: benchmark uses "target" dict, not "ground_truth" key.
            # Prefer integer ground_truth_idx (handles class 0 correctly) set during data loading.
            _gt = question.get('ground_truth')
            if _gt is None and prediction:
                _gt = prediction.get('ground_truth_idx')  # integer class index (may be 0)
            if _gt is None:
                _target = question.get('target', {})
                if isinstance(_target, dict):
                    _gt = _target.get('value')  # integer value from benchmark JSON
            evaluation = self.critic.run(
                results=results,
                question=question,
                original_input=input_tensor,
                original_prediction=prediction,
                ground_truth=_gt,
                processor=model_info.get('processor'),
                device=model_info.get('device', 'cuda'),
                class_names=model_info.get('label_map', {}),
                feature_names=model_info.get('feature_names', []),
                original_features=question.get('features', {}),
                feature_modes=self.data_model_loader.get_feature_modes() if self.data_model_loader else None
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
        if evaluate_faithfulness and model_info and model_info.get('model'):
            faithfulness_result = evaluation.get('faithfulness', {})
            faithfulness_score = faithfulness_result.get('score')
            if faithfulness_score is None:
                faithfulness_score = 0.0

            faithfulness_passed = faithfulness_result.get('passed', faithfulness_score >= 0.1)
            evaluator_threshold = faithfulness_result.get('details', {}).get('threshold', 0.1)

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
                print("\n=== Step 7: Faithfulness Check ===")
                print(f"  Explanation faithfulness ({faithfulness_score:.4f}) failed (threshold: {evaluator_threshold})")

                # Step 8: Improvement Phase (only if enabled and faithfulness below threshold)
                if enable_improvement:
                    print("\n=== Step 8: Improvement Phase ===")

                    print("  Generating Critic reflections...")
                    proposer_reflection, actor_reflection = self.critic.generate_reflections(
                        strategy=strategy,
                        results=results,
                        question=question,
                        faithfulness_result=evaluation.get('faithfulness', {}),
                        tool_importance_scores={},
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
                        ground_truth=_gt,
                        processor=model_info.get('processor'),
                        device=model_info.get('device', 'cuda'),
                        class_names=model_info.get('label_map', {}),
                        feature_names=model_info.get('feature_names', []),
                        suffix="_improved",
                        original_features=question.get('features', {}),
                        feature_modes=self.data_model_loader.get_feature_modes() if self.data_model_loader else None
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

                    self._save_training_datapoint(
                        question=question,
                        original_strategy=strategy,
                        original_results=results,
                        original_evaluation=evaluation,
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
        # Infer modality from question dict, then fall back to dataset path
        modality = question.get('modality')
        if not modality:
            path_lower = dataset_path.lower()
            if '/tabular/' in path_lower or 'tabular' in path_lower:
                modality = 'tabular'
            elif '/text/' in path_lower or 'text' in path_lower:
                modality = 'text'
            else:
                modality = 'vision'
            print(f"  Inferred modality from dataset path: {modality}")
        # Normalize modality aliases (e.g. 'image' -> 'vision')
        modality_map = {'image': 'vision', 'text': 'text', 'tabular': 'tabular', 'vision': 'vision'}
        modality = modality_map.get(modality, modality)

        question['question_id'] = f"q{question_id}"

        print(f"Question: {question.get('question', '')}")

        # Get question type and modality
        q_type = question.get('q_type')
        # Fall back to extracting q_type from dataset filename (e.g. yelp_bert_q1.json -> 1)
        if q_type is None:
            import re as _re
            _m = _re.search(r'_q(\d+)(?:_|\.|$)', os.path.basename(dataset_path), _re.IGNORECASE)
            if _m:
                q_type = int(_m.group(1))
                question['q_type'] = q_type
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

                # Release old model from CUDA before creating new DataModelLoader
                if hasattr(self, 'data_model_loader') and self.data_model_loader is not None:
                    del self.data_model_loader
                    self.data_model_loader = None
                    torch.cuda.empty_cache()

                # Single model load via DataModelLoader; reuse its loader_module for load_data/predict
                self.data_model_loader = DataModelLoader(
                    model_name=Path(model_url).stem,
                    modality=modality,
                    data_path=str(self.dataset_dir),
                    model_path=model_url,
                )
                model = self.data_model_loader.get_model()
                processor = self.data_model_loader.get_processor()
                loader_module = self.data_model_loader.loader_module

                # Get model info from loader module
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
                    "feature_names": extra_info.get('feature_names') or []
                }

                # Tabular: override feature_names from question's features dict (always available)
                if modality == 'tabular':
                    feats = question.get('features', {})
                    if isinstance(feats, dict) and feats:
                        model_info['feature_names'] = list(feats.keys())
                    elif isinstance(feats, list) and feats and isinstance(feats[0], dict):
                        model_info['feature_names'] = list(feats[0].keys())

                print(f"Model loaded: {model_info['architecture']}")

                print("\n=== Initializing XAI Tools ===")
                self.actor.initialize_tools(data_model_loader=self.data_model_loader)
                self.proposer.set_tool_registry(self.actor.tool_registry)
                # Set model for critic
                self.critic.set_model(model)

            # Load data based on modality using loader module
            if modality == 'vision' and loader_module:
                # Check if image_path is provided directly
                if image_path and os.path.exists(image_path):
                    print(f"Loading vision data from image_path: {image_path}")
                    from PIL import Image
                    raw_image = Image.open(image_path).convert('RGB')
                    processed_image = self.data_model_loader._compute_processed_image(raw_image)

                    temp_img_dir = self.output_dir / "temp_images"
                    temp_img_dir.mkdir(parents=True, exist_ok=True)
                    _rid = question.get('rollout_id', 0)
                    temp_img_path = temp_img_dir / f"from_path_{Path(image_path).stem}_r{_rid}.png"
                    processed_image.save(str(temp_img_path))
                    input_tensor = processed_image
                    loaded_data_path = str(temp_img_path)
                    print(f"Data loaded: {loaded_data_path}")

                    # Store in data_model_loader for consistency
                    self.data_model_loader.current_sample_data = {
                        'image': raw_image,
                        'processed_image': processed_image,
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
                    input_tensor = data['processed_image']  # PIL after Resize+CenterCrop

                    # Save processed image to a real file so actor_agent can pass it to VLM.
                    # Include rollout_id in the filename to avoid races when K parallel rollouts
                    # process the same sample concurrently (same sample_index + split).
                    temp_img_dir = self.output_dir / "temp_images"
                    temp_img_dir.mkdir(parents=True, exist_ok=True)
                    _rid = question.get('rollout_id', 0)
                    temp_img_path = temp_img_dir / f"sample_{sample_index}_{split}_r{_rid}.png"
                    input_tensor.save(str(temp_img_path))
                    loaded_data_path = str(temp_img_path)
                    print(f"Data loaded: index={sample_index} -> {loaded_data_path}")
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
                if self.data_model_loader:
                    print("Making prediction...")
                    prediction = loader_module.predict(
                        model=self.data_model_loader.get_model(),
                        text_input=data,
                        tokenizer=self.data_model_loader.get_processor()
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
                # Authoritative feature_names come from the loaded datapoint, not model_info cache
                if model_info and data.get('feature_names'):
                    model_info['feature_names'] = data['feature_names']
                    question['feature_names'] = data['feature_names']
                print(f"Tabular data loaded: {len(data.get('feature_names', []))} features")
                print(f"  Ground truth: {data.get('label_name')} (class {data.get('label')})")

                # Make prediction
                if self.data_model_loader:
                    print("Making prediction...")
                    prediction = loader_module.predict(
                        model=self.data_model_loader.get_model(),
                        input_data=input_tensor,
                        preprocessor=self.data_model_loader.get_processor()
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
            raise RuntimeError(f"Failed to load model/data: {e}") from e

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

                # Release old model from CUDA before creating new DataModelLoader
                if hasattr(self, 'data_model_loader') and self.data_model_loader is not None:
                    del self.data_model_loader
                    self.data_model_loader = None
                    torch.cuda.empty_cache()

                # Single model load via DataModelLoader; reuse its loader_module for predict
                self.data_model_loader = DataModelLoader(
                    model_name=Path(model_url).stem,
                    modality=modality,
                    data_path=str(self.dataset_dir),
                    model_path=model_url,
                )
                model = self.data_model_loader.get_model()
                processor = self.data_model_loader.get_processor()
                loader_module = self.data_model_loader.loader_module

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
                    "feature_names": extra_info.get('feature_names') or []
                }

                # Tabular: override feature_names from question's features list (always available)
                if modality == 'tabular':
                    feats = question.get('features', [])
                    if isinstance(feats, list) and feats and isinstance(feats[0], dict):
                        model_info['feature_names'] = list(feats[0].keys())
                    elif isinstance(feats, dict) and feats:
                        model_info['feature_names'] = list(feats.keys())

                print(f"Model loaded: {model_info['architecture']}")

                print("\n=== Initializing XAI Tools ===")
                self.actor.initialize_tools(data_model_loader=self.data_model_loader)
                self.proposer.set_tool_registry(self.actor.tool_registry)
                self.critic.set_model(model)

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

                temp_img_dir = self.output_dir / "temp_images"
                temp_img_dir.mkdir(parents=True, exist_ok=True)

                # Load each instance
                for i, img_idx in enumerate(image_indices):
                    print(f"  Loading Instance {i}: index={img_idx}")

                    # Load sample and use the processed image (Resize+CenterCrop applied)
                    data = self.data_model_loader.load_sample(index=img_idx, split=split)
                    image = data['processed_image']  # PIL after Resize+CenterCrop
                    input_tensors.append(image)

                    # Always save processed image to temp file.
                    # Include rollout_id to avoid races with parallel rollouts.
                    _rid = question.get('rollout_id', 0)
                    img_path = str(temp_img_dir / f"multi_{img_idx}_{split}_r{_rid}.png")
                    image.save(img_path)
                    data_paths.append(img_path)
                    print(f"    Path: {img_path}")

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
            raise RuntimeError(f"Failed to load instance data for multi-instance: {e}") from e

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
        proposer_reflection: str,
        actor_reflection: str,
        improved_strategy: Dict[str, Any],
        improved_results: Dict[str, Any],
        improved_evaluation: Dict[str, Any]
    ):
        """Save a training datapoint containing full improvement trajectory."""
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
        default=None,
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
        "--no-improvement",
        action="store_true",
        help="Skip improvement phase even if faithfulness is below threshold"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset split to use: 'train' loads from dataset/train/, 'test' loads from dataset/test/ (default: test)"
    )
    parser.add_argument(
        "--tinker_checkpoint",
        type=str,
        default=None,
        help=(
            "Tinker LoRA/DPO checkpoint to evaluate (mode=test only). "
            "Format: 'tinker/<run_id>--<step>', e.g. "
            "'tinker/dpo_Qwen3-VL-30B-A3B-Instruct_1771865187--step-0500'. "
            "Requires --vlm to be a tinker/* base model."
        )
    )
    parser.add_argument(
        "--tinker_lora_rank",
        type=int,
        default=32,
        help="LoRA rank used during DPO/LoRA training (must match the training job, default: 16)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the VLM (default: 0.0)"
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        choices=["tool_only", "autonomous_only"],
        default=None,
        help=(
            "Restrict the Proposer's strategy space for an ablation run. "
            "'tool_only' hides autonomous reasoning and requires selecting an XAI tool; "
            "'autonomous_only' hides external XAI tools and requires an autonomous task. "
            "Default: unrestricted (both allowed)."
        )
    )

    args = parser.parse_args()

    # Auto-derive --vlm from --tinker_checkpoint when not explicitly provided
    if args.vlm is None:
        if args.tinker_checkpoint is not None:
            import re
            checkpoint_name = args.tinker_checkpoint.removeprefix("tinker/")
            run_id = checkpoint_name.split("--")[0]
            # Strip trailing numeric job ID (e.g. _1771865187)
            model_name = re.sub(r'_\d+$', '', run_id)
            # Strip leading method prefix (e.g. dpo_, lora_)
            model_name = re.sub(r'^[a-z]+_', '', model_name)
            args.vlm = f"tinker/{model_name}"
            print(f"Auto-derived --vlm from tinker checkpoint: {args.vlm}")
        else:
            raise ValueError(
                "Must provide --vlm <model_id> or --tinker_checkpoint <path>."
            )

    # Create pipeline
    pipeline = MEAPipeline(
        vlm_model_id=args.vlm,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        mode=args.mode,
        tinker_checkpoint=args.tinker_checkpoint,
        tinker_lora_rank=args.tinker_lora_rank,
        temperature=args.temperature,
        ablation_mode=args.ablation_mode,
        output_size_config=OutputSizeConfig(
            fixed_percentage=0.25,
            apply_to_tabular=True,
            apply_to_text=True,
            apply_to_vision=False,
        ),
    )

    # Run pipeline

    results = pipeline.run(
        question_dataset_path=args.dataset,
        question_id=args.question_id,
        target_model_url=args.model_url,
        image_path=args.image_path,
        evaluate_faithfulness=not args.no_eval,
        enable_improvement=not args.no_improvement,
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

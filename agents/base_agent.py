"""
Base Agent class for XAI Agent Framework

Provides common functionality shared across all agents.
"""

from __future__ import annotations
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the XAI framework.

    Provides common functionality:
    - VLM integration
    - Output directory management
    - JSON parsing utilities
    - Error handling
    """

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None,
        agent_name: str = "BaseAgent"
    ):
        """
        Initialize base agent.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory path
            agent_name: Name for logging
        """
        self.vlm = vlm
        self.agent_name = agent_name

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"{self.agent_name} initialized")
        print(f"  Output directory: {self.output_dir}")

    def invoke_vlm(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """
        Invoke the VLM with a prompt and optional images.

        Args:
            prompt: Text prompt
            images: Optional list of image paths

        Returns:
            VLM response string
        """
        try:
            if images:
                if hasattr(self.vlm, 'invoke_with_images'):
                    return self.vlm.invoke_with_images(prompt, images)
                elif hasattr(self.vlm, 'invoke_multimodal'):
                    return self.vlm.invoke_multimodal(prompt, images)
                else:
                    raise RuntimeError(f"VLM does not support image input but {len(images)} images were provided")
            else:
                return self.vlm.invoke(prompt)
        except Exception as e:
            error_msg = str(e)
            print(f"  Warning: VLM call failed: {error_msg}")

            # Handle CUDA/NVML errors
            if "nvml" in error_msg.lower() or "CUDA" in error_msg:
                print("  Detected CUDA/NVML error. Attempting to recover...")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

            raise

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from VLM response.

        Args:
            response: VLM response string

        Returns:
            Parsed dictionary, or empty dict on failure
        """
        # Pre-processing: remove invalid JSON number prefixes (+0.12 → 0.12).
        # JSON spec forbids a leading '+' on numbers; some VLMs emit it anyway.
        response = re.sub(r'(?<!["\w])\+(\d)', r'\1', response)

        # Strategy 1: existing regex approach (fast path, works for most responses)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: raw_decode from first '{' (handles code fences and trailing content)
        start = response.find('{')
        if start != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(response, start)
                return obj
            except json.JSONDecodeError:
                pass

        # Strategy 3: truncated JSON recovery — balance unmatched brackets and retry.
        # Handles VLM output that was cut off mid-stream (e.g. a long "findings" array).
        if start != -1:
            partial = response[start:].rstrip()
            partial = re.sub(r',\s*$', '', partial)       # trailing comma
            partial = re.sub(r'"[^"]*$', '', partial)     # dangling open string
            open_braces = partial.count('{') - partial.count('}')
            open_brackets = partial.count('[') - partial.count(']')
            suffix = ']' * max(0, open_brackets) + '}' * max(0, open_braces)
            if suffix:
                try:
                    obj = json.loads(partial + suffix)
                    return obj
                except json.JSONDecodeError:
                    pass

        raise RuntimeError(f"JSON parse error in VLM response. Response preview: {response[:300]}")

    def invoke_vlm_for_json(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Dict[str, Any]:
        """
        Call VLM and parse JSON response, retrying on parse errors.

        Args:
            prompt: Prompt string
            images: Optional list of image paths
            max_retries: Maximum number of attempts (default 3)
            retry_delay: Seconds to wait between retries (default 2)

        Returns:
            Parsed JSON dictionary

        Raises:
            RuntimeError: If parsing fails on all attempts
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            response = self.invoke_vlm(prompt, images)
            try:
                return self.parse_json_response(response)
            except RuntimeError as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  JSON parse failed (attempt {attempt}/{max_retries}), retrying in {retry_delay}s... Error: {e}")
                    time.sleep(retry_delay)
        raise last_error

    def save_json(self, data: Dict[str, Any], filename: str, subdir: str = "") -> Path:
        """
        Save dictionary as JSON file.

        Args:
            data: Dictionary to save
            filename: File name (with or without .json extension)
            subdir: Optional subdirectory within output_dir

        Returns:
            Path to saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'

        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir

        filepath = save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def _save_prompt(self, prompt: str, question: Dict[str, Any], label: str) -> Path:
        """Save a prompt string to outputs/prompts/{modality}/{dataset}/{q_type}/{row_no}/{label}.txt"""
        import re
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))
        modality = question.get('modality', 'vision')
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"
        save_dir = self.output_dir / f"prompts/{modality}/{dataset_name}/{q_type_str}/{row_no}"
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{label}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        return filepath

    def log(self, message: str, level: str = "INFO"):
        """Log a message with agent name prefix"""
        print(f"[{self.agent_name}] {level}: {message}")

    @staticmethod
    def _extract_question_context_fields(question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Q5/Q6/Q7 specific fields from the question's 'example' text.

        Returns a dict with zero or more of:
          - target_class  (Q6): the flip-target class label
          - queried_part  (Q5): the feature/region being masked
          - part_to_change (Q7): the feature/region being removed/changed
        """
        q_type = question.get('q_type')
        example = question.get('example', '') or question.get('question', '') or ''
        q_template = question.get('q', '')
        result = {}

        if q_type == 6:
            # "flip the model into X" or "flip the model's prediction to X"
            m = re.search(
                r"flip the (?:model(?:'s prediction)?|prediction)(?:\s+into\s+|\s+to\s+)(.+?)(?:\?|$)",
                example, re.IGNORECASE
            )
            if m:
                result['target_class'] = m.group(1).strip().rstrip('?').strip()

        elif q_type == 5:
            # Text pattern: "mask the word 'X' from this input"
            m = re.search(r"mask the word ['\"](.+?)['\"]", example, re.IGNORECASE)
            if m:
                result['queried_part'] = m.group(1).strip()
            else:
                # Tabular pattern: "mask the X feature of this" or "mask the X of this"
                m = re.search(r'mask the (.+?)(?:\s+feature\b|\s+(?:of|from)\s+this)', example, re.IGNORECASE)
                if m:
                    result['queried_part'] = m.group(1).strip()
                else:
                    # Fallback: {placeholder} in q template (skip generic placeholders)
                    m2 = re.search(r'\{(.+?)\}', q_template)
                    if m2 and m2.group(1) not in ('certain', 'certain_part'):
                        result['queried_part'] = m2.group(1)

        elif q_type == 7:
            # Text pattern: "remove/change the word 'X' from this input"
            m = re.search(r"remove/change (?:the )?word ['\"](.+?)['\"]", example, re.IGNORECASE)
            if m:
                result['part_to_change'] = m.group(1).strip()
            else:
                # Tabular/general pattern: "remove/change X,"
                m = re.search(r'remove/change\s+(.+?)(?:,|\?{1,2}|$)', example, re.IGNORECASE)
                if m:
                    result['part_to_change'] = m.group(1).strip().rstrip('?,').strip()
                else:
                    # Fallback: {placeholder} in q template
                    m2 = re.search(r'\{(.+?)\}', q_template)
                    if m2:
                        result['part_to_change'] = m2.group(1)

        return result

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main execution method. Must be implemented by subclasses.
        """
        pass

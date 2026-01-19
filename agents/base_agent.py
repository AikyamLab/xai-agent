"""
Base Agent class for XAI Agent Framework

Provides common functionality shared across all agents.
"""

from __future__ import annotations
import json
import os
import re
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
            if images and hasattr(self.vlm, 'invoke_with_images'):
                return self.vlm.invoke_with_images(prompt, images)
            elif images and hasattr(self.vlm, 'invoke_multimodal'):
                return self.vlm.invoke_multimodal(prompt, images)
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
        try:
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON parse error: {e}")
        except Exception as e:
            print(f"  Warning: Failed to parse response: {e}")

        return {}

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

    def log(self, message: str, level: str = "INFO"):
        """Log a message with agent name prefix"""
        print(f"[{self.agent_name}] {level}: {message}")

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main execution method. Must be implemented by subclasses.
        """
        pass

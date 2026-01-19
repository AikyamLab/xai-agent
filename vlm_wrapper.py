"""
Vision-Language Model Wrapper (Native Implementation)

A pure Python wrapper for Vision-Language Models without LangChain dependency.
Supports Qwen3-VL-8B-Instruct and other HuggingFace VLMs.
"""

import torch
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import os

# Core transformers imports
from transformers import (
    AutoProcessor,
    AutoConfig,
    GenerationConfig
)

# Try to import Qwen3-VL specific classes
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False
    print("Warning: Qwen3VLForConditionalGeneration not available. "
          "Please update transformers: pip install transformers>=4.45.0")

# Try to import qwen_vl_utils for vision processing
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen_vl_utils not available. "
          "Install with: pip install qwen-vl-utils")

# Fallback imports for other model types
from transformers import (
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
)


class VisionLanguageModel:
    """
    Native wrapper for Vision-Language Models.

    Default model: Qwen/Qwen3-VL-8B-Instruct

    Supports models like:
    - Qwen/Qwen3-VL-8B-Instruct (default, recommended)
    - Qwen/Qwen2-VL-7B-Instruct
    - Other HuggingFace VLMs
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        min_new_tokens: int = 1,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Initialize the VLM wrapper.

        Args:
            model_id: HuggingFace model ID (default: Qwen/Qwen3-VL-8B-Instruct)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate
            min_new_tokens: Minimum number of tokens to generate
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling (True for Qwen3-VL)
            use_flash_attention: Whether to use flash_attention_2 (recommended for Qwen3-VL)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.use_flash_attention = use_flash_attention

        self.model = None
        self.processor = None

        # Image size limits for Qwen3-VL
        self.MIN_PIXELS = 128 * 28 * 28
        self.MAX_PIXELS = 512 * 28 * 28

        # Check if this is a Qwen3-VL model
        self.is_qwen3_vl = "qwen3-vl" in model_id.lower() or "qwen3vl" in model_id.lower()
        self.is_qwen_vl = "qwen" in model_id.lower() and "vl" in model_id.lower()

        self._load_model()

    def _load_model(self):
        """Load the VLM model and processor."""
        print(f"Loading VLM: {self.model_id} on {self.device}")

        try:
            # Load processor with appropriate settings
            if self.is_qwen_vl:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    min_pixels=self.MIN_PIXELS,
                    max_pixels=self.MAX_PIXELS,
                    trust_remote_code=True
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    use_fast=False
                )

            model_loaded = False

            # For Qwen3-VL models, use specific model class
            if self.is_qwen3_vl and QWEN3_VL_AVAILABLE:
                model_loaded = self._load_qwen3_vl_model()

            # For other Qwen-VL models or if Qwen3 class not available
            if not model_loaded and self.is_qwen_vl:
                model_loaded = self._load_qwen_vl_model()

            # Fallback to generic loading for other models
            if not model_loaded:
                model_loaded = self._load_generic_model()

            if not model_loaded:
                raise RuntimeError(
                    f"Failed to load {self.model_id} with any generation-compatible model class. "
                    f"Please ensure the model supports text generation."
                )

            print(f"VLM loaded successfully on {self.device}")
            print(f"  Model class: {self.model.__class__.__name__}")

        except Exception as e:
            print(f"Error loading VLM: {e}")
            raise

    def _load_qwen3_vl_model(self) -> bool:
        """Load Qwen3-VL model with appropriate settings."""
        try:
            print(f"  Loading with Qwen3VLForConditionalGeneration...")

            # Determine dtype
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

            # Try with flash_attention_2 first
            if self.use_flash_attention and self.device == "cuda":
                try:
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2",
                    ).eval()
                    print(f"  ✓ Loaded with flash_attention_2")
                    return True
                except Exception as e:
                    print(f"  Warning: flash_attention_2 failed: {e}")
                    print(f"  Falling back to standard attention...")

            # Fallback to standard attention
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            ).eval()

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            print(f"  ✓ Loaded with standard attention")
            return True

        except Exception as e:
            print(f"  Qwen3VLForConditionalGeneration failed: {e}")
            return False

    def _load_qwen_vl_model(self) -> bool:
        """Load generic Qwen-VL model (Qwen2-VL, etc.)."""
        try:
            print(f"  Trying AutoModelForVision2Seq for Qwen-VL...")

            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            ).eval()

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            print(f"  ✓ Loaded with AutoModelForVision2Seq")
            return True

        except Exception as e:
            print(f"  AutoModelForVision2Seq failed: {e}")
            return False

    def _load_generic_model(self) -> bool:
        """Load model using generic Auto classes."""
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Try different model classes
        model_classes = [
            (AutoModelForVision2Seq, "AutoModelForVision2Seq"),
            (AutoModelForImageTextToText, "AutoModelForImageTextToText"),
            (AutoModelForCausalLM, "AutoModelForCausalLM"),
        ]

        for model_class, class_name in model_classes:
            try:
                print(f"  Trying {class_name}...")
                self.model = model_class.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                ).eval()

                if self.device != "cuda":
                    self.model = self.model.to(self.device)

                print(f"  ✓ Loaded with {class_name}")
                return True

            except Exception as e:
                print(f"  {class_name} failed: {e}")

        return False

    def invoke(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, List[Image.Image], str, List[str]]] = None,
        **kwargs
    ) -> str:
        """
        Invoke the VLM with text and optional images.

        Args:
            prompt: Text prompt
            images: Optional image(s) - PIL Image, image path, or list of either
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        if images:
            # Use first image if list provided
            image = images[0] if isinstance(images, list) else images
            return self.generate_with_image(prompt, image, **kwargs)

        return self._generate_text(prompt, **kwargs)

    def invoke_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """
        Invoke VLM with multiple images.

        Args:
            prompt: Text prompt
            image_paths: List of image paths
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        return self.generate_with_images(prompt, image_paths, **kwargs)

    def invoke_multimodal(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """Alias for invoke_with_images."""
        return self.generate_with_images(prompt, image_paths, **kwargs)

    def _generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text response (text-only mode).

        Args:
            prompt: Text prompt
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        # Add system message for instruction following
        system_message = (
            "You are an AI assistant that follows instructions precisely. "
            "When given a task with a specific format to follow, "
            "you MUST follow that exact format. Never respond with greetings or small talk. "
            "Always focus on the task at hand and provide structured responses as requested."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.processor.tokenizer(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        # Generate
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_k=kwargs.get('top_k', self.top_k),
            top_p=kwargs.get('top_p', self.top_p),
            do_sample=kwargs.get('do_sample', self.do_sample)
        )

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        # Decode - only get the newly generated tokens
        output_ids = gen_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_assistant_response(output_text)

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        **kwargs
    ) -> str:
        """
        Generate response with single image input.

        Args:
            prompt: Text prompt
            image: Image path or PIL Image
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        # Convert to path string if PIL Image
        if isinstance(image, Image.Image):
            # Save temporarily if needed for Qwen3-VL processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f.name)
                image_path = f.name
                is_temp = True
        else:
            image_path = image
            is_temp = False

        try:
            if self.is_qwen3_vl and QWEN_VL_UTILS_AVAILABLE:
                return self._generate_qwen3_vl(prompt, [image_path], **kwargs)
            elif self.is_qwen_vl:
                return self._generate_qwen_vl(prompt, image_path, **kwargs)
            else:
                return self._generate_generic_vl(prompt, image_path, **kwargs)
        finally:
            # Clean up temp file
            if is_temp and os.path.exists(image_path):
                os.unlink(image_path)

    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, Image.Image]],
        image_labels: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate response with multiple image inputs.

        Args:
            prompt: Text prompt
            images: List of image paths or PIL Images
            image_labels: Optional labels for images
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        if not images:
            return self._generate_text(prompt, **kwargs)

        if len(images) == 1:
            return self.generate_with_image(prompt, images[0], **kwargs)

        # Convert PIL Images to paths
        image_paths = []
        temp_files = []

        for img in images:
            if isinstance(img, Image.Image):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    img.save(f.name)
                    image_paths.append(f.name)
                    temp_files.append(f.name)
            else:
                image_paths.append(img)

        try:
            if self.is_qwen3_vl and QWEN_VL_UTILS_AVAILABLE:
                return self._generate_qwen3_vl(prompt, image_paths, image_labels, **kwargs)
            elif self.is_qwen_vl:
                return self._generate_qwen_vl_multi(prompt, image_paths, image_labels, **kwargs)
            else:
                return self._generate_generic_vl_multi(prompt, image_paths, image_labels, **kwargs)
        finally:
            # Clean up temp files
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)

    def _generate_qwen3_vl(
        self,
        prompt: str,
        image_paths: List[str],
        image_labels: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen3-VL with qwen_vl_utils.

        Args:
            prompt: Text prompt
            image_paths: List of image paths
            image_labels: Optional labels for images
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        # Build content list with images and labels
        content = []

        for i, img_path in enumerate(image_paths):
            if image_labels and i < len(image_labels):
                content.append({"type": "text", "text": f"[{image_labels[i]}]:"})
            content.append({"type": "image", "image": img_path})

        # Add the main prompt
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # Apply chat template
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision inputs using qwen_vl_utils
        img_inputs, vid_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[chat_text],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_k=kwargs.get('top_k', self.top_k),
            top_p=kwargs.get('top_p', self.top_p),
            do_sample=kwargs.get('do_sample', self.do_sample)
        )

        # Generate
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        # Decode only new tokens
        output_ids = gen_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_assistant_response(output_text)

    def _generate_qwen_vl(
        self,
        prompt: str,
        image_path: str,
        **kwargs
    ) -> str:
        """Generate response using Qwen2-VL style processing."""
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Prepare messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        # Generate
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_k=kwargs.get('top_k', self.top_k),
            top_p=kwargs.get('top_p', self.top_p),
            do_sample=kwargs.get('do_sample', self.do_sample)
        )

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        # Decode
        output_ids = gen_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_assistant_response(output_text)

    def _generate_qwen_vl_multi(
        self,
        prompt: str,
        image_paths: List[str],
        image_labels: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate response with multiple images using Qwen2-VL style."""
        # Load images
        loaded_images = []
        for img_path in image_paths:
            loaded_images.append(Image.open(img_path).convert('RGB'))

        # Build content
        content = []
        for i, img in enumerate(loaded_images):
            label = image_labels[i] if image_labels and i < len(image_labels) else f"Image {i+1}"
            content.append({"type": "text", "text": f"[{label}]:"})
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": f"\n{prompt}"})

        messages = [{"role": "user", "content": content}]

        # Process
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=loaded_images,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        # Generate
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_k=kwargs.get('top_k', self.top_k),
            top_p=kwargs.get('top_p', self.top_p),
            do_sample=kwargs.get('do_sample', self.do_sample)
        )

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        output_ids = gen_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_assistant_response(output_text)

    def _generate_generic_vl(
        self,
        prompt: str,
        image_path: str,
        **kwargs
    ) -> str:
        """Generate response using generic VL model processing."""
        image = Image.open(image_path).convert('RGB')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_k=kwargs.get('top_k', self.top_k),
            top_p=kwargs.get('top_p', self.top_p),
            do_sample=kwargs.get('do_sample', self.do_sample)
        )

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        output_text = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_assistant_response(output_text)

    def _generate_generic_vl_multi(
        self,
        prompt: str,
        image_paths: List[str],
        image_labels: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate response with multiple images using generic processing."""
        loaded_images = [Image.open(p).convert('RGB') for p in image_paths]

        content = []
        for i, img in enumerate(loaded_images):
            label = image_labels[i] if image_labels and i < len(image_labels) else f"Image {i+1}"
            content.append({"type": "text", "text": f"[{label}]:"})
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": f"\n{prompt}"})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=loaded_images,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_k=kwargs.get('top_k', self.top_k),
            top_p=kwargs.get('top_p', self.top_p),
            do_sample=kwargs.get('do_sample', self.do_sample)
        )

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        output_text = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_assistant_response(output_text)

    def _extract_assistant_response(self, output_text: str) -> str:
        """
        Extract assistant's response from full conversation output.

        Args:
            output_text: Full conversation text

        Returns:
            Cleaned assistant response
        """
        output_text = output_text.strip()

        # Try to extract assistant response from conversation format
        if "assistant\n" in output_text:
            output_text = output_text.split("assistant\n")[-1].strip()
        elif "assistant:" in output_text:
            output_text = output_text.split("assistant:")[-1].strip()

        # Remove any remaining system/user markers
        for marker in ["system\n", "user\n", "system:", "user:"]:
            if output_text.startswith(marker):
                output_text = output_text[len(marker):].strip()

        # For ReAct-style responses, extract from first "Thought:"
        if "Thought:" in output_text:
            thought_index = output_text.find("Thought:")
            output_text = output_text[thought_index:]

        return output_text

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "model_class": self.model.__class__.__name__ if self.model else None,
            "is_qwen3_vl": self.is_qwen3_vl,
            "is_qwen_vl": self.is_qwen_vl,
            "qwen_vl_utils_available": QWEN_VL_UTILS_AVAILABLE
        }


# Test code
if __name__ == "__main__":
    print("Testing VisionLanguageModel (Native) with Qwen3-VL...")

    # Initialize VLM with Qwen3-VL-8B-Instruct
    vlm = VisionLanguageModel(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        temperature=0.1
    )

    print(f"\nModel info: {vlm.get_info()}")

    # Test text-only generation
    print("\n=== Test: Text-only generation ===")

    test_questions = [
        "What is 2+2?",
        "What color is the sky?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        response = vlm.invoke(q)
        print(f"A: {response if response else '(no response)'}")

    print("\nVLM wrapper test complete!")

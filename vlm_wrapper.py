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
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        min_new_tokens: int = 1,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_flash_attention: bool = True,
        cache_dir: Optional[str] = None
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
            cache_dir: Directory to cache downloaded models.
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
        self.cache_dir = cache_dir

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
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    use_fast=False,
                    cache_dir=self.cache_dir
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
                        cache_dir=self.cache_dir
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
                cache_dir=self.cache_dir
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
                trust_remote_code=True,
                cache_dir=self.cache_dir
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
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=self.cache_dir)
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
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
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


class GeminiVLM:
    """
    Gemini API wrapper implementing the same interface as VisionLanguageModel.

    Supports models like:
    - gemini-2.5-pro
    - gemini-3-pro
    - gemini-2.5-flash
    - gemini-3-flash
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        max_new_tokens: int = 8192,
        **kwargs
    ):
        """
        Initialize Gemini API wrapper.

        Args:
            model_id: Gemini model name (e.g., 'gemini-2.5-pro', 'gemini-3-pro')
            temperature: Sampling temperature
            max_new_tokens: Maximum output tokens
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Set it with: export GEMINI_API_KEY='your-api-key'"
            )

        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        self.client = genai.Client(api_key=api_key)

        # Retry settings for rate limiting
        self._max_retries = 5
        self._base_retry_delay = 10  # seconds

        self._system_instruction = (
            "You are an AI assistant that follows instructions precisely. "
            "When given a task with a specific format to follow, "
            "you MUST follow that exact format. Never respond with greetings or small talk. "
            "Always focus on the task at hand and provide structured responses as requested."
        )

        print(f"GeminiVLM initialized: {self.model_id}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max output tokens: {self.max_new_tokens}")

    def invoke(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, List[Image.Image], str, List[str]]] = None,
        **kwargs
    ) -> str:
        """
        Invoke the Gemini API with text and optional images.

        Args:
            prompt: Text prompt
            images: Optional image(s) - PIL Image, image path, or list of either
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        if images:
            if isinstance(images, list):
                image_paths = []
                for img in images:
                    if isinstance(img, Image.Image):
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                            img.save(f.name)
                            image_paths.append(f.name)
                    else:
                        image_paths.append(img)
                return self.invoke_with_images(prompt, image_paths, **kwargs)
            elif isinstance(images, Image.Image):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    images.save(f.name)
                    return self.invoke_with_images(prompt, [f.name], **kwargs)
            else:
                return self.invoke_with_images(prompt, [images], **kwargs)

        return self._generate_text(prompt, **kwargs)

    def _call_with_retry(self, contents, config):
        """Call Gemini API with automatic retry on rate limit (429) errors."""
        import time
        import re as _re

        for attempt in range(self._max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Extract retry delay from error message if available
                    delay_match = _re.search(r'retry in ([\d.]+)s', error_str)
                    if delay_match:
                        delay = float(delay_match.group(1)) + 1
                    else:
                        delay = self._base_retry_delay * (2 ** attempt)

                    if attempt < self._max_retries - 1:
                        print(f"  Rate limited (attempt {attempt + 1}/{self._max_retries}). "
                              f"Retrying in {delay:.0f}s...")
                        time.sleep(delay)
                    else:
                        print(f"  Rate limited: max retries ({self._max_retries}) exceeded.")
                        raise
                else:
                    raise

    def _generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text-only response via Gemini API."""
        config = self._types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            system_instruction=self._system_instruction,
        )

        return self._call_with_retry(prompt, config)

    def invoke_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """
        Invoke Gemini API with multiple images.

        Args:
            prompt: Text prompt
            image_paths: List of image paths
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        parts = []
        for path in image_paths:
            if isinstance(path, Image.Image):
                parts.append(path)
            else:
                img = Image.open(path).convert('RGB')
                parts.append(img)
        parts.append(prompt)

        config = self._types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            system_instruction=self._system_instruction,
        )

        return self._call_with_retry(parts, config)

    def invoke_multimodal(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """Alias for invoke_with_images."""
        return self.invoke_with_images(prompt, image_paths, **kwargs)

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        **kwargs
    ) -> str:
        """Generate response with single image input."""
        if isinstance(image, Image.Image):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f.name)
                return self.invoke_with_images(prompt, [f.name], **kwargs)
        return self.invoke_with_images(prompt, [image], **kwargs)

    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, Image.Image]],
        image_labels: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate response with multiple image inputs."""
        if not images:
            return self._generate_text(prompt, **kwargs)

        # Convert PIL images to paths if needed
        image_paths = []
        for img in images:
            if isinstance(img, Image.Image):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    img.save(f.name)
                    image_paths.append(f.name)
            else:
                image_paths.append(img)

        # Build content with labels
        parts = []
        for i, path in enumerate(image_paths):
            if image_labels and i < len(image_labels):
                parts.append(f"[{image_labels[i]}]:")
            img = Image.open(path).convert('RGB')
            parts.append(img)
        parts.append(prompt)

        config = self._types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            system_instruction=self._system_instruction,
        )

        return self._call_with_retry(parts, config)

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "type": "gemini_api",
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }


class ClaudeVLM:
    """
    Claude API wrapper implementing the same interface as VisionLanguageModel.

    Supports models like:
    - claude-sonnet-4-5-20250929
    - claude-haiku-4-5-20251001
    - claude-opus-4-6
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
        max_new_tokens: int = 8192,
        **kwargs
    ):
        """
        Initialize Claude API wrapper.

        Args:
            model_id: Claude model name (e.g., 'claude-sonnet-4-5-20250929')
            temperature: Sampling temperature
            max_new_tokens: Maximum output tokens
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-api-key'"
            )

        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)

        self._system_instruction = (
            "You are an AI assistant that follows instructions precisely. "
            "When given a task with a specific format to follow, "
            "you MUST follow that exact format. Never respond with greetings or small talk. "
            "Always focus on the task at hand and provide structured responses as requested."
        )

        # Retry settings for rate limiting
        self._max_retries = 5
        self._base_retry_delay = 10  # seconds

        print(f"ClaudeVLM initialized: {self.model_id}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max output tokens: {self.max_new_tokens}")

    def _call_with_retry(self, messages, system=None):
        """Call Claude API with automatic retry on rate limit (429) errors."""
        import time

        kwargs = {
            "model": self.model_id,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        for attempt in range(self._max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                return response.content[0].text
            except self._anthropic.RateLimitError as e:
                delay = self._base_retry_delay * (2 ** attempt)
                if attempt < self._max_retries - 1:
                    print(f"  Rate limited (attempt {attempt + 1}/{self._max_retries}). "
                          f"Retrying in {delay:.0f}s...")
                    time.sleep(delay)
                else:
                    print(f"  Rate limited: max retries ({self._max_retries}) exceeded.")
                    raise
            except self._anthropic.APIError as e:
                if e.status_code in (500, 529):  # Internal server error or overloaded
                    delay = self._base_retry_delay * (2 ** attempt)
                    if attempt < self._max_retries - 1:
                        label = "API overloaded" if e.status_code == 529 else "Internal server error (500)"
                        print(f"  {label} (attempt {attempt + 1}/{self._max_retries}). "
                              f"Retrying in {delay:.0f}s...")
                        time.sleep(delay)
                    else:
                        raise
                else:
                    raise

    def _encode_image(self, image_path: str) -> dict:
        """Encode an image file as a base64 content block for Claude API."""
        import base64

        if isinstance(image_path, Image.Image):
            import io
            buffer = io.BytesIO()
            image_path.save(buffer, format='PNG')
            image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
            media_type = "image/png"
        else:
            ext = os.path.splitext(image_path)[1].lower()
            media_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }
            media_type = media_type_map.get(ext, 'image/png')

            with open(image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            }
        }

    def _generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text-only response via Claude API."""
        messages = [{"role": "user", "content": prompt}]
        return self._call_with_retry(messages, system=self._system_instruction)

    def invoke(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, List[Image.Image], str, List[str]]] = None,
        **kwargs
    ) -> str:
        """
        Invoke the Claude API with text and optional images.

        Args:
            prompt: Text prompt
            images: Optional image(s) - PIL Image, image path, or list of either
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        if images:
            if isinstance(images, list):
                return self.invoke_with_images(prompt, images, **kwargs)
            else:
                return self.invoke_with_images(prompt, [images], **kwargs)

        return self._generate_text(prompt, **kwargs)

    def invoke_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """
        Invoke Claude API with multiple images.

        Args:
            prompt: Text prompt
            image_paths: List of image paths or PIL Images
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        content = []
        for path in image_paths:
            content.append(self._encode_image(path))
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        return self._call_with_retry(messages, system=self._system_instruction)

    def invoke_multimodal(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """Alias for invoke_with_images."""
        return self.invoke_with_images(prompt, image_paths, **kwargs)

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        **kwargs
    ) -> str:
        """Generate response with single image input."""
        return self.invoke_with_images(prompt, [image], **kwargs)

    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, Image.Image]],
        image_labels: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate response with multiple image inputs."""
        if not images:
            return self._generate_text(prompt, **kwargs)

        content = []
        for i, img in enumerate(images):
            if image_labels and i < len(image_labels):
                content.append({"type": "text", "text": f"[{image_labels[i]}]:"})
            content.append(self._encode_image(img))
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        return self._call_with_retry(messages, system=self._system_instruction)

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "type": "claude_api",
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }


# ============================================================================
# Tinker API VLM
# ============================================================================

# Tinker API imports (lazy)
try:
    import tinker
    from tinker import types as tinker_types
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False


class TinkerVisionLanguageModel:
    """
    Tinker API wrapper for Vision-Language Models.

    Default model: Qwen/Qwen3-VL-30B-A3B-Instruct

    Supports models like:
    - Qwen/Qwen3-VL-30B-A3B-Instruct (default)
    - Qwen/Qwen3-VL-235B-A22B-Instruct
    - Other Tinker-supported VLMs
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        tinker_api_key: Optional[str] = None,
        **kwargs,
    ):
        if not TINKER_AVAILABLE:
            raise ImportError(
                "tinker package not available. Install with: pip install tinker"
            )

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_new_tokens
        self.top_p = top_p

        # API configuration
        if tinker_api_key:
            os.environ["TINKER_API_KEY"] = tinker_api_key

        # Detect model type
        self.is_qwen_vl = "qwen" in model_id.lower() and "vl" in model_id.lower()

        # Initialize tokenizer and Tinker client
        self.tokenizer = None
        self.sampling_client = None
        self._initialize_client()

        print(f"TinkerVLM initialized with model: {model_id}")

    def _initialize_client(self):
        """Initialize the Tinker SamplingClient and tokenizer."""
        print("Creating Tinker ServiceClient...")
        service_client = tinker.ServiceClient()

        print(f"Creating SamplingClient for {self.model_id}...")
        self.sampling_client = service_client.create_sampling_client(
            base_model=self.model_id
        )
        print(f"Tinker SamplingClient initialized for {self.model_id}")

        # Get tokenizer from the sampling client (server-provided, no HF download needed)
        try:
            self.tokenizer = self.sampling_client.get_tokenizer()
            print("Tokenizer loaded from Tinker client")
        except Exception as e:
            print(f"Warning: Could not get tokenizer from Tinker client: {e}")
            # Fallback: try loading from HuggingFace transformers
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                print(f"Tokenizer loaded from HuggingFace for {self.model_id}")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load tokenizer from both Tinker client ({e}) "
                    f"and HuggingFace ({e2}). Tokenizer is required for Tinker API."
                )

    def _image_to_bytes(self, image: Union[str, Image.Image]) -> bytes:
        """Convert image path or PIL Image to bytes."""
        if isinstance(image, str):
            with open(image, "rb") as f:
                return f.read()
        elif isinstance(image, Image.Image):
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _get_image_format(self, image: Union[str, Image.Image]) -> str:
        """Get image format string."""
        if isinstance(image, str):
            ext = os.path.splitext(image)[1].lower()
            format_map = {
                ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
                ".gif": "gif", ".webp": "webp",
            }
            return format_map.get(ext, "png")
        return "png"

    def _encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs using the tokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_text_only_input(self, prompt: str):
        """Build ModelInput for text-only prompts."""
        system_message = (
            "You are an AI assistant that follows instructions precisely. "
            "When given a task with a specific format to follow, "
            "you MUST follow that exact format. Never respond with greetings or small talk. "
            "Always focus on the task at hand and provide structured responses as requested."
        )

        if self.is_qwen_vl:
            full_text = (
                f"<|im_start|>system\n{system_message}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            full_text = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"

        tokens = self._encode_text(full_text)

        # Try ModelInput.from_ints first, fall back to chunks
        if hasattr(tinker_types, 'ModelInput') and hasattr(tinker_types.ModelInput, 'from_ints'):
            return tinker_types.ModelInput.from_ints(tokens)
        elif hasattr(tinker, 'ModelInput') and hasattr(tinker.ModelInput, 'from_ints'):
            return tinker.ModelInput.from_ints(tokens)
        else:
            return tinker.ModelInput(
                chunks=[tinker_types.EncodedTextChunk(tokens=tokens)]
            )

    def _build_model_input(
        self,
        prompt: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
        image_labels: Optional[List[str]] = None,
    ):
        """Build Tinker ModelInput for multimodal input with images."""
        if not images:
            return self._build_text_only_input(prompt)

        # For multimodal, we need to build chunks with ImageChunk
        chunks = []

        system_message = (
            "You are an AI assistant that follows instructions precisely. "
            "When given a task with a specific format to follow, "
            "you MUST follow that exact format. Never respond with greetings or small talk. "
            "Always focus on the task at hand and provide structured responses as requested."
        )

        if self.is_qwen_vl:
            header = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n"
            header_tokens = self._encode_text(header)
            if header_tokens:
                chunks.append(tinker_types.EncodedTextChunk(tokens=header_tokens))

            for i, img in enumerate(images):
                if image_labels and i < len(image_labels):
                    label_tokens = self._encode_text(f"[{image_labels[i]}]:\n")
                    if label_tokens:
                        chunks.append(tinker_types.EncodedTextChunk(tokens=label_tokens))

                vision_start_tokens = self._encode_text("<|vision_start|>")
                if vision_start_tokens:
                    chunks.append(tinker_types.EncodedTextChunk(tokens=vision_start_tokens))

                img_bytes = self._image_to_bytes(img)
                img_format = self._get_image_format(img)
                chunks.append(tinker_types.ImageChunk(data=img_bytes, format=img_format))

                vision_end_tokens = self._encode_text("<|vision_end|>\n")
                if vision_end_tokens:
                    chunks.append(tinker_types.EncodedTextChunk(tokens=vision_end_tokens))

            footer = f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
            footer_tokens = self._encode_text(footer)
            if footer_tokens:
                chunks.append(tinker_types.EncodedTextChunk(tokens=footer_tokens))
        else:
            text = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"
            tokens = self._encode_text(text)
            chunks.append(tinker_types.EncodedTextChunk(tokens=tokens))

        return tinker.ModelInput(chunks=chunks)

    def _sample(self, prompt_or_input) -> str:
        """Run inference using Tinker sampling.

        Args:
            prompt_or_input: Either a raw text string (text-only) or
                             a tinker.ModelInput (multimodal with images).
        """
        sampling_params = tinker.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        result_future = self.sampling_client.sample(
            prompt=prompt_or_input,
            sampling_params=sampling_params,
            num_samples=1,
        )

        # Handle async result
        if hasattr(result_future, 'result') and callable(result_future.result):
            result = result_future.result()
        else:
            result = result_future

        text = self._extract_text_from_response(result)

        if text:
            return self._extract_assistant_response(text)

        print("Warning: Failed to extract text from Tinker response")
        return ""

    def _extract_text_from_sample(self, sample: Any) -> str:
        """Extract text from a single sample object."""
        # Primary: tokens + tokenizer.decode (standard Tinker pattern)
        if hasattr(sample, 'tokens') and sample.tokens and self.tokenizer:
            try:
                tokens = sample.tokens
                if hasattr(tokens, 'tolist'):
                    tokens = tokens.tolist()
                elif hasattr(tokens, '__iter__') and not isinstance(tokens, (str, bytes)):
                    tokens = list(tokens)
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception as e:
                print(f"Warning: Token decode error: {e}")

        # Fallback: direct text attribute
        if hasattr(sample, 'text') and sample.text:
            return str(sample.text)
        if hasattr(sample, 'completion') and sample.completion:
            return str(sample.completion)
        if isinstance(sample, str):
            return sample
        return ""

    def _extract_text_from_response(self, result: Any) -> str:
        """Extract text from Tinker SampleResponse."""
        try:
            # Strategy 1: sequences attribute (PRIMARY for Tinker API)
            if hasattr(result, 'sequences') and result.sequences:
                try:
                    first_seq = result.sequences[0] if hasattr(result.sequences, '__getitem__') else next(iter(result.sequences))

                    # Check for tokens on sequence object
                    tokens = None
                    if hasattr(first_seq, 'tokens'):
                        tokens = first_seq.tokens
                    elif hasattr(first_seq, 'token_ids'):
                        tokens = first_seq.token_ids
                    elif isinstance(first_seq, (list, tuple)):
                        tokens = list(first_seq)

                    if tokens is not None and self.tokenizer:
                        if hasattr(tokens, 'tolist'):
                            tokens = tokens.tolist()
                        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                        if decoded:
                            return decoded

                except (IndexError, StopIteration, TypeError):
                    pass

            # Strategy 2: Direct text attribute
            if hasattr(result, 'text') and result.text:
                return str(result.text)

            # Strategy 3: Completion attribute
            if hasattr(result, 'completion') and result.completion:
                return str(result.completion)

            # Strategy 4: Samples list
            if hasattr(result, 'samples') and result.samples:
                try:
                    sample = result.samples[0] if hasattr(result.samples, '__getitem__') else next(iter(result.samples))
                    text = self._extract_text_from_sample(sample)
                    if text:
                        return text
                except (IndexError, StopIteration, TypeError):
                    pass

            # Strategy 5: Completions list
            if hasattr(result, 'completions') and result.completions:
                try:
                    completion = result.completions[0] if hasattr(result.completions, '__getitem__') else next(iter(result.completions))
                    if hasattr(completion, 'text'):
                        return str(completion.text)
                    elif isinstance(completion, str):
                        return completion
                except (IndexError, StopIteration):
                    pass

            # Strategy 6: Tokens attribute directly on result
            if hasattr(result, 'tokens') and result.tokens and self.tokenizer:
                return self.tokenizer.decode(result.tokens, skip_special_tokens=True)

            # Strategy 7: Result is string
            if isinstance(result, str):
                return result

            # Debug output if nothing worked
            print(f"Warning: Could not extract text from Tinker response")
            print(f"  Type: {type(result)}")
            print(f"  Attributes: {[a for a in dir(result) if not a.startswith('_')]}")

        except Exception as e:
            print(f"Warning: Error extracting text from Tinker response: {e}")

        return ""

    def _extract_assistant_response(self, output_text: str) -> str:
        """Extract assistant's response from full output."""
        output_text = output_text.strip()

        if "assistant\n" in output_text:
            output_text = output_text.split("assistant\n")[-1].strip()
        elif "assistant:" in output_text:
            output_text = output_text.split("assistant:")[-1].strip()
        elif "Assistant:" in output_text:
            output_text = output_text.split("Assistant:")[-1].strip()

        for marker in ["<|im_end|>", "<|endoftext|>", "system\n", "user\n"]:
            output_text = output_text.replace(marker, "").strip()

        if "Thought:" in output_text:
            thought_index = output_text.find("Thought:")
            output_text = output_text[thought_index:]

        return output_text

    def invoke(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, List[Image.Image], str, List[str]]] = None,
        **kwargs
    ) -> str:
        """Invoke the VLM with text and optional images."""
        if images:
            if not isinstance(images, list):
                images = [images]
            model_input = self._build_model_input(prompt, images)
        else:
            model_input = self._build_text_only_input(prompt)

        return self._sample(model_input)

    def invoke_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """Invoke VLM with multiple images."""
        return self.generate_with_images(prompt, image_paths, **kwargs)

    def invoke_multimodal(
        self,
        prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """Alias for invoke_with_images."""
        return self.generate_with_images(prompt, image_paths, **kwargs)

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        **kwargs
    ) -> str:
        """Generate response with single image input."""
        model_input = self._build_model_input(prompt, [image])
        return self._sample(model_input)

    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, Image.Image]],
        image_labels: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate response with multiple image inputs."""
        if not images:
            return self.invoke(prompt)

        model_input = self._build_model_input(prompt, images, image_labels)
        return self._sample(model_input)

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "backend": "tinker",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "is_qwen_vl": self.is_qwen_vl,
            "client_initialized": self.sampling_client is not None,
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_vlm(model_id: str, **kwargs):
    """
    Factory function to create the appropriate VLM based on model_id.

    Args:
        model_id: Model identifier.
                  Use 'tinker/*' for Tinker API models (e.g. 'tinker/Qwen3-VL-30B-A3B-Instruct'),
                  use 'gemini-*' for Gemini API models,
                  use 'claude-*' for Claude API models,
                  or HuggingFace model IDs for local models.
        **kwargs: Additional arguments passed to the VLM constructor.

    Returns:
        TinkerVisionLanguageModel, ClaudeVLM, GeminiVLM, or VisionLanguageModel instance.
    """
    if model_id.startswith("tinker/"):
        # Convert "tinker/Qwen3-VL-30B-A3B-Instruct" -> "Qwen/Qwen3-VL-30B-A3B-Instruct"
        tinker_model_name = model_id[len("tinker/"):]
        # If user provided just the model name (no org prefix), add Qwen/ prefix
        if "/" not in tinker_model_name:
            tinker_model_id = f"Qwen/{tinker_model_name}"
        else:
            tinker_model_id = tinker_model_name
        return TinkerVisionLanguageModel(model_id=tinker_model_id, **kwargs)
    elif model_id.startswith("gemini-"):
        return GeminiVLM(model_id=model_id, **kwargs)
    elif model_id.startswith("claude-"):
        return ClaudeVLM(model_id=model_id, **kwargs)
    else:
        return VisionLanguageModel(model_id=model_id, **kwargs)


# Test code
if __name__ == "__main__":
    print("Testing VisionLanguageModel (Native) with Qwen3-VL...")

    # Initialize VLM with Qwen3-VL-8B-Instruct
    vlm = VisionLanguageModel(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        temperature=0.0
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

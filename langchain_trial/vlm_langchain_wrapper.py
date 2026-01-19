"""
LangChain wrapper for Vision-Language Models
Supports amd/Instella-VL-1B and other VLMs
"""

import torch
from typing import Any, Dict, List, Optional, Union
from transformers import (
    AutoProcessor,
    AutoModel,
    GenerationConfig
)
from langchain_classic.llms.base import LLM
from langchain_classic.callbacks.manager import CallbackManagerForLLMRun
from PIL import Image


class VisionLanguageModel(LLM):
    """
    LangChain-compatible wrapper for Vision-Language Models

    Supports models like:
    - Qwen/Qwen2-VL-7B-Instruct (default)
    - amd/Instella-VL-1B (lightweight model)
    - Qwen/Qwen2-VL-2B-Instruct
    - Other HuggingFace VLMs
    """

    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: Any = None
    processor: Any = None
    temperature: float = 0.1
    max_new_tokens: int = 1024  # Increased for Qwen2-VL-7B to support detailed explanations
    min_new_tokens: int = 1
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = False  # Use greedy decoding for more stable results

    class Config:
        """Configuration for this pydantic object."""
        extra = 'allow'
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize the VLM wrapper"""
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self):
        """Load the VLM model and processor"""
        print(f"Loading VLM: {self.model_id} on {self.device}")

        try:
            # Load processor
            # Set use_fast=False to avoid warnings about slow processors
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=False
            )

            # Try to load model with generation capabilities
            # Priority: Vision2Seq > ImageTextToText > CausalLM > custom auto_map
            from transformers import (
                AutoConfig,
                AutoModelForVision2Seq,
                AutoModelForImageTextToText,
                AutoModelForCausalLM
            )

            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            model_loaded = False

            # Try AutoModelForVision2Seq first (for Qwen2-VL and similar models)
            if not model_loaded:
                try:
                    print(f"  Trying AutoModelForVision2Seq...")
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    ).eval()
                    model_loaded = True
                    print(f"  ✓ Loaded with AutoModelForVision2Seq")
                except Exception as e:
                    print(f"  AutoModelForVision2Seq failed: {e}")

            # Try AutoModelForImageTextToText
            if not model_loaded:
                try:
                    print(f"  Trying AutoModelForImageTextToText...")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    ).eval()
                    model_loaded = True
                    print(f"  ✓ Loaded with AutoModelForImageTextToText")
                except Exception as e:
                    print(f"  AutoModelForImageTextToText failed: {e}")

            # Try AutoModelForCausalLM
            if not model_loaded:
                try:
                    print(f"  Trying AutoModelForCausalLM...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    ).eval()
                    model_loaded = True
                    print(f"  ✓ Loaded with AutoModelForCausalLM")
                except Exception as e:
                    print(f"  AutoModelForCausalLM failed: {e}")

            # Try custom auto_map if available
            if not model_loaded and hasattr(config, 'auto_map'):
                for key in ['AutoModelForVision2Seq', 'AutoModelForImageTextToText', 'AutoModelForCausalLM']:
                    if key in config.auto_map:
                        try:
                            print(f"  Trying custom {key} from auto_map...")
                            from transformers.dynamic_module_utils import get_class_from_dynamic_module
                            model_class_reference = config.auto_map[key]
                            model_class = get_class_from_dynamic_module(
                                model_class_reference,
                                self.model_id,
                                cache_dir=None,
                                force_download=False,
                                resume_download=None,
                                proxies=None,
                                token=None,
                                revision=None,
                                local_files_only=False,
                                code_revision=None,
                            )
                            self.model = model_class.from_pretrained(
                                self.model_id,
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                device_map="auto" if self.device == "cuda" else None,
                                trust_remote_code=True
                            ).eval()
                            model_loaded = True
                            print(f"  ✓ Loaded with custom {key}")
                            break
                        except Exception as e:
                            print(f"  Custom {key} failed: {e}")

            if not model_loaded:
                raise RuntimeError(
                    f"Failed to load {self.model_id} with any generation-compatible model class. "
                    f"Please ensure the model supports text generation."
                )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            print(f"✓ VLM loaded successfully on {self.device}")
            print(f"  Model class: {self.model.__class__.__name__}")

        except Exception as e:
            print(f"Error loading VLM: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "vision_language_model"

    def invoke(
        self,
        input: str,
        config: Optional[Any] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Invoke the VLM with text and optional images.

        This override ensures that 'images' parameter is properly passed
        to the underlying _call method.

        Args:
            input: Text prompt
            config: Optional config (not used)
            stop: Optional stop sequences
            **kwargs: Additional arguments including 'images' for vision tasks

        Returns:
            Generated text response
        """
        return self._call(input, stop=stop, **kwargs)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the VLM with text prompt and optional images

        Args:
            prompt: Input text prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments including 'images' for vision tasks

        Returns:
            Generated text response
        """
        # Check if images are provided - if so, use vision mode
        images = kwargs.pop('images', None)
        if images:
            # Use first image if list provided
            image = images[0] if isinstance(images, list) else images
            return self.generate_with_image(prompt, image, **kwargs)

        # For text-only (agent mode), use instruction-following format
        # Add system message to help the model follow the ReAct format
        system_message = (
            "You are an AI assistant that follows instructions precisely. "
            "When given a task with a specific format to follow (like Thought/Action/Action Input), "
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
                inputs=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )

        # Decode
        # Qwen2-VL returns full conversation including input
        output_text = self.processor.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract only the assistant's response
        output_text = self._extract_assistant_response(output_text)
        return output_text

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
            # Split by assistant marker and take the last part
            output_text = output_text.split("assistant\n")[-1].strip()
        elif "assistant:" in output_text:
            output_text = output_text.split("assistant:")[-1].strip()

        # Remove any remaining system/user markers that might appear at the start
        for marker in ["system\n", "user\n", "system:", "user:"]:
            if output_text.startswith(marker):
                output_text = output_text[len(marker):].strip()

        # If the response is for a ReAct agent, it should contain "Thought:".
        # Strip any conversational preamble before the first "Thought:".
        if "Thought:" in output_text:
            # Find the index of the first occurrence of "Thought:"
            thought_index = output_text.find("Thought:")
            # Slice the string from that index to the end
            output_text = output_text[thought_index:]

        return output_text

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        **kwargs: Any
    ) -> str:
        """
        Generate response with image input

        Args:
            prompt: Text prompt
            image: Image path or PIL Image
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

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

        # For VLMs, we need to handle image inputs properly
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
                inputs=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )

        # Decode
        # Qwen2-VL returns full conversation including input
        output_text = self.processor.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract only the assistant's response
        output_text = self._extract_assistant_response(output_text)
        return output_text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }


# Test code
if __name__ == "__main__":
    print("Testing VisionLanguageModel wrapper...")

    # Initialize VLM
    vlm = VisionLanguageModel(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        temperature=0.1
    )

    # Test text-only generation
    print("\n=== Test 1: Text-only generation ===")

    # Test with a simple math question (works better with small models)
    test_questions = [
        "What is 2+2?",
        "What color is the sky?",
        "What is machine learning?"
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        response = vlm.invoke(q)
        print(f"A: {response if response else '(no response)'}")

    print("\n✓ VLM wrapper test complete!")

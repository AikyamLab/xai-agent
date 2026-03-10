"""
Stable Diffusion Inpainting for Q6 Counterfactual Image Generation

Uses SD v1.5 Inpainting (runwayml/stable-diffusion-inpainting) to generate
counterfactual images by modifying specified bounding box regions based on
text prompts from the agent's change plan.

Accepts both PIL Image and torch.Tensor inputs; returns the same type as input.

Parallelism design
------------------
Each available CUDA GPU gets its own independent pipeline instance stored in a
thread-safe Queue (_pipeline_pool).  Rollout threads check out a pipeline,
run inference on its dedicated GPU, then return it.  With N GPUs this gives N
truly concurrent SD inferences.  With a single GPU the behaviour is equivalent
to a serialising lock (one slot in the queue), but without any explicit lock.

Scheduler
---------
DPM-Solver++ (20 steps) replaces the default PNDM (50 steps).  Quality is
comparable while wall-clock time drops ~2.5x.
"""

import queue as _queue
import threading
import torch
import numpy as np
from PIL import Image
from typing import Any, Tuple, Union

_pipeline_pool: "_queue.Queue | None" = None   # Queue[pipeline]; one slot per GPU
_pool_init_lock = threading.Lock()


def _build_pipeline_pool() -> "_queue.Queue":
    """Lazily create one SD pipeline per available CUDA GPU (thread-safe)."""
    global _pipeline_pool
    if _pipeline_pool is not None:
        return _pipeline_pool

    with _pool_init_lock:
        if _pipeline_pool is not None:   # another thread beat us here
            return _pipeline_pool

        from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError(
                "[SD Inpainting] No CUDA GPU found. "
                "SD inpainting requires at least one GPU. "
                "Add --gres=gpu:1 (or more) to your SLURM job."
            )

        pool: "_queue.Queue" = _queue.Queue()
        for gpu_id in range(n_gpus):
            device = f"cuda:{gpu_id}"
            print(f"[SD Inpainting] Loading pipeline on {device} ...")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
            ).to(device)
            pipe.safety_checker = None
            # DPM-Solver++ scheduler: 20 steps ≈ PNDM 50 steps quality, ~2.5x faster
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            pool.put(pipe)
            print(f"[SD Inpainting] Pipeline ready on {device}.")

        _pipeline_pool = pool
        print(f"[SD Inpainting] Pool initialised with {n_gpus} GPU(s).")

    return _pipeline_pool


def _to_pil(input_data: Any) -> Tuple[Image.Image, bool]:
    """Convert input to PIL Image. Returns (pil_image, was_tensor)."""
    if isinstance(input_data, Image.Image):
        return input_data.convert("RGB"), False

    if isinstance(input_data, torch.Tensor):
        t = input_data
        if t.dim() == 4:
            t = t.squeeze(0)
        t = t.detach().cpu().float()

        # Undo ImageNet normalization if values are outside [0, 1]
        if t.min() < -0.1:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = t * std + mean

        t = t.clamp(0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr), True

    raise TypeError(f"Unsupported input type: {type(input_data)}")


def _pil_to_tensor(img: Image.Image, original_tensor: torch.Tensor) -> torch.Tensor:
    """Convert PIL back to tensor matching original_tensor format."""
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

    # Re-apply ImageNet normalization if original was normalized
    sample = original_tensor.squeeze(0) if original_tensor.dim() == 4 else original_tensor
    if sample.min() < -0.1:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std

    if original_tensor.dim() == 4:
        t = t.unsqueeze(0)
    return t.to(original_tensor.device, dtype=original_tensor.dtype)


def _make_mask(size: Tuple[int, int], bbox: list) -> Image.Image:
    """Create a binary mask (white=inpaint region) from bounding box.

    Args:
        size: (width, height) of the mask
        bbox: [x1, y1, x2, y2] bounding box
    """
    mask = Image.new("L", size, 0)  # black = keep
    pixels = mask.load()
    w, h = size
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    for y in range(y1, y2):
        for x in range(x1, x2):
            pixels[x, y] = 255  # white = inpaint
    return mask


def generate_counterfactual_image(
    original_input: Union[Image.Image, torch.Tensor],
    bounding_box: list,
    prompt: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
) -> Union[Image.Image, torch.Tensor]:
    """Generate a counterfactual image using SD inpainting.

    Args:
        original_input: Original image (PIL Image or torch Tensor)
        bounding_box: [x1, y1, x2, y2] region to modify
        prompt: Text description of what the region should look like
        num_inference_steps: SD inference steps
        guidance_scale: Classifier-free guidance scale

    Returns:
        Modified image in the same type as original_input (PIL or Tensor)
    """
    pool = _build_pipeline_pool()

    # Convert to PIL (CPU-side, no lock needed)
    pil_image, was_tensor = _to_pil(original_input)
    orig_size = pil_image.size  # (w, h)

    mask = _make_mask(orig_size, bounding_box)

    sd_size = (512, 512)
    pil_resized = pil_image.resize(sd_size, Image.LANCZOS)
    mask_resized = mask.resize(sd_size, Image.NEAREST)

    # Check out a pipeline from the pool (blocks only when all GPUs are busy).
    # Each pipeline owns a dedicated GPU, so concurrent inferences on different
    # GPUs run without any serialisation.
    pipe = pool.get()
    try:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=pil_resized,
                mask_image=mask_resized,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
    finally:
        pool.put(pipe)   # always return, even on exception

    # Resize back to original size
    result_resized = result.resize(orig_size, Image.LANCZOS)

    if was_tensor:
        return _pil_to_tensor(result_resized, original_input)
    else:
        return result_resized

"""
Shared helpers for the representation-space CAV experiment (Part B):
loading the two CUB classifiers, capturing penultimate-layer activations via
a forward pre-hook, and re-running just the classifier head on a modified
activation (for concept ablation). No masking/perturbation logic here --
that's Part A's job (reuses evaluation/masking_utils.py instead).

Both CUB models (models_to_read/vision/load_cub_resnet.py and
load_cub_densenet.py) share the same interface (set_dataset_root, CUB_Dataset,
get_transform, load_model) but differ in which attribute holds the classifier
head ("fc" for resnet, "classifier" for densenet) and its input width.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models_to_read" / "vision"))
import load_cub_resnet
import load_cub_densenet

MODEL_CONFIG = {
    "cub_resnet": {
        "module": load_cub_resnet,
        "checkpoint": "models_to_read/vision/cub_resnet.pth",
        "head_attr": "fc",
        "activation_dim": 2048,
    },
    "cub_densenet": {
        "module": load_cub_densenet,
        "checkpoint": "models_to_read/vision/cub_densenet.pth",
        "head_attr": "classifier",
        "activation_dim": 1920,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_dataset(model_name: str, split: str, dataset_dir: str = "dataset_full"):
    """Return (model, head_module, dataset) for the given CUB model/split."""
    cfg = MODEL_CONFIG[model_name]
    module = cfg["module"]
    module.set_dataset_root(dataset_dir)
    model, transform = module.load_model(cfg["checkpoint"])
    model.eval()
    dataset = module.CUB_Dataset(root=module.DATASET_ROOT, dataset_type=split, transform=transform)
    head_module = getattr(model, cfg["head_attr"])
    return model, head_module, dataset


def get_head(model, model_name: str):
    return getattr(model, MODEL_CONFIG[model_name]["head_attr"])


def get_image_ids_for_dataset(dataset, cub_root: str):
    """
    Return the list of CUB image_ids (1-indexed, as used in
    image_attribute_labels.txt / part_locs.txt) aligned with `dataset`'s
    iteration order.

    CUB_Dataset.img_name_list stores relative image paths (e.g.
    "001.Black_footed_Albatross/xxx.jpg"); images.txt maps those paths to
    image_id, so a reverse lookup gives us the alignment without assuming
    anything about how CUB_Dataset's internal filtering reorders rows.
    """
    import os
    path_to_id = {}
    with open(os.path.join(cub_root, "images.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_str, rel_path = line.split(" ", 1)
            path_to_id[rel_path] = int(id_str)
    return [path_to_id[p] for p in dataset.img_name_list]


def extract_activations(model, head_module, dataset, batch_size: int = 32, num_workers: int = 0):
    """
    Run `dataset` through `model` and return (activations, labels) where
    activations[i] is the penultimate vector fed into the classifier head for
    dataset[i], captured via a forward pre-hook on head_module.fc0.

    num_workers defaults to 0 (no multiprocessing DataLoader workers) since
    forking extra worker processes can fail with OOM on memory-constrained
    nodes; raise it explicitly if running on a machine with more headroom.
    """
    captured = {}

    def _hook(module, inputs):
        captured["activation"] = inputs[0].detach()

    handle = head_module.fc0.register_forward_pre_hook(_hook)

    activations = []
    labels = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.to(DEVICE)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            model(images)
            activations.append(captured["activation"].cpu().numpy())
            labels.append(targets.numpy())

    handle.remove()
    return np.concatenate(activations, axis=0), np.concatenate(labels, axis=0)


def head_forward_probs(model, model_name: str, activation: torch.Tensor) -> torch.Tensor:
    """Run `activation` (penultimate vector, [N, D]) through just the classifier
    head and return softmax probabilities [N, num_classes]."""
    head = get_head(model, model_name)
    with torch.no_grad():
        logits = head(activation)
        return torch.softmax(logits, dim=-1)

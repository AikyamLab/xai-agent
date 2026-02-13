# code to load both model and dataset
# please refer to sample_use.py to get a sample template to use the label maps and dataset


import json
import torch
import timm
import itertools

from torchvision.datasets import STL10
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_dataset(split="test"):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = STL10(
        root="./data",
        split=split,
        download=True,
        transform=transform
    )

    label_map = {
        0: "airplane",
        1: "bird",
        2: "car",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "horse",
        7: "monkey",
        8: "ship",
        9: "truck"
    }

    return dataset, label_map
model = timm.create_model(
    "densenet121",
    pretrained=True,
    num_classes=10
)

# Note: "/content/densenet121_stl10_head.pth" - please change the .pth path accordingly inside torch.load()

model.load_state_dict(
    torch.load("/content/densenet121_stl10_head.pth", map_location=device)
)

model.to(device)
model.eval()

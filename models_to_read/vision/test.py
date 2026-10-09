import json
import torch
import timm

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
        root="dataset/image/stl-10",
        split=split,
        download=False,
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
    "resnet50",
    pretrained=True,
    num_classes=10
)

model.load_state_dict(
    torch.load("models_to_read/vision/stl10_resnet_head.pth", map_location=device)
)

model = model.to(device)
model.eval()

dataset, label_map = load_dataset(split="test")

with torch.no_grad():
    for idx in range(0, 5):

        image, label = dataset[idx]
        label = int(label)

        image = image.unsqueeze(0).to(device)  # [1, 3, 224, 224]

        logits = model(image)
        pred = torch.argmax(logits, dim=1).item()

        label_name = label_map[label]
        pred_name = label_map[pred]

        print("Prediction is: ", pred_name)
        print("Ground truth is: ", label_name)
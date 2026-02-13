# the below is the code to generate q10 outputs from resnet50 for stl10, it has the logic for getting predictions from the model and also getting ground truth. 

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
    "resnet50",
    pretrained=True,
    num_classes=10
)

# Note: "/content/resnet50_stl10_head.pth" - please change the .pth path accordingly inside torch.load()

model.load_state_dict(
    torch.load("/content/resnet50_stl10_head.pth", map_location=device)
)

model.to(device)
model.eval()
dataset, label_map = load_dataset(split="test")

correct_by_class = {i: [] for i in range(10)}
misclassified = []

with torch.no_grad():
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        label = int(label)

        image = image.unsqueeze(0).to(device)
        logits = model(image)
        pred = torch.argmax(logits, dim=1).item()

        entry = {
            "row_no": idx,
            "image_index": idx,
            "target": {
                "value": label,
                "label": label_map[label]
            },
            "predicted": {
                "value": pred,
                "label": label_map[pred]
            }
        }

        if pred == label:
            correct_by_class[label].append(entry)
        else:
            misclassified.append(entry)

q10_output = []
max_samples = 200  # consistent with SNLI-style cap

for mis in misclassified:
    if len(q10_output) >= max_samples:
        break

    cls = mis["target"]["value"]
    correct_pool = correct_by_class[cls]

    if len(correct_pool) < 3:
        continue

    # pick 3 distinct correct examples
    A1, A2, A3 = correct_pool[:3]

    entry = {
        "dataset": "stl10",
        "model": "resnet-50",
        "modality": "image",

        "row_no": [
            A1["row_no"],
            A2["row_no"],
            A3["row_no"],
            mis["row_no"]
        ],

        "image_indices": [
            A1["image_index"],
            A2["image_index"],
            A3["image_index"],
            mis["image_index"]
        ],

        "target": [
            A1["target"],
            A2["target"],
            A3["target"],
            mis["target"]
        ],

        "predicted": [
            A1["predicted"],
            A2["predicted"],
            A3["predicted"],
            mis["predicted"]
        ],

        "example": (
            f"Images {A1['row_no']}, {A2['row_no']}, and {A3['row_no']} "
            f"are correctly classified as {A1['predicted']['label']}, "
            f"whereas image {mis['row_no']} is misclassified as "
            f"{mis['predicted']['label']}."
        ),

        "q_type": 10,
        "q": (
            f"Why are images {A1['row_no']}, {A2['row_no']}, "
            f"{A3['row_no']} classified correctly, while image "
            f"{mis['row_no']} is misclassified by the model?"
        )
    }

    q10_output.append(entry)

with open("q10_stl10_resnet50.json", "w") as f:
    json.dump(q10_output, f, indent=2)

print(f"Saved {len(q10_output)} Q10 samples.")
print(json.dumps(q10_output[:1], indent=2))

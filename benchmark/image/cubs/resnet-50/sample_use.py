# the below is the code to generate q1 outputs from resnet50 for cubs, it has the logic for getting predictions from the model and also getting ground truth. 

import json
import torch
import os

model.eval()

output_q1 = []
count = 120
correct_collected = 0
global_idx = 0  # absolute index over test split

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for idx_in_batch in range(labels.size(0)):
            if correct_collected >= count:
                break

            # absolute dataset index
            absolute_idx = global_idx + idx_in_batch

            true_label = int(labels[idx_in_batch].item())
            pred_label = int(preds[idx_in_batch].item())

            # Q1: only correct predictions
            if pred_label != true_label:
                continue

            # retrieve image identity (CUB style)
            img_name = test_data.img_name_list[absolute_idx]
            img_path = os.path.join(test_data.root, "images", img_name)

            entry = {
                "row_no": int(absolute_idx),
                "image_path": img_path,
                "modality": "image",
                "dataset": "CUB-200-2011",
                "model": "resnet-50",
                "features": {
                    "image_shape": list(images[idx_in_batch].shape)
                },
                "target": {
                    "value": true_label,
                    "label": label_map[true_label]
                },
                "predicted": {
                    "value": pred_label,
                    "label": label_map[pred_label]
                },
                "example": (
                    f"The model predicted the image as "
                    f"{label_map[pred_label]}. "
                    "Which part of the input was most responsible for the model’s prediction?"
                ),
                "q_type": 1,
                "q": "Which part of the input was most responsible for the model’s prediction?"
            }

            output_q1.append(entry)
            correct_collected += 1

        global_idx += labels.size(0)

        if correct_collected >= count:
            break


with open("q1_cubs_resnet50.json", "w") as f:
    json.dump(output_q1, f, indent=2)

print(f"Saved {len(output_q1)} Q1 instances.")
print(json.dumps(output_q1[:2], indent=2))  # sanity check

# this has been deprecated, please refer to /benchmark/image/stl-10/resnet-50 for the updated file


import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

RESNET_PTH_PATH = "resnet50_stl10.pth"

NUM_CLASSES = 10  # STL-10

model = models.resnet50(pretrained=False)

model.fc = nn.Linear(
    model.fc.in_features,  # 2048
    NUM_CLASSES
)

state_dict = torch.load(RESNET_PTH_PATH, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("ResNet-50 loaded successfully.")

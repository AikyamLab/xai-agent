import torch
import torch.nn as nn
from torchvision import models
# import gdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DENSENET_PTH_PATH = "densenet201_stl10.pth"

NUM_CLASSES = 10  # STL-10

model = models.densenet201(pretrained=False)

model.classifier = nn.Linear(
    model.classifier.in_features,  # 1920
    NUM_CLASSES
)

state_dict = torch.load(DENSENET_PTH_PATH, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("DenseNet-201 loaded successfully.")

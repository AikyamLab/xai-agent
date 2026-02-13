import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CLASS_NUM = 200          # CUB-200
WEIGHTS_PATH = "densenet201_cub.pth"  # your .pth file

model = models.densenet201(weights=None)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc0', nn.Linear(1920, 256)),
    ('norm0', nn.BatchNorm1d(256)),
    ('relu0', nn.ReLU(inplace=True)),
    ('fc1', nn.Linear(256, CLASS_NUM))
]))

model.classifier = classifier
model = model.to(device)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)

model.eval()

print("DenseNet-201 CUB model loaded successfully.")
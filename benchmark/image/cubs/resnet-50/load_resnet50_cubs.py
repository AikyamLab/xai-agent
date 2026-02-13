import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CLASS_NUM = 200                    # CUB-200
WEIGHTS_PATH = "resnet50_cub.pth"  # your saved .pth

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
  
in_features = model.fc.in_features  # 2048

classifier = nn.Sequential(OrderedDict([
    ('fc0', nn.Linear(in_features, 256)),
    ('norm0', nn.BatchNorm1d(256)),
    ('relu0', nn.ReLU(inplace=True)),
    ('fc1', nn.Linear(256, CLASS_NUM))
]))

model.fc = classifier
model = model.to(device)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)

model.eval()

print("ResNet-50 CUB model loaded successfully.")

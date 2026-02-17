import torch
import sys

model_path = "models_to_read/vision/stl10_resnet.pth"

try:
    # Attempt to load the model
    # It's good practice to map to CPU if you're just checking integrity
    # without needing a GPU, to avoid potential CUDA issues on different systems.
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"Successfully loaded model from {model_path}. Model appears to be intact.")
    # You could add more checks here, e.g., model.eval() or check for specific keys if it's a state_dict
    # For now, just successful loading is enough for a basic integrity check.

except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    print("The model file might be corrupted or is not a valid PyTorch model file.")
    sys.exit(1)

import torch
import sys
# Add the current directory to the path to find DataModelLoader
sys.path.append('.')
from DataModelLoader import DataModelLoader, TabNN

def inspect_model(file_path):
    print(f"--- Inspecting: {file_path} ---")
    try:
        # We need to instantiate the DataModelLoader to use its methods
        # But for this inspection, we can just use torch.load directly
        loaded_object = torch.load(file_path, map_location='cpu')
        if isinstance(loaded_object, dict):
            print(f"Result: This is a state_dict (contains model weights only).")
            print(f"Sample Keys: {list(loaded_object.keys())[:5]}")
        elif isinstance(loaded_object, torch.nn.Module):
            print(f"Result: This is a full model (contains architecture and weights).")
        else:
            print(f"Result: Unknown format. Type is {type(loaded_object)}.")
    except Exception as e:
        print(f"Error loading file: {e}")
    print("-" * (len(file_path) + 16))


if __name__ == "__main__":
    model_files = [
        "models_to_read/resnet_stl10.pth",
        "models_to_read/tabular/tabnn_census.pth",
        "models_to_read/cnn_stl10 (1).pth"
    ]
    
    print("Inspecting model files...")
    for model_file in model_files:
        inspect_model(model_file)
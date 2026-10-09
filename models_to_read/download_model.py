# do pip install gdown
import gdown
import torch
import os
import sys
import re

# Ensure the parent directory is in the path to find DataModelLoader if needed later
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) 

def download_and_test_model(url, output_path, model_name_for_print):
    print(f"\n--- Attempting to download {model_name_for_print} ---")
    print(f"URL: {url}")
    print(f"Output path: {output_path}")
    
    # Extract file ID from URL for gdown
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if not match:
        print(f"Error: Could not extract Google Drive ID from URL: {url}")
        return False
    file_id = match.group(1)

    try:
        # gdown.download handles overwriting by default if output is specified
        gdown.download(id=file_id, output=output_path, quiet=False)
        print(f"--- Download of {model_name_for_print} successful! ---")

        print(f"--- Attempting to load {model_name_for_print} with torch.load ---")
        # Load the object from the .pth file
        loaded_object = torch.load(output_path, map_location='cpu')

        if isinstance(loaded_object, dict):
            print(f"Result: SUCCESS - This is a state_dict (contains model weights only).")
            print(f"Sample Keys: {list(loaded_object.keys())[:5]}")
        elif isinstance(loaded_object, torch.nn.Module):
            print(f"Result: SUCCESS - This is a full model (contains architecture and weights).")
        else:
            print(f"Result: SUCCESS - Unknown PyTorch object type: {type(loaded_object)}.")
        return True

    except Exception as e:
        print(f"--- FAILED to download or load {model_name_for_print} ---")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # List of models to download and test
    models_to_process = [
        {
            "url": "https://drive.google.com/file/d/1mAypR_7TAcAYQFoJKIGkJ_IZCXMlt6_q/view?usp=drive_link", # !!! REPLACE THIS WITH THE ACTUAL RESNET GOOGLE DRIVE LINK !!!
            "output": "models_to_read/text/snli_2layernn.pth",
            "name": "snli_2layernn.pth"
        }
    ]

    # Ensure the output directory exists
    output_base_dir = "models_to_read/vision/"
    os.makedirs(output_base_dir, exist_ok=True)

    for model_info in models_to_process:
        download_and_test_model(model_info["url"], model_info["output"], model_info["name"])

    print("\n--- Testing complete. ---")
    print("Please ensure you replaced the placeholder URL for resnet.pth with the correct one.")

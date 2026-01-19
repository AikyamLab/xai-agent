import torch
import torchvision.models as models
import torchvision.transforms as transforms

def load_model(model_path):
    """
    Loads the STL-10 ResNet model.

    Args:
        model_path (str): The path to the .pth model file.

    Returns:
        tuple: A tuple containing the loaded model and the image transform.
    """
    model = models.resnet50(pretrained=True)
    # Modify the final fully connected layer for STL-10 (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform
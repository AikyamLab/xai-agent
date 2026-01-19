import torch
import torch.nn as nn

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def load_model(model_path, input_dim=5000, hidden_dim=100, output_dim=2):
    """
    Loads the IMDB 2-layer NN model.

    Args:
        model_path (str): The path to the .pth model file.
        input_dim (int): Input dimension for the model.
        hidden_dim (int): Hidden dimension for the model.
        output_dim (int): Output dimension for the model.

    Returns:
        tuple: A tuple containing the loaded model and a processor (None).
    """
    model = TwoLayerNN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, None
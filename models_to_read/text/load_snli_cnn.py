import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def load_model(model_path, vocab_size=30000, embed_dim=300, n_filters=100, filter_sizes=[3,4,5], output_dim=3, dropout=0.5):
    """
    Loads the SNLI CNN model.

    Args:
        model_path (str): The path to the .pth model file.
        vocab_size (int): The vocabulary size.
        embed_dim (int): The dimension of the embeddings.
        n_filters (int): The number of filters.
        filter_sizes (list): The filter sizes.
        output_dim (int): The output dimension.
        dropout (float): The dropout rate.

    Returns:
        tuple: A tuple containing the loaded model and a processor (None).
    """
    model = SimpleCNN(vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, None
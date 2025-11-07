import torch
from torch import nn
from config import GNN_OUT_DIM
class LinkPredictor(nn.Module):
    """
    Simple MLP predictor.
    Takes concatenated node embeddings and predicts an edge logit.
    """
    def __init__(self, in_dim, hidden_dim=GNN_OUT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(x)

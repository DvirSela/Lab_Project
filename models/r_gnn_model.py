from torch import nn
from torch_geometric.nn import RGCNConv
from config import DROPOUT

class R_GNN_Model(nn.Module):
    """
    A 2-layer Relational-GCN (R-GCN) model.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=31)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=31)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type).relu()
        return x

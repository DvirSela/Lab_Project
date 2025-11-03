import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Dict
from utils.util import log

class GraphEdgeDataset(Dataset):
    """
    Creates a dataset of (anchor, positive neighbor) pairs based on edges.
    """
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        super().__init__()
        
        # Assumes 'node_id' is the unique integer ID
        # Create a map from node_id -> node_data
        if 'node_id' not in nodes_df.columns:
            raise ValueError("nodes_df must have a 'node_id' column.")
        
        self.nodes_map = nodes_df.set_index('node_id').to_dict(orient='index')
        
        # Create a list of (src_id, dst_id) edges
        if 'src_id' not in edges_df.columns or 'dst_id' not in edges_df.columns:
             raise ValueError("edges_df must have 'src_id' and 'dst_id' columns.")
        
        self.edges = edges_df[['src_id', 'dst_id']].values.tolist()
        
        # Filter edges where src or dst node is not in the nodes_map
        self.edges = [
            (s, d) for s, d in self.edges 
            if s in self.nodes_map and d in self.nodes_map
        ]
        log(f"Initialized GraphEdgeDataset with {len(self.nodes_map)} nodes and {len(self.edges)} valid edges.")

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        src_id, dst_id = self.edges[idx]
        anchor_data = self.nodes_map[src_id]
        positive_data = self.nodes_map[dst_id]
        return anchor_data, positive_data
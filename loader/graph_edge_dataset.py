import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Dict
from utils.util import log

class GraphEdgeDataset(Dataset):
    """
    Creates a dataset of (anchor, positive neighbor) pairs based on edges.
    Optimized to avoid creating full dict copies in memory.
    """
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        super().__init__()
        
        # Assumes 'node_id' is the unique integer ID
        if 'node_id' not in nodes_df.columns:
            raise ValueError("nodes_df must have a 'node_id' column.")
        
        # Store nodes_df directly with indexed access (more memory efficient)
        self.nodes_df = nodes_df.set_index('node_id')
        
        # Create a list of (src_id, dst_id) edges
        if 'src_id' not in edges_df.columns or 'dst_id' not in edges_df.columns:
             raise ValueError("edges_df must have 'src_id' and 'dst_id' columns.")
        
        # Filter edges where src or dst node is not in the nodes_df
        valid_node_ids = set(self.nodes_df.index)
        self.edges = [
            (s, d) for s, d in edges_df[['src_id', 'dst_id']].values.tolist()
            if s in valid_node_ids and d in valid_node_ids
        ]
        log(f"Initialized GraphEdgeDataset with {len(self.nodes_df)} nodes and {len(self.edges)} valid edges.")

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        src_id, dst_id = self.edges[idx]
        # Use .loc[] for efficient indexed access and convert to dict on-the-fly
        anchor_data = self.nodes_df.loc[src_id].to_dict()
        positive_data = self.nodes_df.loc[dst_id].to_dict()
        return anchor_data, positive_data
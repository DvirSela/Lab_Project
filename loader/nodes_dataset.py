import pandas as pd
from torch.utils.data import Dataset
class NodesDataset(Dataset):
    def __init__(self, nodes_df: pd.DataFrame):
        self.rows = nodes_df.to_dict(orient='records')
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

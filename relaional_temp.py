import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel, CLIPTokenizer
from PIL import Image, ImageFile
import pandas as pd
from tqdm.auto import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, RGCNConv, to_hetero
import numpy as np
# --- PyG Models ---
from torch_geometric.nn import SAGEConv
from config import DEVICE, CLIP_MODEL_NAME, FUSED_DIM, PROJ_DIM, BATCH_SIZE, SEED
from utils.util import log, relation_to_meta
from models.multimodal_graph_encoder import MultimodalGraphEncoder
from loader.nodes_dataset import NodesDataset
from loader.inference_collator import InferenceCollator
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------
# ========== CONFIG ========
# -------------------------

# --- Paths ---
# Path to the model saved by 'pretrain_end_to_end.py'
PRETRAINED_CHECKPOINT = Path("checkpoints") / "multimodal_encoder_final.pt"
# Path to cache the generated node features
FEATURE_CACHE_DIR = Path("cache_node_features")
FEATURE_CACHE_DIR.mkdir(exist_ok=True, parents=True)

# --- GNN Config ---
GNN_HIDDEN_DIM = 256
GNN_OUT_DIM = 128
GNN_LAYERS = 2
GNN_EPOCHS = 100
GNN_LR = 1e-3

# --- Eval Config ---
EVAL_NEG_SAMPLES = 100 # How many negatives to rank against for Hits@K/MRR

torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class R_GNN_Model(nn.Module):
    """
    A 2-layer Relational-GCN (R-GCN) model.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        # num_bases=30 is a good default to control parameters. Adjust if needed.
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=30)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=30)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_type):
        # Now pass edge_type to the conv layers
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type).relu()
        return x

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

def generate_features_all_nodes(
    nodes_df: pd.DataFrame, 
    checkpoint_path: Path,
    modes: List[str] = ['fused', 'text_only', 'image_only', 'concat']
) -> Dict[str, torch.Tensor]:
    """
    Runs all nodes through the (pretrained) encoder to get static features.
    Generates features for our 'fused' method AND all baselines at once.
    """
    
    # Check cache first
    cached_features = {}
    all_cached = True
    for mode in modes:
        cache_file = FEATURE_CACHE_DIR / f"features_{mode}.pt"
        if cache_file.exists():
            log(f"Loading cached features for mode: {mode}")
            cached_features[mode] = torch.load(cache_file, map_location='cpu')
        else:
            all_cached = False
            
    if all_cached:
        return cached_features

    log("Cache not found for all modes. Generating features...")
    
    # 1. Load pretrained model
    model = MultimodalGraphEncoder(CLIP_MODEL_NAME, FUSED_DIM, PROJ_DIM).to(DEVICE)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. "
                              "Please run pretrain_end_to_end.py first.")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    log("Loaded pretrained encoder checkpoint.")

    # 2. Load tokenizer/processor
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)

    # 3. Create dataloader for all nodes
    dataset = NodesDataset(nodes_df)
    collator = InferenceCollator(tokenizer, processor)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collator, num_workers=4)

    # 4. Run inference
    outputs = {mode: [] for mode in modes}
    for batch in tqdm(loader, desc="Extracting node features"):
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        # Use our custom .encode() method
        batch_features = model.encode(**inputs)
        
        for mode in modes:
            outputs[mode].append(batch_features[mode].cpu())

    # 5. Concatenate and cache
    final_features = {}
    for mode in modes:
        final_features[mode] = torch.cat(outputs[mode], dim=0)
        cache_file = FEATURE_CACHE_DIR / f"features_{mode}.pt"
        torch.save(final_features[mode], cache_file)
        log(f"Saved {mode} features to {cache_file} (Shape: {final_features[mode].shape})")

    return final_features

# ----------------------------------------------------------------------
# ===== STEP 4: GNN TRAINING & EVALUATION FUNCTIONS ======
# ----------------------------------------------------------------------

def train_gnn(
    gnn_model: R_GNN_Model, 
    link_predictor: LinkPredictor, 
    data: Data, 
    optimizer: torch.optim.Optimizer
):
    gnn_model.train()
    link_predictor.train()
    
    # We use LinkNeighborLoader for efficient negative sampling
    loader = LinkNeighborLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        neg_sampling_ratio=1.0, # 1 negative sample per positive sample
        num_neighbors=[10, 5], # 2-hop neighborhood
    )
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        
        # GNN provides node embeddings
        z = gnn_model(batch.x, batch.edge_index, batch.edge_type)
        
        # Get embeddings for positive edges
        z_src_pos = z[batch.edge_label_index[0]]
        z_dst_pos = z[batch.edge_label_index[1]]
        
        # Predict logits
        logits = link_predictor(z_src_pos, z_dst_pos).squeeze()
        
        # Calculate loss
        loss = criterion(logits, batch.edge_label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def test_gnn(
    gnn_model: R_GNN_Model, 
    link_predictor: LinkPredictor, 
    data: Data
):
    print("Starting evaluation...")
    """
    Evaluate the model on Hits@10 and MRR.
    """
    gnn_model.eval()
    link_predictor.eval()
    
    # Get final embeddings for ALL nodes using the training graph
    # This is important: we use the message-passing graph from training
    # data = data.to(DEVICE)
    # z = gnn_model(data.x, data.edge_index, data.edge_type)
    gnn_model = gnn_model.to(DEVICE)
    link_predictor = link_predictor.to(DEVICE)
    # Compute z in batches.
    # We pass 'data' (which is val_data or test_data).
    # The loader will correctly use its .x, .edge_index, and .edge_type
    z = get_all_embeddings(gnn_model, data, DEVICE)
    # z is now a complete [num_nodes, GNN_OUT_DIM] tensor on the DEVICE
    # --- END FIX ---
    hits10_ranks = []
    mrr_ranks = []
    
    # We evaluate on the 'val_pos_edge_index' or 'test_pos_edge_index'
    # which contains the positive edges we need to rank.
    test_edges = data.edge_label_index
    num_test_edges = test_edges.shape[1]

    for i in tqdm(range(num_test_edges), desc="Evaluating links"):
        # Get one positive edge
        src = test_edges[0, i]
        pos_dst = test_edges[1, i]

        # Get embeddings
        z_src = z[src].unsqueeze(0)
        z_pos_dst = z[pos_dst].unsqueeze(0)
        
        # Sample negative destinations
        neg_dst = torch.randint(0, data.num_nodes, (EVAL_NEG_SAMPLES,), device=DEVICE)
        z_neg_dst = z[neg_dst]
        
        # Create embedding pairs
        z_src_rpt = z_src.repeat(EVAL_NEG_SAMPLES, 1) # [N_neg, D]
        
        # Calculate scores
        pos_score = link_predictor(z_src, z_pos_dst)
        neg_scores = link_predictor(z_src_rpt, z_neg_dst)
        
        # --- Calculate Rank ---
        # Combine positive score with all negative scores
        all_scores = torch.cat([pos_score.squeeze(1), neg_scores.squeeze(1)])        
        # Get the rank of the positive score (index 0)
        # We sort in descending order
        sorted_indices = all_scores.argsort(descending=True)
        rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
        
        mrr_ranks.append(1.0 / rank)
        if rank <= 10:
            hits10_ranks.append(1.0)
        else:
            hits10_ranks.append(0.0)

    mrr = torch.tensor(mrr_ranks).mean().item()
    hits10 = torch.tensor(hits10_ranks).mean().item()
    
    return {"MRR": mrr, "Hits@10": hits10}

from torch_geometric.loader import NeighborLoader # Add this import at the top

@torch.no_grad()
def get_all_embeddings(
    gnn_model: R_GNN_Model, 
    data: Data, 
    device: torch.device
) -> torch.Tensor:
    """
    Performs batched inference to get all node embeddings
    without causing an OOM error.
    """
    gnn_model.eval()
    
    # We need a loader to compute embeddings for all nodes
    loader = NeighborLoader(
        data,
        num_neighbors=[-1] * GNN_LAYERS, # Use all neighbors (full-batch message passing)
        batch_size=BATCH_SIZE * 4, # Use a larger batch size for inference
        shuffle=False,
        num_workers=4,
    )
    
    # Tensor to store all final embeddings
    z_all = torch.zeros(data.num_nodes, GNN_OUT_DIM, device=device)
    
    log("Computing all node embeddings for evaluation...")
    for batch in tqdm(loader, desc="Batched GNN Inference"):
        batch = batch.to(device)
        
        # Run GNN on the sampled subgraph
        z_batch = gnn_model(batch.x, batch.edge_index, batch.edge_type)
        
        # Get the embeddings for the seed nodes (the first 'batch_size' nodes)
        z_seed = z_batch[:batch.batch_size]
        
        # Get the original IDs of these seed nodes
        original_ids = batch.n_id[:batch.batch_size]
        
        # Store them in the correct position in the full tensor
        z_all[original_ids] = z_seed
        
    return z_all
# ----------------------------------------------------------------------
# ===== STEP 5: MAIN EXECUTION SCRIPT ======
# ----------------------------------------------------------------------
if __name__ == "__main__":
    log(f"Starting run. Using device: {DEVICE}") # <-- ADD THIS
    # 1. Load Node and Edge Data
    try:
        nodes_df = pd.read_csv("./data/processed/nodes.csv")
        edges_df = pd.read_csv("./data/processed/edges.csv")
    except FileNotFoundError:
        log("Error: nodes.csv or edges.csv not found. Exiting.")
        exit()

    # Ensure nodes are indexed 0...N-1
    if 'node_id' not in nodes_df.columns or not (nodes_df['node_id'] == range(len(nodes_df))).all():
        log("Warning: 'node_id' column missing or not sequential. Resetting index.")
        nodes_df = nodes_df.reset_index(drop=True).rename(columns={'index': 'node_id'})
        # We'd also need to remap edges_df 'src_id' and 'dst_id' if they weren't sequential
        # For this script, we assume 'node_id', 'src_id', 'dst_id' are all 0-indexed and sequential
    
    num_nodes = len(nodes_df)
    log(f"Loaded {num_nodes} nodes and {len(edges_df)} edges.")

    # 2. Generate/Load All Feature Matrices
    baseline_modes = ['fused', 'text_only', 'image_only', 'concat']
    baseline_modes = ['concat']
    try:
        feature_matrices = generate_features_all_nodes(
            nodes_df,
            PRETRAINED_CHECKPOINT,
            modes=baseline_modes
        )
    except FileNotFoundError as e:
        log(str(e))
        exit()

    # 3. Create PyG Data and Split
    # Convert to a single numpy array first, then to tensor
    edge_index_np = np.array([
        edges_df['src_id'].values,
        edges_df['dst_id'].values
    ])
    edge_index = torch.from_numpy(edge_index_np).to(torch.long)
    
    # map relation IDs to meta-relations using relation_to_meta
    edges_df['meta_rel_name'] = edges_df['rel_id'].map(relation_to_meta)
    edges_df['meta_rel_id'] = edges_df['meta_rel_name'].astype('category').cat.codes
    mapping = dict(enumerate(edges_df['meta_rel_name'].astype('category').cat.categories))
    log(f"Meta-relation mapping: {mapping}")
    num_of_meta_relations = len(set(relation_to_meta.values()))
    if num_of_meta_relations > 31:
        log(f"Error: Number of meta-relations exceeds 31. Found: {num_of_meta_relations}")
        raise ValueError("Number of meta-relations exceeds 31.")
    log(f"Mapped original relations to {num_of_meta_relations} meta-relations.")
    # Load the relation IDs as edge_type
    edge_type = torch.tensor(edges_df['meta_rel_id'].values, dtype=torch.long)
    num_relations = edge_type.max().item() + 1
    log(f"Found {num_relations} unique relation types.")
    
    # We create a placeholder data object. We'll swap `data.x` in the loop.
    data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_type=edge_type)
    
    # Split edges into train/val/test
    # This transform adds:
    #   - data.train_pos_edge_index
    #   - data.val_pos_edge_index, data.val_neg_edge_index
    #   - data.test_pos_edge_index, data.test_neg_edge_index
    # It also *modifies* data.edge_index to *only* contain training edges
    # (to prevent message passing leakage)
    transform = T.RandomLinkSplit(
        num_val=0.1,  # 10% for validation
        num_test=0.1, # 10% for testing
        is_undirected=False, # Our graph is directional
        add_negative_train_samples=False, # LinkNeighborLoader handles this
    )
    train_data, val_data, test_data = transform(data)
    log(f"Split edges: {train_data.edge_index.shape[1]} train, "
        f"{val_data.edge_label_index.shape[1]} val, "
        f"{test_data.edge_label_index.shape[1]} test")

    # 4. Run Training & Eval Loop for each feature set
    final_results = {}
    
    for mode in baseline_modes:
        log("\n" + "="*50)
        log(f"🚀 STARTING RUN FOR: {mode.upper()} 🚀")
        log("="*50)
        
        # 4a. Set the correct feature matrix
        X = feature_matrices[mode].to(DEVICE)
        train_data.x = X
        val_data.x = X
        test_data.x = X
        
        in_dim = X.shape[1]
        
        # 4b. Initialize GNN and Predictor  
        gnn = R_GNN_Model(in_dim, GNN_HIDDEN_DIM, GNN_OUT_DIM, num_relations).to(DEVICE)
        predictor = LinkPredictor(GNN_OUT_DIM).to(DEVICE)
        
        params = list(gnn.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=GNN_LR)

        best_val_mrr = 0
        best_test_metrics = {}

        # 4c. GNN Training Loop
        for epoch in range(1, GNN_EPOCHS + 1):
            loss = train_gnn(gnn, predictor, train_data, optimizer)
            
            if epoch % 10 == 0:
                # 4d. Validation
                val_metrics = test_gnn(gnn, predictor, val_data)
                log(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Val MRR: {val_metrics['MRR']:.4f} | "
                    f"Val Hits@10: {val_metrics['Hits@10']:.4f}")

                if val_metrics['MRR'] > best_val_mrr:
                    best_val_mrr = val_metrics['MRR']
                    log(f"  -> New best val MRR! Testing...")
                    # 4e. Test on best validation
                    best_test_metrics = test_gnn(gnn, predictor, test_data)
                    log(f"  -> Test MRR: {best_test_metrics['MRR']:.4f} | "
                        f"Test Hits@10: {best_test_metrics['Hits@10']:.4f}")

        final_results[mode] = best_test_metrics
        log(f"Finished run for {mode}. Clearing memory...")
        del gnn, predictor, optimizer, X
        torch.cuda.empty_cache()

    # 5. Print Final Comparison Table
    log("\n" + "="*50)
    log("🏁 FINAL RESULTS 🏁")
    log("="*50)
    log(f"{'Method':<15} | {'Test MRR':<10} | {'Test Hits@10':<12}")
    log("-"*50)
    
    for mode, metrics in final_results.items():
        log(f"{mode:<15} | {metrics.get('MRR', 0):<10.4f} | {metrics.get('Hits@10', 0):<12.4f}")
    
    log("="*50)
    log("Run complete.")
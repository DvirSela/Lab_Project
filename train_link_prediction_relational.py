import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPTokenizer
from PIL import ImageFile
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from config import DEVICE, CLIP_MODEL_NAME, BATCH_SIZE, SEED, PRETRAINED_CHECKPOINT, FEATURE_CACHE_DIR, \
    GNN_HIDDEN_DIM, GNN_OUT_DIM, GNN_EPOCHS, GNN_LR
from utils.util import log, relation_to_meta, load_pretrained_encoder
from models.link_predictor import LinkPredictor
from models.r_gnn_model import R_GNN_Model
from models.multimodal_graph_encoder import MultimodalGraphEncoder
from loader.inference_collator import InferenceCollator
from loader.nodes_dataset import NodesDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


FEATURE_CACHE_DIR.mkdir(exist_ok=True, parents=True)

EVAL_NEG_SAMPLES = 100  # How many negatives to rank against for eval

torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def generate_features_all_nodes(
    nodes_df: pd.DataFrame,
    checkpoint_path: Path,
    modes: List[str] = ['fused', 'text_only', 'image_only', 'concat']
) -> Dict[str, torch.Tensor]:
    """
    Runs all nodes through the (pretrained) encoder to get static features.
    Generates features for our 'fused' method AND all baselines at once.
    Caches features to disk for faster reloads.
    Args:
        nodes_df: DataFrame containing all nodes.
        checkpoint_path: Path to the pretrained encoder checkpoint.
        modes: List of feature modes to generate.
    Returns:
        A dict mapping mode names to feature tensors.
    """
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

    model = load_pretrained_encoder(checkpoint_path, DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(
        CLIP_MODEL_NAME, use_safetensors=True, use_fast=True)
    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME, use_safetensors=True, use_fast=True)

    dataset = NodesDataset(nodes_df)
    collator = InferenceCollator(tokenizer, processor)
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        collate_fn=collator, num_workers=4)

    outputs = {mode: [] for mode in modes}
    for batch in tqdm(loader, desc="Extracting node features"):
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        batch_features = model.encode(**inputs)

        for mode in modes:
            outputs[mode].append(batch_features[mode].cpu())

    final_features = {}
    for mode in modes:
        final_features[mode] = torch.cat(outputs[mode], dim=0)
        cache_file = FEATURE_CACHE_DIR / f"features_{mode}.pt"
        torch.save(final_features[mode], cache_file)
        log(
            f"Saved {mode} features to {cache_file} (Shape: {final_features[mode].shape})")

    return final_features


def train_gnn(
    gnn_model: R_GNN_Model,
    link_predictor: LinkPredictor,
    data: Data,
    optimizer: torch.optim.Optimizer
):
    """
    Single epoch training for the GNN + Link Predictor.
    Uses LinkNeighborLoader for efficient negative sampling.
    Args:
        gnn_model: The R-GCN model.
        link_predictor: The link predictor MLP.
        data: The training graph data.
        optimizer: The optimizer for training.
    Returns:
        Average training loss for the epoch.
    """
    gnn_model.train()
    link_predictor.train()

    # We use LinkNeighborLoader for efficient negative sampling
    loader = LinkNeighborLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        neg_sampling_ratio=1.0,  # 1 negative sample per positive sample
        num_neighbors=[10, 5], 
    )

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        z = gnn_model(batch.x, batch.edge_index, batch.edge_type)
        z_src_pos = z[batch.edge_label_index[0]]
        z_dst_pos = z[batch.edge_label_index[1]]

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
    Evaluate the model on Hits@10 and MRR metrics.
    For each positive edge, samples negative edges and computes ranks.
    Args:
        gnn_model: The trained R-GCN model.
        link_predictor: The trained link predictor MLP.
        data: The evaluation graph data (val or test).
    Returns:
        A dict with 'MRR' and 'Hits@10' scores.
    """
    gnn_model.eval()
    link_predictor.eval()

    data = data.to(DEVICE)
    z = gnn_model(data.x, data.edge_index, data.edge_type)

    hits10_ranks = []
    mrr_ranks = []


    test_edges = data.edge_label_index
    num_test_edges = test_edges.shape[1]

    for i in tqdm(range(num_test_edges), desc="Evaluating links"):
        src = test_edges[0, i]
        pos_dst = test_edges[1, i]

        z_src = z[src].unsqueeze(0)
        z_pos_dst = z[pos_dst].unsqueeze(0)

        neg_dst = torch.randint(
            0, data.num_nodes, (EVAL_NEG_SAMPLES,), device=DEVICE)
        z_neg_dst = z[neg_dst]

        z_src_rpt = z_src.repeat(EVAL_NEG_SAMPLES, 1)  # [N_neg, D]

        pos_score = link_predictor(z_src, z_pos_dst)
        neg_scores = link_predictor(z_src_rpt, z_neg_dst)

        all_scores = torch.cat([pos_score.squeeze(1), neg_scores.squeeze(1)])

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


def load_data(nodes_path: Path, edges_path: Path):
    """
    Loads nodes and edges from CSV files.
    Ensures node IDs are sequential.
    Args:
        nodes_path: Path to nodes CSV.
        edges_path: Path to edges CSV.
    Returns:
        nodes_df: DataFrame of nodes.
        edges_df: DataFrame of edges.
    """
    try:
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
    except FileNotFoundError:
        log("Error: nodes.csv or edges.csv not found. Exiting.")
        exit()
    if 'node_id' not in nodes_df.columns or not (nodes_df['node_id'] == range(len(nodes_df))).all():
        log("Warning: 'node_id' column missing or not sequential. Resetting index.")
        nodes_df = nodes_df.reset_index(drop=True).rename(
            columns={'index': 'node_id'})
    return nodes_df, edges_df


if __name__ == "__main__":
    log(f'starting link prediction training on device: {DEVICE}')
    nodes_df, edges_df = load_data(
        Path("./data/processed/nodes.csv"),
        Path("./data/processed/edges.csv")
    )
    num_nodes = len(nodes_df)
    log(f"Loaded {num_nodes} nodes and {len(edges_df)} edges.")

    baseline_modes = ['fused', 'text_only', 'image_only', 'concat']
    try:
        feature_matrices = generate_features_all_nodes(
            nodes_df,
            PRETRAINED_CHECKPOINT,
            modes=baseline_modes
        )
    except FileNotFoundError as e:
        log(str(e))
        exit()

    edge_index_np = np.array([
        edges_df['src_id'].values,
        edges_df['dst_id'].values
    ])
    edge_index = torch.from_numpy(edge_index_np).to(torch.long)

    edges_df['meta_rel_name'] = edges_df['rel_id'].map(relation_to_meta)
    edges_df['meta_rel_id'] = edges_df['meta_rel_name'].astype(
        'category').cat.codes
    mapping = dict(
        enumerate(edges_df['meta_rel_name'].astype('category').cat.categories))
    log(f"Meta-relation mapping: {mapping}")
    num_of_meta_relations = len(set(relation_to_meta.values()))
    if num_of_meta_relations > 31:
        log(
            f"Error: Number of meta-relations exceeds 31. Found: {num_of_meta_relations}")
        raise ValueError("Number of meta-relations exceeds 31.")
    log(f"Mapped original relations to {num_of_meta_relations} meta-relations.")
    edge_type = torch.tensor(edges_df['meta_rel_id'].values, dtype=torch.long)
    num_relations = edge_type.max().item() + 1
    log(f"Found {num_relations} unique relation types.")
    data = Data(num_nodes=num_nodes,
                edge_index=edge_index, edge_type=edge_type)

    # Split edges into train/val/test
    # This transform adds:
    #   - data.train_pos_edge_index
    #   - data.val_pos_edge_index, data.val_neg_edge_index
    #   - data.test_pos_edge_index, data.test_neg_edge_index
    # It also *modifies* data.edge_index to *only* contain training edges
    # (to prevent message passing leakage)
    transform = T.RandomLinkSplit(
        num_val=0.1, 
        num_test=0.1, 
        is_undirected=False, 
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(data)
    log(f"Split edges: {train_data.edge_index.shape[1]} train, "
        f"{val_data.edge_label_index.shape[1]} val, "
        f"{test_data.edge_label_index.shape[1]} test")

    final_results = {}

    for mode in baseline_modes:
        log("\n" + "="*50)
        log(f"----------- STARTING RUN FOR: {mode.upper()} -----------")
        log("="*50)

        X = feature_matrices[mode].to(DEVICE)
        train_data.x = X
        val_data.x = X
        test_data.x = X

        in_dim = X.shape[1]

        gnn = R_GNN_Model(in_dim, GNN_HIDDEN_DIM,
                          GNN_OUT_DIM, num_relations).to(DEVICE)
        predictor = LinkPredictor(GNN_OUT_DIM).to(DEVICE)

        params = list(gnn.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=GNN_LR)

        best_val_mrr = 0
        best_test_metrics = {}

        for epoch in range(1, GNN_EPOCHS + 1):
            loss = train_gnn(gnn, predictor, train_data, optimizer)

            if epoch % 5 == 0:
                val_metrics = test_gnn(gnn, predictor, val_data)
                log(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Val MRR: {val_metrics['MRR']:.4f} | "
                    f"Val Hits@10: {val_metrics['Hits@10']:.4f}")

                if val_metrics['MRR'] > best_val_mrr:
                    best_val_mrr = val_metrics['MRR']
                    log(f"  -> New best val MRR! Testing...")
                    best_test_metrics = test_gnn(gnn, predictor, test_data)
                    log(f"  -> Test MRR: {best_test_metrics['MRR']:.4f} | "
                        f"Test Hits@10: {best_test_metrics['Hits@10']:.4f}")

        final_results[mode] = best_test_metrics
        log(f"Finished run for {mode}. Clearing memory...")
        del gnn, predictor, optimizer, X
        torch.cuda.empty_cache()

    log("\n" + "="*50)
    log("----------- FINAL RESULTS -----------")
    log("="*50)
    log(f"{'Method':<15} | {'Test MRR':<10} | {'Test Hits@10':<12}")
    log("-"*50)

    for mode, metrics in final_results.items():
        log(f"{mode:<15} | {metrics.get('MRR', 0):<10.4f} | {metrics.get('Hits@10', 0):<12.4f}")

    log("="*50)
    log("Run complete.")

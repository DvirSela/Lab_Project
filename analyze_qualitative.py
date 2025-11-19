import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch_geometric.transforms as T
from torch_geometric.data import Data

from config import DEVICE, SEED, FEATURE_CACHE_DIR, GNN_HIDDEN_DIM, GNN_OUT_DIM
from models.r_gnn_model import R_GNN_Model
from models.link_predictor import LinkPredictor
from utils.util import log, relation_to_meta
from config import DEVICE, SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

FUSED_MODEL_PATH = Path("./saved_models/link_prediction/gnn_fused_best.pt")
FUSED_PRED_PATH = Path(
    "./saved_models/link_prediction/predictor_fused_best.pt")
TEXT_MODEL_PATH = Path("./saved_models/link_prediction/gnn_text_only_best.pt")
TEXT_PRED_PATH = Path(
    "./saved_models/link_prediction/predictor_text_only_best.pt")

NODES_PATH = Path("./data/processed/nodes.csv")
EDGES_PATH = Path("./data/processed/edges.csv")

EVAL_NEG_SAMPLES = 100


def get_edge_ranks(model, predictor, data, feature_matrix):
    """
    Calculates rank of positive edges against N negatives.
    Args:
        model: Trained GNN model
        predictor: Trained link predictor
        data: PyG Data object containing test edges
        feature_matrix: Node feature matrix
    Returns:
        ranks: Array of ranks
        valid_indices: The original indices of these edges in the test_data.edge_label_index
    """
    model.eval()
    predictor.eval()

    data.x = feature_matrix.to(DEVICE)
    data = data.to(DEVICE)

    with torch.no_grad():
        z = model(data.x, data.edge_index, data.edge_type)

    ranks = []

    mask = data.edge_label == 1
    positive_test_edges = data.edge_label_index[:, mask]
    num_test_edges = positive_test_edges.shape[1]

    g_cpu = torch.Generator()
    g_cpu.manual_seed(SEED)

    valid_indices = torch.nonzero(mask).squeeze().cpu().numpy()

    for i in tqdm(range(num_test_edges), desc="Ranking positive edges"):
        src = positive_test_edges[0, i]
        pos_dst = positive_test_edges[1, i]

        z_src = z[src].unsqueeze(0)
        z_pos_dst = z[pos_dst].unsqueeze(0)

        # Sample negatives
        neg_dst = torch.randint(
            0, data.num_nodes, (EVAL_NEG_SAMPLES,), generator=g_cpu).to(DEVICE)
        z_neg_dst = z[neg_dst]

        z_src_rpt = z_src.repeat(EVAL_NEG_SAMPLES, 1)

        with torch.no_grad():
            pos_score = predictor(z_src, z_pos_dst)
            neg_scores = predictor(z_src_rpt, z_neg_dst)

        all_scores = torch.cat([pos_score.squeeze(1), neg_scores.squeeze(1)])
        sorted_indices = all_scores.argsort(descending=True)

        rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    return np.array(ranks), valid_indices


def load_graph_data():
    """Loads and aligns graph data from CSV files."""
    log("Loading and aligning graph data...")
    nodes_df = pd.read_csv(NODES_PATH)
    edges_df = pd.read_csv(EDGES_PATH)
    id_to_index_map = {node_id: i for i,
                       node_id in enumerate(nodes_df['node_id'].values)}

    try:
        src_tensor_idx = [id_to_index_map[x]
                          for x in edges_df['src_id'].values]
        dst_tensor_idx = [id_to_index_map[x]
                          for x in edges_df['dst_id'].values]
    except KeyError as e:
        log(
            f"CRITICAL ERROR: Edge list contains node ID {e} which is not in nodes.csv!")
        exit()

    edge_index_np = np.array([src_tensor_idx, dst_tensor_idx])
    edge_index = torch.from_numpy(edge_index_np).to(torch.long)

    nodes_df['tensor_index'] = list(range(len(nodes_df)))
    node_map = nodes_df.set_index('tensor_index').to_dict('index')

    edges_df['meta_rel_name'] = edges_df['rel_id'].map(relation_to_meta)
    edges_df['meta_rel_id'] = edges_df['meta_rel_name'].astype(
        'category').cat.codes
    edge_type = torch.tensor(edges_df['meta_rel_id'].values, dtype=torch.long)

    num_relations = edge_type.max().item() + 1
    num_nodes = len(nodes_df)

    data = Data(num_nodes=num_nodes,
                edge_index=edge_index, edge_type=edge_type)

    transform = T.RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=False, add_negative_train_samples=False
    )
    _, _, test_data = transform(data)

    return test_data, node_map, num_relations, num_nodes


def load_trained_model(gnn_path, pred_path, input_dim, num_relations):
    gnn = R_GNN_Model(input_dim, GNN_HIDDEN_DIM,
                      GNN_OUT_DIM, num_relations).to(DEVICE)
    pred = LinkPredictor(GNN_OUT_DIM).to(DEVICE)
    gnn.load_state_dict(torch.load(gnn_path, map_location=DEVICE))
    pred.load_state_dict(torch.load(pred_path, map_location=DEVICE))
    return gnn, pred


if __name__ == "__main__":
    log("--- Starting Qualitative Analysis (ALIGNED) ---")

    test_data, node_map, num_relations, num_nodes = load_graph_data()

    log("Loading features...")
    feats_fused = torch.load(
        FEATURE_CACHE_DIR / "features_fused.pt", map_location='cpu')
    feats_text = torch.load(
        FEATURE_CACHE_DIR / "features_text_only.pt", map_location='cpu')

    log("Loading models...")
    gnn_fused, pred_fused = load_trained_model(
        FUSED_MODEL_PATH, FUSED_PRED_PATH, feats_fused.shape[1], num_relations)
    gnn_text, pred_text = load_trained_model(
        TEXT_MODEL_PATH, TEXT_PRED_PATH, feats_text.shape[1], num_relations)

    log("Calculating Fused Ranks...")
    ranks_fused, valid_indices = get_edge_ranks(
        gnn_fused, pred_fused, test_data, feats_fused)

    log("Calculating Text-Only Ranks...")
    ranks_text, _ = get_edge_ranks(gnn_text, pred_text, test_data, feats_text)

    diffs = ranks_text - ranks_fused
    interesting_indices = np.where((ranks_fused <= 10) & (ranks_text > 40))[0]

    sorted_interesting = sorted(
        interesting_indices, key=lambda i: diffs[i], reverse=True)

    log(f"\nFound {len(sorted_interesting)} valid positive edges where Fused >>>> Text-only.")

    print("\n" + "="*80)
    print(f"{'Src Name':<25} | {'Dst Name':<25} | {'Fused Rank':<10} | {'Text Rank':<10}")
    print("="*80)
    edges_df = pd.read_csv(EDGES_PATH)
    edges_df['meta_rel_name'] = edges_df['rel_id'].map(relation_to_meta)
    for idx in sorted_interesting[:5]:
        original_idx = valid_indices[idx]

        src_id = test_data.edge_label_index[0, original_idx].item()
        dst_id = test_data.edge_label_index[1, original_idx].item()

        src_info = node_map[src_id]
        dst_info = node_map[dst_id]

        src_name = src_info['short_name']
        dst_name = dst_info['short_name']

        mask_edge = (edges_df['src_id'] == src_id) & (
            edges_df['dst_id'] == dst_id)
        if mask_edge.any():
            rel_name = edges_df.loc[mask_edge, 'meta_rel_name'].values[0]
        else:
            rel_name = "Unknown"

        print(
            f"{src_name[:25]:<25} | {dst_name[:25]:<25} | {ranks_fused[idx]:<10} | {ranks_text[idx]:<10}")
        print("-" * 80)
        print(f"   [Relation]: {rel_name}")
        print(f"   [Src Summary]: {str(src_info.get('summary', ''))[:100]}...")
        print(f"   [Dst Summary]: {str(dst_info.get('summary', ''))[:100]}...")

        img_path = Path(str(src_info.get('image_path', '')))
        img_status = "(Exists)" if img_path.exists() else "(MISSING!)"
        print(f"   [Src Image]: {img_path} {img_status}")
        print("-" * 80)
        print("\n")

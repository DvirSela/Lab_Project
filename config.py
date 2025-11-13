import torch
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {DEVICE}")

# Paths
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_SAVE_PATH = SAVE_DIR / "multimodal_encoder_final.pt"
PRETRAINED_CHECKPOINT = Path("checkpoints") / "multimodal_encoder_final.pt"
FEATURE_CACHE_DIR = Path("cache_node_features")

# Model / feature dims
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FUSED_DIM = 512
PROJ_DIM = 256

# Training hyperparams
# Total batch size will be 2 * BATCH_SIZE (anchors + positives)
BATCH_SIZE = 64
EPOCHS = 20
LR = 5e-5
TEMPERATURE = 0.07
GRAPH_LOSS_WEIGHT = 1.0  # Weight for L_graph relative to L_mm
SEED = 42

# GNN Config
GNN_HIDDEN_DIM = 256
GNN_OUT_DIM = 128
GNN_LAYERS = 2
GNN_EPOCHS = 10
GNN_LR = 1e-3
DROPOUT = 0.5

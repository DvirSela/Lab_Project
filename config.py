import torch
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_SAVE_PATH = SAVE_DIR / "multimodal_encoder_final.pt"

# Model / feature dims
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FUSED_DIM = 51
PROJ_DIM = 256 

# Training hyperparams
BATCH_SIZE = 64      # Total batch size will be 2 * BATCH_SIZE (anchors + positives)
EPOCHS = 20
LR = 5e-5
TEMPERATURE = 0.07
GRAPH_LOSS_WEIGHT = 1.0  # Weight for L_graph relative to L_mm
SEED = 42
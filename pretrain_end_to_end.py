import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPTokenizer
from PIL import ImageFile
import pandas as pd
from tqdm.auto import tqdm

from config import DEVICE, SAVE_DIR, MODEL_SAVE_PATH, CLIP_MODEL_NAME, FUSED_DIM, PROJ_DIM, BATCH_SIZE, EPOCHS, LR, TEMPERATURE, GRAPH_LOSS_WEIGHT, SEED
from models.multimodal_graph_encoder import MultimodalGraphEncoder
from loader.graph_edge_dataset import GraphEdgeDataset
from loader.multimodal_collator import MultimodalCollator
from utils.util import log, info_nce_loss

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def train_model(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """
    Main training function. Trains the MultimodalGraphEncoder end-to-end.
    Arguments:
        nodes_df: DataFrame with node data.
        edges_df: DataFrame with edge data.
    """
    log(f"Starting end-to-end pretraining on {DEVICE}")
    log(f"Params: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}, TEMP={TEMPERATURE}, GRAPH_WEIGHT={GRAPH_LOSS_WEIGHT}")

    # Initializing the Model
    model = MultimodalGraphEncoder(
        clip_name=CLIP_MODEL_NAME,
        fused_dim=FUSED_DIM,
        proj_dim=PROJ_DIM
    ).to(DEVICE)
    model.train()

    # Tokenizer and Processor (for collator).
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)

    # Initialize Dataloader
    dataset = GraphEdgeDataset(nodes_df, edges_df)
    collator = MultimodalCollator(tokenizer, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(), 
        lr=LR, 
        weight_decay=1e-4
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training Loop
    total_steps = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_loss_mm = 0.0
        epoch_loss_graph = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Forward pass - returns (2*B, PROJ_DIM) tensors
                z_graph, z_text, z_vis = model(**inputs)
                
                # --- 1. Multimodal Loss (L_mm) ---
                # Contrast all 2*B text vectors with all 2*B image vectors
                logits_mm = torch.matmul(z_text, z_vis.t()) / TEMPERATURE
                loss_mm = info_nce_loss(logits_mm)
                
                # --- 2. Graph Loss (L_graph) ---
                # Split the (2*B, D) graph tensor into (B, D) for anchors and (B, D) for positives
                z_graph_A, z_graph_P = z_graph.chunk(2, dim=0)
                
                # Contrast B anchors with B positives
                logits_graph = torch.matmul(z_graph_A, z_graph_P.t()) / TEMPERATURE
                loss_graph = info_nce_loss(logits_graph)
                
                # --- 3. Total Loss ---
                loss = loss_mm + GRAPH_LOSS_WEIGHT * loss_graph
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_loss_mm += loss_mm.item()
            epoch_loss_graph += loss_graph.item()
            total_steps += 1
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                L_mm=f"{loss_mm.item():.4f}", 
                L_graph=f"{loss_graph.item():.4f}"
            )
    
        avg_loss = epoch_loss / len(dataloader)
        avg_loss_mm = epoch_loss_mm / len(dataloader)
        avg_loss_graph = epoch_loss_graph / len(dataloader)
        log(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f} "
            f"[L_mm: {avg_loss_mm:.5f}, L_graph: {avg_loss_graph:.5f}]")
        
        # Save a checkpoint
        epoch_save_path = SAVE_DIR / f"multimodal_encoder_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, epoch_save_path)
        log(f"Saved checkpoint to {epoch_save_path}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    log(f"Training complete. Final model saved to {MODEL_SAVE_PATH}")


def load_data(nodes_path, edges_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    loads the datasets from CSV files and performs basic checks.
    Arguments:
        nodes_path: Path to nodes CSV.
        edges_path: Path to edges CSV.
    Returns:
        nodes_df, edges_df
    """
    try:
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
    except FileNotFoundError:
        raise FileNotFoundError("Please ensure 'nodes.csv' and 'edges.csv' are present in the working directory.")
    if not all(c in nodes_df.columns for c in ['node_id', 'summary', 'short_name', 'image_path']):
        raise ValueError("nodes_df is missing one of 'node_id', 'summary', 'short_name', 'image_path'")
    if not all(c in edges_df.columns for c in ['src_id', 'dst_id']):
        raise ValueError("edges_df is missing 'src_id' or 'dst_id'")

    sample_img_path = nodes_df['image_path'].dropna().iloc[random.randint(0, len(nodes_df)-1)]
    image_exists = Path(str(sample_img_path).replace("\\", "/")).exists()
    if not image_exists:
        raise FileNotFoundError(f"Sample image path {sample_img_path} does not exist. Please check image paths in nodes.csv.")
    return nodes_df, edges_df
if __name__ == "__main__":
    nodes_df, edges_df = load_data('./data/processed/nodes.csv', './data/processed/edges.csv')
    if Path(MODEL_SAVE_PATH).exists():
        log(f"Model already trained and saved at {MODEL_SAVE_PATH}. To retrain, please delete the existing model file or rename it.")
    else:
        # train_model(nodes_df, edges_df)
        pass
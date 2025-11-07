import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPVisionModel
from models.cross_attention_fusion import CrossAttentionFusion

class MultimodalGraphEncoder(nn.Module):
    """
    The main end-to-end model.
    Encapsulates frozen CLIP models, the trainable fusion module,
    and trainable projection heads.
    """
    def __init__(self, clip_name: str, fused_dim: int, proj_dim: int):
        super().__init__()
        # log("Loading CLIP models...")
        # 1. Encoders
        self.text_model = CLIPTextModel.from_pretrained(clip_name, use_safetensors=True).eval()
        self.vision_model = CLIPVisionModel.from_pretrained(clip_name, use_safetensors=True).eval()
        # Freeze CLIP parameters
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        d_text = self.text_model.config.hidden_size
        d_vis = self.vision_model.config.hidden_size

        # 2. Fusion Module (Trainable)
        self.fusion = CrossAttentionFusion(d_text=d_text, d_vis=d_vis, d_model=fused_dim)
        
        # 3. Projection Heads (Trainable)
        d_pool = fused_dim # d_model from fusion
        self.graph_proj = nn.Linear(fused_dim, proj_dim)
        self.text_proj = nn.Linear(d_pool, proj_dim)
        self.vis_proj = nn.Linear(d_pool, proj_dim)

    def get_trainable_parameters(self):
        """Returns parameters of fusion and projection heads."""
        return (
            list(self.fusion.parameters()) +
            list(self.graph_proj.parameters()) +
            list(self.text_proj.parameters()) +
            list(self.vis_proj.parameters())
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor):
        # 1. Get tokens from CLIP (frozen)
        with torch.no_grad():
            txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            vis_out = self.vision_model(pixel_values=pixel_values)
            text_tokens = txt_out.last_hidden_state  # (B, L, d_text)
            img_patches = vis_out.last_hidden_state[:, 1:, :] # (B, P, d_vis), skip CLS
        
        # 2. Get fused embeddings and intermediate pools (trainable)
        fused_vec, text_pool, vis_pool = self.fusion(text_tokens, attention_mask, img_patches)
        
        # 3. Project for losses (trainable)
        z_graph = F.normalize(self.graph_proj(fused_vec), dim=1)
        z_text = F.normalize(self.text_proj(text_pool), dim=1)
        z_vis = F.normalize(self.vis_proj(vis_pool), dim=1)
        
        return z_graph, z_text, z_vis

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor):
        """
        New method for inference. Runs the encoder and returns the 
        intermediate features *before* the projection heads.
        """
        # --- Raw CLIP features ---
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vis_out = self.vision_model(pixel_values=pixel_values)
        
        # [CLS] pools for baselines
        text_cls_pool = txt_out.pooler_output
        vis_cls_pool = vis_out.pooler_output

        # --- Fused features ---
        text_tokens = txt_out.last_hidden_state
        img_patches = vis_out.last_hidden_state[:, 1:, :]
        
        # Note: self.fusion IS trained (from checkpoint)
        fused_vec, text_pool_fused, vis_pool_fused = self.fusion(text_tokens, attention_mask, img_patches)
        
        return {
            "fused": fused_vec,          # Our method
            "text_only": text_cls_pool,  # Baseline 1
            "image_only": vis_cls_pool, # Baseline 2
            "concat": torch.cat([text_cls_pool, vis_cls_pool], dim=-1) # Baseline 3
        }

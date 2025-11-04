import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FUSED_DIM


class CrossAttentionFusion(nn.Module):
    """
    The fusion module. Takes text and image tokens, performs bidirectional
    cross-attention, and returns a fused vector AND the intermediate pools.
    """
    def __init__(self, d_text: int, d_vis: int, d_model: int = FUSED_DIM, n_heads: int = 8):
        super().__init__()
        self.text_proj = nn.Linear(d_text, d_model)
        self.vis_proj = nn.Linear(d_vis, d_model)
        self.text2vis_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.vis2text_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.text_ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
        self.vis_ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)) 
        
        
        # This MLP fuses the pooled representations
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model), 
            nn.LayerNorm(d_model), 
            nn.GELU(), 
            nn.Linear(d_model, d_model)
        )

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor, img_patches: torch.Tensor):
        """
        Runs the fusion module.
        Arguments:
            text_tokens: (B, L, d_text) - text token embeddings
            text_mask: (B, L) - attention mask for text (1 for valid, 0 for padding)
            img_patches: (B, P, d_vis) - image patch embeddings
        Returns:
            fused: (B, d_model) - fused representation for graph-level tasks
            text_pool: (B, d_model) - pooled text representation
            vis_pool: (B, d_model) - pooled visual representation
        """
        T = self.text_proj(text_tokens)  # (B, L, d_model)
        V = self.vis_proj(img_patches)   # (B, P, d_model)

        text_key_padding = (text_mask == 0) if text_mask is not None else None

        # text attends to image (query=T, key=V)
        t2v_out, _ = self.text2vis_attn(query=T, key=V, value=V, key_padding_mask=None)
        t2v = self.text_ffn(t2v_out + T)

        # image attends to text (query=V, key=T)
        v2t_out, _ = self.vis2text_attn(query=V, key=T, value=T, key_padding_mask=text_key_padding)
        v2t = self.vis_ffn(v2t_out + V)

        # --- Pooling ---
        if text_mask is not None:
            mask = text_mask.float().unsqueeze(-1)
            # Pooled text representation from the text-attends-to-visual branch
            # Use max to avoid division by very small numbers
            text_pool = (t2v * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            text_pool = t2v.mean(dim=1)
        
        # Pooled visual representation from the visual-attends-to-text branch
        vis_pool = v2t.mean(dim=1) # (B, d_model)

        fused = self.fusion_mlp(torch.cat([text_pool, vis_pool], dim=-1))
        
        # Return all three for the dual losses
        return fused, text_pool, vis_pool
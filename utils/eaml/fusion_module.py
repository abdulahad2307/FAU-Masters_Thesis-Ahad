import torch
import torch.nn as nn

class EnhancedFusionModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.2):
        """
        Implementation of enhanced self-attention-based fusion module with residual connections
        and dropout for better regularization.
        
        Parameters:
            embed_dim (int): Feature dimension of image and text embeddings.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(EnhancedFusionModule, self).__init__()
        # Image attention block
        self.img_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.img_norm = nn.LayerNorm(embed_dim)
        
        # Text attention block
        self.txt_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.txt_norm = nn.LayerNorm(embed_dim)
        
        # Channel-wise gating
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Feature combiner
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )

    def forward(self, image_feat, text_feat):
        # Reshape for attention: [seq_len, batch, features]
        I = image_feat.unsqueeze(0)  # [1, B, D]
        T = text_feat.unsqueeze(0)   # [1, B, D]
        
        # Cross-modal attention
        img_attn_out, _ = self.img_attn(I, T, T)  # Image attends to text
        txt_attn_out, _ = self.txt_attn(T, I, I)  # Text attends to image
        
        # Residual connections
        img_out = self.img_norm(I + img_attn_out).squeeze(0)
        txt_out = self.txt_norm(T + txt_attn_out).squeeze(0)
        
        # Channel-wise gating
        gate_signal = self.gate(torch.cat([img_out, txt_out], dim=-1))
        gated_img = img_out * gate_signal
        gated_txt = txt_out * (1 - gate_signal)
        
        # Feature fusion
        fused = self.combiner(torch.cat([gated_img, gated_txt], dim=-1))
        return fused

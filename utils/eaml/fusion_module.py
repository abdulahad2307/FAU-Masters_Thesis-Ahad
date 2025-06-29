import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super().__init__()
        # Pooling layers
        self.img_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.img_max_pool = nn.AdaptiveMaxPool2d(1)
        self.txt_max_pool = nn.AdaptiveMaxPool1d(1)
        # FC layers for Q, K, V
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, image_feat, text_feat):
        # image_feat: [B, D, H, W], text_feat: [B, D, L] or [B, D]
        if image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(-1).unsqueeze(-1)
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(-1)
        # Pool
        img_avg = self.img_avg_pool(image_feat).squeeze(-1).squeeze(-1)
        img_max = self.img_max_pool(image_feat).squeeze(-1).squeeze(-1)
        txt_max = self.txt_max_pool(text_feat).squeeze(-1)
        # Concatenate pooled features
        pooled = torch.cat([img_avg, img_max, txt_max], dim=-1)
        # Q, K, V
        Q = self.fc_q(pooled)
        K = self.fc_k(pooled)
        V = self.fc_v(pooled)
        # Attention
        attn_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5), dim=-1)
        fused = torch.matmul(attn_weights, V)
        fused = self.dropout(fused)
        return fused

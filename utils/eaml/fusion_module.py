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
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network with residual connection
        self.fc1 = nn.Linear(embed_dim, embed_dim * 2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, image_feat, text_feat):
        # Stack features for cross-modal attention
        fusion_input = torch.cat((image_feat.unsqueeze(0), text_feat.unsqueeze(0)), dim=0)
        
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(fusion_input, fusion_input, fusion_input)
        attn_output = self.dropout1(attn_output)
        fusion_output = self.norm1(fusion_input + attn_output)
        
        # Mean pooling across modalities
        pooled_output = fusion_output.mean(dim=0)
        
        # Feed-forward network with residual connection
        ff_output = self.fc1(pooled_output)
        ff_output = self.activation(ff_output)
        ff_output = self.fc2(ff_output)
        ff_output = self.dropout2(ff_output)
        
        return self.norm2(pooled_output + ff_output)

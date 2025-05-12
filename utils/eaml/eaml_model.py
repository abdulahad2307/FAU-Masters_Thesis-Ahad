import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion_module import EnhancedFusionModule

class EAMLModel(nn.Module):
    def __init__(self, num_classes=16, embed_dim=512, dropout_rate=0.2):
        super().__init__()
        # Encoders
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        
        # Enhanced fusion module
        self.fusion_module = EnhancedFusionModule(
            embed_dim=embed_dim, 
            num_heads=8, 
            dropout_rate=dropout_rate
        )
        
        # Separate classifiers for mutual learning
        self.image_classifier = nn.Linear(embed_dim, num_classes)
        self.text_classifier = nn.Linear(embed_dim, num_classes)
        self.fusion_classifier = nn.Linear(embed_dim, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize OCR model
        self.ocr_processor = None
        self.ocr_model = None

    def forward(self, images, texts, return_features=False):
        # Extract features
        image_feat = self.image_encoder(images)
        text_feat = self.text_encoder(texts)
        
        # Apply dropout for regularization
        image_feat = self.dropout(image_feat)
        text_feat = self.dropout(text_feat)
        
        # Fuse features
        fused_feat = self.fusion_module(image_feat, text_feat)
        
        # Get predictions from each branch
        image_logits = self.image_classifier(image_feat)
        text_logits = self.text_classifier(text_feat)
        fusion_logits = self.fusion_classifier(fused_feat)
        
        if return_features:
            return {
                'image_logits': image_logits,
                'text_logits': text_logits,
                'fusion_logits': fusion_logits,
                'image_feat': image_feat,
                'text_feat': text_feat,
                'fused_feat': fused_feat
            }
        
        # During inference, return only fusion logits
        return fusion_logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion_module import EnhancedFusionModule

class EAMLModel(nn.Module):
    def __init__(self, num_classes=16, embed_dim=512, dropout_rate=0.2, freeze_image_encoder=False):
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
        #self.ocr_processor = None
        #self.ocr_model = None

    """
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

        """
    
    def forward(self, images=None, texts=None, input_ids=None, attention_mask=None, return_features=False, **kwargs):
        """
        Forward pass for the EAML model with flexible parameter handling.
        
        Args:
            images: Tensor of document images [batch_size, channels, height, width]
            texts: Dictionary or tensor containing text features
            input_ids: Text token IDs (used if texts is None)
            attention_mask: Attention mask for text tokens (used if texts is None)
            return_features: Whether to return intermediate features
            **kwargs: Additional arguments (ignored)
        
        Returns:
            logits or dictionary of features
        """

        # Input validation
        if images is None:
            raise ValueError("Images input cannot be None")
        # Handle different input formats for text features
        if texts is None and input_ids is not None:
            # If input_ids is provided directly
            text_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids)
            }
        elif texts is not None:
            # If texts is provided (either as dict or tensor)
            text_inputs = texts
        else:
            raise ValueError("Either 'texts' or 'input_ids' must be provided")
        
        # Extract features
        image_feat = self.image_encoder(images)
        text_feat = self.text_encoder(text_inputs)
        
        # Apply dropout for regularization
        image_feat = self.dropout(image_feat)
        text_feat = self.dropout(text_feat)

        #Normalization
        image_feat = F.normalize(image_feat, p=2, dim=-1)
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        #print("Features Shape:, image_feat.shape, text_feat.shape)

        # Stack along modality dimension for multi-head attention
        fusion_input = torch.stack([image_feat, text_feat], dim=1)  
        # shape: [batch_size, 2, embed_dim]
        fusion_module = EnhancedFusionModule(embed_dim=512, num_heads=8)
        
        # Fuse features
        fused_feat = self.fusion_module(image_feat, text_feat)
        #print("Fusion output shape:", fused_feat.shape)

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


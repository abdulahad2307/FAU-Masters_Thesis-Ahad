import torch
import torch.nn as nn
from typing import Dict, Union

class DomainIncrementalWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes_per_domain: Dict[str, int]):
        """
        Wrapper for domain incremental learning with enhanced model support
        """
        super().__init__()
        self.base_model = base_model
        self.domain_heads = nn.ModuleDict()
        
        # Determine output dimension
        self.output_dim = self._get_output_dim()
        
        # Create domain-specific classification heads
        for domain, num_classes in num_classes_per_domain.items():
            self.domain_heads[domain] = nn.Linear(self.output_dim, num_classes)
            print(f"Created head for domain '{domain}' with {num_classes} classes (dim: {self.output_dim})")
    
    def _get_output_dim(self):
        """Determine output dimension from base model"""
        if hasattr(self.base_model, 'output_dim'):
            return self.base_model.output_dim
        elif hasattr(self.base_model, 'num_features'):
            return self.base_model.num_features
        elif hasattr(self.base_model, 'hidden_size'):
            return self.base_model.hidden_size
        else:
            # For EAML model, check fusion module output
            if hasattr(self.base_model, 'fusion_classifier'):
                return self.base_model.fusion_classifier.in_features
            elif hasattr(self.base_model, 'image_classifier'):
                return self.base_model.image_classifier.in_features
            else:
                raise ValueError("Cannot determine output dimension. Base model must have 'output_dim' attribute.")
    
    def forward(self, images, domain: str, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass for domain-specific classification with multi-modal support
        """
        if domain not in self.domain_heads:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.domain_heads.keys())}")
        
        # Extract features based on model type
        if hasattr(self.base_model, 'forward_features'):
            # Standard feature extraction (for timm models, etc.)
            features = self.base_model.forward_features(images)
        elif hasattr(self.base_model, 'image_encoder') and hasattr(self.base_model, 'text_encoder'):
            # EAML model - extract fused features
            image_feat = self.base_model.image_encoder(images)
            
            # Handle text input
            if input_ids is not None:
                text_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids)
                }
            else:
                # Create dummy text inputs for EAML
                batch_size = images.size(0)
                text_inputs = {
                    'input_ids': torch.zeros((batch_size, 10), dtype=torch.long, device=images.device),
                    'attention_mask': torch.ones((batch_size, 10), dtype=torch.long, device=images.device)
                }
            
            text_feat = self.base_model.text_encoder(text_inputs)
            
            # Apply dropout if available
            if hasattr(self.base_model, 'dropout'):
                image_feat = self.base_model.dropout(image_feat)
                text_feat = self.base_model.dropout(text_feat)
            
            # Fuse features
            features = self.base_model.fusion_module(image_feat, text_feat)
        else:
            raise ValueError("Base model must implement feature extraction method")
        
        # Flatten if necessary
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Apply domain-specific head
        logits = self.domain_heads[domain](features)
        return logits
    
    def freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Base model frozen")
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("Base model unfrozen")
    
    def freeze_heads(self):
        """Freeze domain-specific heads"""
        for head in self.domain_heads.values():
            for param in head.parameters():
                param.requires_grad = False
        print("Domain heads frozen")
    
    def unfreeze_heads(self):
        """Unfreeze domain-specific heads"""
        for head in self.domain_heads.values():
            for param in head.parameters():
                param.requires_grad = True
        print("Domain heads unfrozen")

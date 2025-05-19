import torch
import torch.nn as nn
from typing import Dict, Union

class DomainIncrementalWrapper(nn.Module):
    """Wrapping a base model to handle multiple domain-specific classifiers"""
    def __init__(self, base_model, num_classes_per_domain):
        """
        Args:
            base_model: The shared feature extractor (e.g., EAML, DocFormer)
            num_classes_per_domain: dict of domain name -> number of classes
        """
        super().__init__()
        self.base_model = base_model
        
        # Check if base model has required output dimension attribute
        if not hasattr(base_model, 'output_dim'):
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                output_dim = base_model.config.hidden_size
            else:
                # Try to infer from existing classifier
                if hasattr(base_model, 'classifier'):
                    output_dim = base_model.classifier.in_features
                else:
                    raise ValueError("Cannot determine output dimension of base model")
        else:
            output_dim = base_model.output_dim
            
        self.output_dim = output_dim
        self.domain_heads = nn.ModuleDict()
        
        # Initialize domain-specific classifiers
        for domain, num_classes in num_classes_per_domain.items():
            self.domain_heads[domain] = nn.Linear(output_dim, num_classes)

    def forward(self, inputs, domain):
        """
        Forward pass using domain-specific classifier
        Args:
            inputs: Dict containing inputs to base_model (image, text, etc.)
            domain: Domain identifier string
        Returns:
            Logits from domain-specific head
        """
        # Check if the domain exists
        if domain not in self.domain_heads:
            raise ValueError(f"Domain {domain} not recognized. Available domains: {list(self.domain_heads.keys())}")
        
        # Get features from base model
        if hasattr(self.base_model, 'forward_features'):
            features = self.base_model.forward_features(inputs)
        else:
            # Try to get features directly from base model
            features = self.base_model(inputs)
            if isinstance(features, dict) and 'features' in features:
                features = features['features']
            elif isinstance(features, tuple) and len(features) > 1:
                features = features[0]  # Assuming first element is features
            
        # Apply domain-specific classifier
        logits = self.domain_heads[domain](features)
        
        return logits
    
    def extract_features(self, inputs, domain=None):
        """Extract features from the base model"""
        if hasattr(self.base_model, 'forward_features'):
            return self.base_model.forward_features(inputs)
        elif hasattr(self.base_model, 'extract_features'):
            return self.base_model.extract_features(inputs)
        else:
            raise NotImplementedError("Base model does not provide a feature extraction method")
            
    def add_domain_head(self, domain, num_classes=None):
        """Add a new domain-specific classifier"""
        if domain in self.domain_heads:
            print(f"Domain {domain} already exists. Not adding a new classifier.")
            return
            
        # Determine number of classes
        if num_classes is None:
            # Use the same number of classes as the first domain
            first_domain = next(iter(self.domain_heads))
            num_classes = self.domain_heads[first_domain].out_features
            
        # Add new classifier
        self.domain_heads[domain] = nn.Linear(self.output_dim, num_classes)
        print(f"Added classifier for domain {domain} with {num_classes} classes")
    
    def freeze_base_model(self):
        """Freeze all parameters in the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = True
            
    def freeze_domain_head(self, domain=None):
        """Freeze parameters of specific domain head or all domain heads"""
        if domain is not None:
            for param in self.domain_heads[domain].parameters():
                param.requires_grad = False
        else:
            for head in self.domain_heads.values():
                for param in head.parameters():
                    param.requires_grad = False
                    
    def unfreeze_domain_head(self, domain=None):
        """Unfreeze parameters of specific domain head or all domain heads"""
        if domain is not None:
            for param in self.domain_heads[domain].parameters():
                param.requires_grad = True
        else:
            for head in self.domain_heads.values():
                for param in head.parameters():
                    param.requires_grad = True

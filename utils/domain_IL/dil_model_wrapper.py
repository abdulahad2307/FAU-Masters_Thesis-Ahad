import torch
import torch.nn as nn
from typing import Dict, Union

class DomainIncrementalWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes_per_domain: Dict[str, int]):
        """
        Wrapping a base model to handle multiple domain-specific classifiers
        
        Args:
            base_model: The shared feature extractor (e.g., EAML, DocFormer)
            num_classes_per_domain: dict of domain name -> number of classes
        """
        super().__init__()
        self.base_model = base_model
        self.domain_heads = nn.ModuleDict()

        for domain, num_classes in num_classes_per_domain.items():
            self.domain_heads[domain] = nn.Linear(base_model.output_dim, num_classes)

    def forward(self, inputs: Dict[str, Union[torch.Tensor, Dict]], domain: str):
        """
        Forward pass using domain-specific classifier

        Args:
            inputs: Dict containing inputs to base_model (image, text, etc.)
            domain: Domain identifier string
        Returns:
            Logits from domain-specific head
        """
        if not hasattr(self.base_model, 'forward_features'):
            raise ValueError("Base model must implement `forward_features()` method.")

        features = self.base_model.forward_features(inputs)  # output feature vector
        logits = self.domain_heads[domain](features)
        return logits

    def freeze_all_except_classifier(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for head in self.domain_heads.values():
            for param in head.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

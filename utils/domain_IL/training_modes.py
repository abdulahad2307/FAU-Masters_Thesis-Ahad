import torch
import torch.nn as nn
from typing import List, Optional

class DILTrainingMode:
    """Base class for different domain-incremental learning training modes."""
    def __init__(self, model: nn.Module):
        self.model = model
    
    def prepare_for_training(self):
        """Set model to correct trainable/frozen state."""
        raise NotImplementedError
    
    def get_trainable_params(self):
        """Return parameters to train in optimizer."""
        raise NotImplementedError

class DILFullModelTraining(DILTrainingMode):
    """Train the full model for domain incremental adaptation."""
    def prepare_for_training(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return self.model
    
    def get_trainable_params(self):
        return self.model.parameters()

class DILLastLayerTraining(DILTrainingMode):
    """Train only the last output heads/classifier layers (typical for a new domain increment)."""
    def prepare_for_training(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # EAML
        if hasattr(self.model, 'image_classifier'):
            for param in self.model.image_classifier.parameters():
                param.requires_grad = True
            if hasattr(self.model, 'text_classifier'):
                for param in self.model.text_classifier.parameters():
                    param.requires_grad = True
            if hasattr(self.model, 'fusion_classifier'):
                for param in self.model.fusion_classifier.parameters():
                    param.requires_grad = True
        # DocFormer or unified classifier
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        return self.model
    
    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

class DILPartialLayerTraining(DILTrainingMode):
    """Train N final layers (e.g., for partial fine-tuning in DIL)."""
    def __init__(self, model: nn.Module, unfreeze_depth: int = 1):
        super().__init__(model)
        self.unfreeze_depth = unfreeze_depth

    def prepare_for_training(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'encoder'):
            encoder = self.model.base_model.encoder
            if hasattr(encoder, 'layer'):
                layers = list(encoder.layer)
                for layer in layers[-self.unfreeze_depth:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        # Also always unfreeze heads
        if hasattr(self.model, 'image_classifier'):
            for param in self.model.image_classifier.parameters():
                param.requires_grad = True
            if hasattr(self.model, 'text_classifier'):
                for param in self.model.text_classifier.parameters():
                    param.requires_grad = True
            if hasattr(self.model, 'fusion_classifier'):
                for param in self.model.fusion_classifier.parameters():
                    param.requires_grad = True
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        return self.model

    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

def get_dil_training_mode(model: nn.Module, mode: str, unfreeze_depth: Optional[int] = None):
    """
    Get proper training mode object for domain incremental learning.
    mode: 'full_finetune', 'head_only', 'partial_finetune'
    """
    if mode == "full_finetune":
        return DILFullModelTraining(model)
    elif mode == "head_only":
        return DILLastLayerTraining(model)
    elif mode == "partial_finetune":
        if unfreeze_depth is None:
            raise ValueError("unfreeze_depth must be provided for partial_finetune mode")
        return DILPartialLayerTraining(model, unfreeze_depth)
    else:
        raise ValueError(f"Unknown DIL training mode: {mode}")

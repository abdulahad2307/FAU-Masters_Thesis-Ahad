import torch
import os
from typing import Dict, Union, List, Optional

from utils.domain_IL.dil_model_wrapper import DomainIncrementalWrapper
from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig

def load_eaml(checkpoint_path, num_classes, device='cuda'):
    """Load EAML model from checkpoint"""
    model = EAMLModel(num_classes=num_classes)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"EAML model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading EAML model: {e}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Using randomly initialized model.")
        
    model = model.to(device)
    return model

def load_docformer(checkpoint_path, num_classes, device='cuda'):
    """Load DocFormer model from checkpoint"""
    config = DocFormerConfig()
    model = DocFormer(config, num_classes=num_classes)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"DocFormer model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading DocFormer model: {e}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Using randomly initialized model.")
        
    model = model.to(device)
    return model

class JointModelWrapper(torch.nn.Module):
    """Ensemble wrapper combining EAML and DocFormer models"""
    def __init__(self, eaml_model, docformer_model, mode="average"):
        super().__init__()
        self.eaml = eaml_model
        self.docformer = docformer_model
        self.mode = mode  # "average" or "concat"
        
    def forward(self, inputs, domain=None):
        """Forward pass through both models and combine results"""
        # Extract relevant inputs for each model
        eaml_inputs = {
            'images': inputs.get('images'),
            'texts': inputs.get('texts')
        }
        
        docformer_inputs = {
            'pixel_values': inputs.get('pixel_values', inputs.get('images')),
            'input_ids': inputs.get('input_ids', inputs.get('texts', {}).get('input_ids')),
            'attention_mask': inputs.get('attention_mask', inputs.get('texts', {}).get('attention_mask')),
            'bboxes': inputs.get('bboxes')
        }
        
        # Forward pass through each model
        eaml_outputs = self.eaml(eaml_inputs, domain) if domain else self.eaml(**eaml_inputs)
        docformer_outputs = self.docformer(**docformer_inputs, task="classification") if domain is None else self.docformer(docformer_inputs, domain)
        
        # Extract logits
        eaml_logits = eaml_outputs
        docformer_logits = docformer_outputs['logits'] if isinstance(docformer_outputs, dict) else docformer_outputs
        
        # Combine outputs based on mode
        if self.mode == "average":
            return (eaml_logits + docformer_logits) / 2
        elif self.mode == "concat":
            # This assumes both models have the same number of classes
            return torch.cat([eaml_logits, docformer_logits], dim=1)
        else:
            raise ValueError(f"Unknown ensemble mode: {self.mode}")
            
    def extract_features(self, inputs, domain=None):
        """Extract features from EAML and DocFormer models"""
        # Extract relevant inputs for each model
        eaml_inputs = {
            'images': inputs.get('images'),
            'texts': inputs.get('texts')
        }
        
        docformer_inputs = {
            'pixel_values': inputs.get('pixel_values', inputs.get('images')),
            'input_ids': inputs.get('input_ids', inputs.get('texts', {}).get('input_ids')),
            'attention_mask': inputs.get('attention_mask', inputs.get('texts', {}).get('attention_mask')),
            'bboxes': inputs.get('bboxes')
        }
        
        # Extract features
        if hasattr(self.eaml, 'extract_features'):
            eaml_features = self.eaml.extract_features(eaml_inputs, domain)
        else:
            eaml_features = None
            
        if hasattr(self.docformer, 'extract_features'):
            docformer_features = self.docformer.extract_features(docformer_inputs, domain)
        else:
            docformer_features = None
            
        # Combine features
        if eaml_features is not None and docformer_features is not None:
            return torch.cat([eaml_features, docformer_features], dim=1)
        elif eaml_features is not None:
            return eaml_features
        elif docformer_features is not None:
            return docformer_features
        else:
            raise ValueError("Neither model provides a feature extraction method")

def prepare_dil_models(config, class_per_domain: Dict[str, int], device='cuda'):
    """
    Returns models for DIL setup.
    Args:
        config: Configuration dictionary with the following keys:
            - 'strategy': 'eaml_only', 'docformer_only', 'pre_ensemble', or 'post_ensemble'
            - 'eaml_ckpt': Path to EAML checkpoint
            - 'doc_ckpt': Path to DocFormer checkpoint
            - 'dil_mode': Whether to wrap models in DomainIncrementalWrapper
        class_per_domain: Dict mapping domain names to number of classes
        device: Device to load models onto
    Returns:
        Prepared model(s) for DIL
    """
    strategy = config.get("strategy", "eaml_only")
    dil = config.get("dil_mode", True)
    total_classes = sum(class_per_domain.values())
    
    # Load base models
    eaml = None
    docformer = None
    
    if strategy in ["eaml_only", "pre_ensemble", "post_ensemble"]:
        eaml = load_eaml(config.get("eaml_ckpt", ""), total_classes, device)
        
    if strategy in ["docformer_only", "pre_ensemble", "post_ensemble"]:
        docformer = load_docformer(config.get("doc_ckpt", ""), total_classes, device)
    
    # Prepare according to strategy
    if strategy == "eaml_only":
        model = eaml
    elif strategy == "docformer_only":
        model = docformer
    elif strategy == "pre_ensemble":
        model = JointModelWrapper(eaml, docformer, mode=config.get("ensemble_mode", "average"))
    elif strategy == "post_ensemble":
        model = {
            "eaml": eaml,
            "docformer": docformer
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Wrap in DomainIncrementalWrapper if required
    if dil:
        if isinstance(model, dict):  # post-ensemble, wrapping each model separately
            model = {
                name: DomainIncrementalWrapper(m, class_per_domain).to(device)
                for name, m in model.items()
            }
        else:
            model = DomainIncrementalWrapper(model, class_per_domain).to(device)
    
    return model

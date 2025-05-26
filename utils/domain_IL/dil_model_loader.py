import torch
import torch.nn as nn
import os
from .dil_model_wrapper import DomainIncrementalWrapper
from utils.ensemble.joint_wrapper import JointModelWrapper
from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig

def load_model_with_size_mismatch_handling(model, checkpoint_path, device, ignore_classifier=True):
    """
    Load model state dict while handling size mismatches in classifier layers
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
        ignore_classifier: Whether to ignore classifier layers (for DIL)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return model
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint with proper handling
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get current model state dict
    current_state_dict = model.state_dict()
    
    # Filter out incompatible layers
    filtered_state_dict = {}
    classifier_keys = ['image_classifier', 'text_classifier', 'fusion_classifier', 'classifier', 'fc', 'head']
    skipped_keys = []
    loaded_keys = []
    
    for key, value in state_dict.items():
        if key in current_state_dict:
            # Check if shapes match
            if value.shape == current_state_dict[key].shape:
                filtered_state_dict[key] = value
                loaded_keys.append(key)
            else:
                # Skip classifier layers if ignore_classifier is True
                if ignore_classifier and any(cls_key in key for cls_key in classifier_keys):
                    print(f"Ignoring size mismatch in classifier layer: {key} "
                          f"(checkpoint: {value.shape} vs model: {current_state_dict[key].shape})")
                    skipped_keys.append(key)
                    continue
                else:
                    print(f"Size mismatch for {key}: checkpoint {value.shape} vs model {current_state_dict[key].shape}")
                    skipped_keys.append(key)
        else:
            print(f"Key not found in current model: {key}")
            skipped_keys.append(key)
    
    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"Successfully loaded {len(loaded_keys)} parameters")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} incompatible parameters")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    
    return model

def load_eaml(checkpoint_path, num_classes, device='cpu'):
    """Load EAML model with size mismatch handling"""
    model = EAMLModel(num_classes=num_classes)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_model_with_size_mismatch_handling(
            model, checkpoint_path, device, ignore_classifier=True
        )
    else:
        print(f"EAML checkpoint not found at {checkpoint_path}, using random initialization")
    
    # Add output_dim attribute for DIL compatibility
    model.output_dim = 512  # EAML embedding dimension
    model.eval()
    return model

def load_docformer(checkpoint_path, num_classes, device='cpu'):
    """Load DocFormer model with size mismatch handling"""
    config = DocFormerConfig()
    model = DocFormer(config, num_classes=num_classes)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_model_with_size_mismatch_handling(
            model, checkpoint_path, device, ignore_classifier=True
        )
    else:
        print(f"DocFormer checkpoint not found at {checkpoint_path}, using random initialization")
    
    # Add output_dim attribute for DIL compatibility
    model.output_dim = getattr(model, 'hidden_size', 768)
    model.eval()
    return model

def load_timm_model(checkpoint_path, num_classes=None, model_name='inception_resnet_v2'):
    """Load timm model with size mismatch handling"""
    import timm
    
    model = timm.create_model(model_name, pretrained=False)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_model_with_size_mismatch_handling(
            model, checkpoint_path, 'cpu', ignore_classifier=True
        )
    
    # Modify classifier if needed
    if num_classes is not None:
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'head'):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
    
    # Add output_dim for DIL compatibility
    if hasattr(model, 'num_features'):
        model.output_dim = model.num_features
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        model.output_dim = model.classifier.in_features
    else:
        model.output_dim = 2048  # Default for inception_resnet_v2
    
    return model

def prepare_dil_models(config, class_per_domain: dict, device='cuda'):
    """
    Returns models for DIL setup with enhanced loading capabilities.
    
    config: {
        'strategy': 'eaml_only' | 'docformer_only' | 'pre_ensemble' | 'post_ensemble' | 'timm_only',
        'eaml_ckpt': 'path/to/eaml_best.pth',
        'doc_ckpt': 'path/to/docformer_best.pth',
        'timm_ckpt': 'path/to/timm_model.pth',
        'model_name': 'inception_resnet_v2',  # for timm models
        'dil_mode': True/False
    }
    """
    strategy = config.get("strategy", "eaml_only")
    dil = config.get("dil_mode", True)
    total_classes = sum(class_per_domain.values())
    
    print(f"Preparing DIL models with strategy: {strategy}")
    print(f"Total classes across domains: {total_classes}")
    print(f"Domain class distribution: {class_per_domain}")
    
    ## === Loading Base Models === ##
    models = {}
    
    if strategy in ["eaml_only", "pre_ensemble", "post_ensemble"]:
        print("Loading EAML model...")
        eaml = load_eaml(config.get("eaml_ckpt", ""), num_classes=total_classes, device=device)
        eaml = eaml.to(device)
        models["eaml"] = eaml
    
    if strategy in ["docformer_only", "pre_ensemble", "post_ensemble"]:
        print("Loading DocFormer model...")
        docformer = load_docformer(config.get("doc_ckpt", ""), num_classes=total_classes, device=device)
        docformer = docformer.to(device)
        models["docformer"] = docformer
    
    if strategy == "timm_only":
        print("Loading TIMM model...")
        timm_model = load_timm_model(
            config.get("timm_ckpt", ""), 
            num_classes=total_classes,
            model_name=config.get("model_name", "inception_resnet_v2")
        )
        timm_model = timm_model.to(device)
        models["timm"] = timm_model
    
    ## === Preparing according to strategy === ##
    if strategy == "eaml_only":
        model = models["eaml"]
    elif strategy == "docformer_only":
        model = models["docformer"]
    elif strategy == "timm_only":
        model = models["timm"]
    elif strategy == "pre_ensemble":
        # Joint wrapper for pre-ensemble
        if "eaml" in models and "docformer" in models:
            model = JointModelWrapper(models["eaml"], models["docformer"], mode="average")
        else:
            raise ValueError("Pre-ensemble requires both EAML and DocFormer models")
    elif strategy == "post_ensemble":
        # Dictionary for post-ensemble
        model = models
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    ## === Wrapping in DomainIncrementalWrapper if required === ##
    if dil:
        if isinstance(model, dict):  # post-ensemble, wrapping each model separately
            wrapped_models = {}
            for name, m in model.items():
                print(f"Wrapping {name} model for DIL...")
                wrapped_models[name] = DomainIncrementalWrapper(m, class_per_domain).to(device)
            model = wrapped_models
        else:
            print("Wrapping model for DIL...")
            model = DomainIncrementalWrapper(model, class_per_domain).to(device)
    
    print("Model preparation completed successfully!")
    return model

def prepare_dil_models_simple(eaml_path, docformer_path, class_per_domain, device='cuda', model_type='eaml'):
    """
    Simplified version for basic DIL setup
    """
    config = {
        'strategy': f'{model_type}_only',
        'eaml_ckpt': eaml_path,
        'doc_ckpt': docformer_path,
        'dil_mode': True
    }
    return prepare_dil_models(config, class_per_domain, device)

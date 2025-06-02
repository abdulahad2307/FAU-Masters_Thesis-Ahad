import torch
import torch.nn as nn
import os
import time
from tqdm import tqdm

def set_finetune_mode(model, mode="head_only", encoder_unfreeze_depth=1):
    """Set fine-tuning mode for DIL models with enhanced support"""
    
    # Handle DIL wrapped models
    if hasattr(model, 'base_model') and hasattr(model, 'domain_heads'):
        # This is a DomainIncrementalWrapper
        print(f"Setting DIL model to {mode} mode")
        
        if mode == "head_only":
            # Freeze base model
            for param in model.base_model.parameters():
                param.requires_grad = False
            # Unfreeze domain heads
            for head in model.domain_heads.values():
                for param in head.parameters():
                    param.requires_grad = True
            print("Frozen base model, unfrozen domain heads")
            
        elif mode == "partial_finetune":
            # Freeze base model first
            for param in model.base_model.parameters():
                param.requires_grad = False
            
            # Unfreeze domain heads
            for head in model.domain_heads.values():
                for param in head.parameters():
                    param.requires_grad = True
            
            # Unfreeze last few encoder layers if available
            if hasattr(model.base_model, "encoder") and hasattr(model.base_model.encoder, "layer"):
                encoder_blocks = list(model.base_model.encoder.layer)
                for layer in encoder_blocks[-encoder_unfreeze_depth:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Unfrozen last {encoder_unfreeze_depth} encoder layers")
            elif hasattr(model.base_model, "image_encoder"):
                # For EAML model - unfreeze last layers of image encoder
                if hasattr(model.base_model.image_encoder, "model"):
                    layers = list(model.base_model.image_encoder.model.children())
                    for layer in layers[-encoder_unfreeze_depth:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    print(f"Unfrozen last {encoder_unfreeze_depth} image encoder layers")
            
            print("Partial fine-tuning mode enabled")
            
        elif mode == "full_finetune":
            # Unfreeze everything
            for param in model.parameters():
                param.requires_grad = True
            print("Full fine-tuning mode enabled")
            
    else:
        # Handle non-DIL models (backward compatibility)
        print(f"Setting standard model to {mode} mode")
        
        if mode == "head_only":
            for param in model.parameters():
                param.requires_grad = False
            # Try different classifier names
            if hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
                    
        elif mode == "partial_finetune":
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier
            if hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            
            # Unfreeze encoder layers if available
            if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
                encoder_blocks = list(model.encoder.layer)
                for layer in encoder_blocks[-encoder_unfreeze_depth:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        
        elif mode == "full_finetune":
            for param in model.parameters():
                param.requires_grad = True
    
    # Print parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

def save_checkpoint_dil(model, optimizer, epoch, path):
    """Save checkpoint for DIL models"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if isinstance(model, dict):
        # Multiple models (ensemble)
        checkpoint['model_state_dict'] = {
            name: m.state_dict() for name, m in model.items()
        }
    else:
        # Single model
        checkpoint['model_state_dict'] = model.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint_dil(model, optimizer, path, device):
    """Load checkpoint for DIL models"""
    if not os.path.exists(path):
        print("No checkpoint found, starting from scratch.")
        return 0
    
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(model, dict):
        for name, submodel in model.items():
            if name in checkpoint['model_state_dict']:
                submodel.load_state_dict(checkpoint['model_state_dict'][name])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resuming from epoch {epoch + 1}")
    return epoch + 1

def train_one_epoch_dil(model, train_loaders, optimizer, criterion, device):
    """Train DIL model for one epoch across all domains"""
    if isinstance(model, dict):
        for m in model.values():
            m.train()
    else:
        model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()
    
    # Train on each domain
    for domain, dataloader in train_loaders.items():
        print(f"Training on domain: {domain}")
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Training {domain}", leave=False)):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, dict):
                # Ensemble case
                outputs = []
                for model_name, single_model in model.items():
                    if hasattr(single_model.base_model, 'text_encoder'):
                        # EAML model needs text inputs
                        batch_size = images.size(0)
                        dummy_input_ids = torch.zeros((batch_size, 10), dtype=torch.long).to(device)
                        dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
                        output = single_model(images, domain, input_ids=dummy_input_ids, 
                                            attention_mask=dummy_attention_mask)
                    else:
                        output = single_model(images, domain)
                    outputs.append(output)
                final_output = torch.stack(outputs).mean(dim=0)
            else:
                # Single model case
                if hasattr(model.base_model, 'text_encoder'):
                    # EAML model
                    batch_size = images.size(0)
                    dummy_input_ids = torch.zeros((batch_size, 10), dtype=torch.long).to(device)
                    dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
                    final_output = model(images, domain, input_ids=dummy_input_ids, 
                                       attention_mask=dummy_attention_mask)
                else:
                    final_output = model(images, domain)
            
            loss = criterion(final_output, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            _, predicted = final_output.max(1)
            total_loss += loss.item() * images.size(0)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    epoch_time = time.time() - start_time
    
    print(f"Training - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Time: {epoch_time:.2f}s")
    return avg_loss, avg_acc
"""
def evaluate_dil(model, val_loaders, device):
    if isinstance(model, dict):
        for m in model.values():
            m.eval()
    else:
        model.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for domain, dataloader in val_loaders.items():
            print(f"Evaluating domain: {domain}")
            
            for images, labels in tqdm(dataloader, desc=f"Evaluating {domain}", leave=False):
                images, labels = images.to(device), labels.to(device)
                
                if isinstance(model, dict):
                    # Ensemble case
                    outputs = []
                    for model_name, single_model in model.items():
                        if hasattr(single_model.base_model, 'text_encoder'):
                            batch_size = images.size(0)
                            dummy_input_ids = torch.zeros((batch_size, 10), dtype=torch.long).to(device)
                            dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
                            output = single_model(images, domain, input_ids=dummy_input_ids, 
                                                attention_mask=dummy_attention_mask)
                        else:
                            output = single_model(images, domain)
                        outputs.append(output)
                    final_output = torch.stack(outputs).mean(dim=0)
                else:
                    # Single model case
                    if hasattr(model.base_model, 'text_encoder'):
                        batch_size = images.size(0)
                        dummy_input_ids = torch.zeros((batch_size, 10), dtype=torch.long).to(device)
                        dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
                        final_output = model(images, domain, input_ids=dummy_input_ids, 
                                           attention_mask=dummy_attention_mask)
                    else:
                        final_output = model(images, domain)
                
                _, predicted = final_output.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f"Overall Validation Accuracy: {accuracy:.4f}")
    return accuracy
"""

def evaluate_dil(
    model, 
    val_loaders, 
    device, 
    evm_classifier=None, 
    use_evm=False, 
    class_name_to_idx=None
):
    """
    Evaluate model or EVM on all domains.

    Args:
        model: PyTorch model or dict of models (ensemble)
        val_loaders: dict of domain_name -> DataLoader
        device: torch device
        evm_classifier: EVM classifier instance (optional)
        use_evm: bool, whether to use EVM for predictions
        class_name_to_idx: dict mapping class names to indices (for EVM, optional)

    Returns:
        overall_accuracy: float
    """
    if isinstance(model, dict):
        for m in model.values():
            m.eval()
    else:
        model.eval()
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for domain, dataloader in val_loaders.items():
            print(f"Evaluating domain: {domain}")
            domain_correct = 0
            domain_total = 0

            for images, labels in tqdm(dataloader, desc=f"Evaluating {domain}", leave=False):
                images, labels = images.to(device), labels.to(device)

                if use_evm and evm_classifier is not None:
                    # Extract features for EVM
                    if isinstance(model, dict):
                        # Use the first model for features
                        feature_model = list(model.values())[0]
                    else:
                        feature_model = model

                    if hasattr(feature_model, 'forward_features'):
                        features = feature_model.forward_features(images, domain)
                    else:
                        # fallback: use logits as features
                        features = feature_model(images, domain)
                    if isinstance(features, dict) and 'logits' in features:
                        features = features['logits']
                    if len(features.shape) > 2:
                        features = features.view(features.size(0), -1)
                    features_np = features.cpu().numpy()
                    evm_preds, _ = evm_classifier.predict(features_np)
                    # If EVM returns class names, map to indices
                    if class_name_to_idx is not None:
                        evm_preds = [class_name_to_idx.get(str(p), -1) for p in evm_preds]
                    preds = torch.tensor(evm_preds, device=device)
                else:
                    # Standard model prediction (with ensemble support)
                    if isinstance(model, dict):
                        outputs = []
                        for model_name, single_model in model.items():
                            if hasattr(single_model.base_model, 'text_encoder'):
                                batch_size = images.size(0)
                                dummy_input_ids = torch.zeros((batch_size, 10), dtype=torch.long).to(device)
                                dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
                                output = single_model(images, domain, input_ids=dummy_input_ids, 
                                                    attention_mask=dummy_attention_mask)
                            else:
                                output = single_model(images, domain)
                            outputs.append(output)
                        final_output = torch.stack(outputs).mean(dim=0)
                    else:
                        if hasattr(model.base_model, 'text_encoder'):
                            batch_size = images.size(0)
                            dummy_input_ids = torch.zeros((batch_size, 10), dtype=torch.long).to(device)
                            dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
                            final_output = model(images, domain, input_ids=dummy_input_ids, 
                                               attention_mask=dummy_attention_mask)
                        else:
                            final_output = model(images, domain)
                    if isinstance(final_output, dict) and 'logits' in final_output:
                        final_output = final_output['logits']
                    _, preds = final_output.max(1)

                domain_correct += preds.eq(labels).sum().item()
                domain_total += labels.size(0)

            acc = domain_correct / domain_total if domain_total > 0 else 0
            print(f"Domain '{domain}' Accuracy: {acc:.4f}")
            total_correct += domain_correct
            total_samples += domain_total

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Overall Validation Accuracy: {accuracy:.4f}")
    return accuracy
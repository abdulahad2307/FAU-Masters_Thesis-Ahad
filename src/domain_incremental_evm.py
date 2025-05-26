import argparse
import torch
import os
import time
import numpy as np
from torch import nn
from pathlib import Path

# Import DIL-specific modules
from utils.domain_IL.dil_dataloader import DILDataLoader
from utils.domain_IL.dil_model_loader import prepare_dil_models
from utils.domain_IL.dil_train_utils import (
    train_one_epoch_dil,
    evaluate_dil,
    save_checkpoint_dil,
    load_checkpoint_dil,
    set_finetune_mode
)
from utils.domain_IL.evm_classifier import EVMDomainClassifier

def parse_class_counts(class_counts_str, domain_list, default_classes):
    """Parse class counts string or use defaults"""
    if class_counts_str:
        class_counts = {}
        for item in class_counts_str.split(','):
            domain, count = item.split(':')
            class_counts[domain] = int(count)
        return class_counts
    else:
        return {domain: default_classes for domain in domain_list}

def extract_features_for_evm(model, data_loaders, device):
    """Extract features from model for EVM training"""
    print("Extracting features for EVM classifier...")
    
    if isinstance(model, dict):
        # Use first model for feature extraction
        feature_model = list(model.values())[0]
    else:
        feature_model = model
    
    feature_model.eval()
    domain_features = {}
    
    with torch.no_grad():
        for domain, dataloader in data_loaders.items():
            features_list = []
            
            for images, labels in dataloader:
                images = images.to(device)
                
                # Extract features using base model
                if hasattr(feature_model, 'base_model'):
                    # DIL wrapped model
                    if hasattr(feature_model.base_model, 'text_encoder'):
                        # EAML model
                        batch_size = images.size(0)
                        dummy_input_ids = torch.zeros((batch_size, 10), dtype=torch.long).to(device)
                        dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
                        
                        image_feat = feature_model.base_model.image_encoder(images)
                        text_feat = feature_model.base_model.text_encoder({
                            'input_ids': dummy_input_ids,
                            'attention_mask': dummy_attention_mask
                        })
                        features = feature_model.base_model.fusion_module(image_feat, text_feat)
                    else:
                        # Other models with forward_features
                        features = feature_model.base_model.forward_features(images)
                else:
                    # Direct model
                    if hasattr(feature_model, 'forward_features'):
                        features = feature_model.forward_features(images)
                    else:
                        # Extract manually
                        features = feature_model(images)
                
                # Flatten features if necessary
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                features_list.append(features.cpu().numpy())
            
            domain_features[domain] = np.vstack(features_list)
            print(f"Extracted {domain_features[domain].shape[0]} features for domain '{domain}'")
    
    return domain_features

def main(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    ## ---------------------- Loading Class Info ------------------------------ ##
    if args.class_list_path:
        import json
        with open(args.class_list_path) as f:
            class_list = json.load(f)
    else:
        class_list = None
    
    ## ---------------------- Data Loading ------------------------------------ ##
    print("=== Loading Data ===")
    data_start = time.time()
    
    dil_loader = DILDataLoader(
        data_root=args.data_dir,
        domain_list=args.domain_list,
        batch_size=args.batch_size,
        img_size=(224, 224),
        num_workers=4
    )
    
    train_loaders = dil_loader.get_domain_loaders('train')
    val_loaders = dil_loader.get_domain_loaders('val')
    
    if not train_loaders:
        raise ValueError("No training data found. Check domain paths and directory structure.")
    
    class_counts = parse_class_counts(args.class_counts, args.domain_list, args.num_classes)
    
    print(f"Class counts: {class_counts}")
    print(f"Data loaded in {time.time() - data_start:.2f}s")
    
    ## ---------------------- Model Preparation ------------------------------- ##
    print("=== Preparing Models ===")
    
    model_config = {
        'strategy': f'{args.model}_only',
        'eaml_ckpt': args.eaml_path,
        'doc_ckpt': args.docformer_path,
        'dil_mode': True
    }
    
    if args.ensemble_first and args.model == "both":
        model_config['strategy'] = 'pre_ensemble'
    elif args.model == "both":
        model_config['strategy'] = 'post_ensemble'
    
    model = prepare_dil_models(model_config, class_counts, device)
    
    print(f"Setting finetune mode: {args.finetune_mode}")
    if isinstance(model, dict):
        for single_model in model.values():
            set_finetune_mode(single_model, mode=args.finetune_mode, 
                            encoder_unfreeze_depth=args.unfreeze_depth)
    else:
        set_finetune_mode(model, mode=args.finetune_mode, 
                         encoder_unfreeze_depth=args.unfreeze_depth)
    
    ## ---------------------- EVM Classifier Setup ---------------------------- ##
    evm_classifier = None
    if args.use_evm:
        print("=== Setting up EVM Classifier ===")
        evm_classifier = EVMDomainClassifier(
            tailsize=args.evm_tailsize,
            threshold=args.evm_threshold
        )
        
        # Extract features for EVM training
        domain_features = extract_features_for_evm(model, train_loaders, device)
        
        # Train EVM classifier
        evm_classifier.fit(domain_features)
        print("EVM classifier trained successfully!")
    
    ## ---------------------- Training Setup ---------------------------------- ##
    print("=== Setting up Training ===")
    
    if isinstance(model, dict):
        all_params = []
        for single_model in model.values():
            trainable_params = [p for p in single_model.parameters() if p.requires_grad]
            all_params.extend(trainable_params)
    else:
        all_params = [p for p in model.parameters() if p.requires_grad]
    
    if len(all_params) == 0:
        raise ValueError("No trainable parameters found. Check finetune_mode setting.")
    
    print(f"Total trainable parameters: {sum(p.numel() for p in all_params):,}")
    
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_path = os.path.join(args.ckpt_dir, f"{args.model}_dil.pth")
    start_epoch = load_checkpoint_dil(model, optimizer, checkpoint_path, device)
    
    ## ---------------------- Training Loop ----------------------------------- ##
    print("=== Starting Training ===")
    best_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_one_epoch_dil_with_evm(
            model, train_loaders, optimizer, criterion, device, 
            evm_classifier=evm_classifier, use_evm=args.use_evm
        )
        
        # Validation
        val_acc = evaluate_dil_with_evm(
            model, val_loaders, device,
            evm_classifier=evm_classifier, use_evm=args.use_evm
        )
        
        # Update EVM with new features if enabled
        if args.use_evm and epoch % args.evm_update_freq == 0:
            print("Updating EVM classifier...")
            domain_features = extract_features_for_evm(model, train_loaders, device)
            evm_classifier.incremental_update(domain_features)
        
        # Save checkpoint
        save_checkpoint_dil_with_evm(model, optimizer, epoch, checkpoint_path, evm_classifier)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.ckpt_dir, f"{args.model}_best.pth")
            save_checkpoint_dil_with_evm(model, optimizer, epoch, best_path, evm_classifier)
            print(f"✓ New best model saved: {best_acc:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n=== Training Completed ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

def train_one_epoch_dil_with_evm(model, train_loaders, optimizer, criterion, device, 
                                evm_classifier=None, use_evm=False):
    """Enhanced training function with optional EVM"""
    if isinstance(model, dict):
        for m in model.values():
            m.train()
    else:
        model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    evm_correct = 0 if use_evm else 0
    
    for domain, dataloader in train_loaders.items():
        print(f"Training on domain: {domain}")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Get model predictions
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
    return avg_loss, avg_acc

def evaluate_dil_with_evm(model, val_loaders, device, evm_classifier=None, use_evm=False):
    """Enhanced evaluation function with optional EVM"""
    if isinstance(model, dict):
        for m in model.values():
            m.eval()
    else:
        model.eval()
    
    total_correct = 0
    total_samples = 0
    evm_correct = 0 if use_evm else 0
    
    with torch.no_grad():
        for domain, dataloader in val_loaders.items():
            print(f"Evaluating domain: {domain}")
            
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                
                # Get model predictions
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
                
                _, predicted = final_output.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
                
                # EVM domain prediction (optional)
                if use_evm and evm_classifier:
                    # Extract features for EVM
                    if isinstance(model, dict):
                        feature_model = list(model.values())[0]
                    else:
                        feature_model = model
                    
                    # Get features for EVM prediction
                    if hasattr(feature_model, 'base_model'):
                        if hasattr(feature_model.base_model, 'text_encoder'):
                            image_feat = feature_model.base_model.image_encoder(images)
                            text_feat = feature_model.base_model.text_encoder({
                                'input_ids': dummy_input_ids,
                                'attention_mask': dummy_attention_mask
                            })
                            features = feature_model.base_model.fusion_module(image_feat, text_feat)
                        else:
                            features = feature_model.base_model.forward_features(images)
                    
                    if len(features.shape) > 2:
                        features = features.view(features.size(0), -1)
                    
                    # EVM domain prediction
                    evm_predictions = evm_classifier.predict(features.cpu().numpy())
                    evm_correct += sum([1 for pred in evm_predictions if pred == domain])
    
    accuracy = total_correct / total_samples
    
    if use_evm and evm_classifier:
        evm_accuracy = evm_correct / total_samples
        print(f"EVM Domain Accuracy: {evm_accuracy:.4f}")
    
    return accuracy

def save_checkpoint_dil_with_evm(model, optimizer, epoch, path, evm_classifier=None):
    """Save checkpoint including EVM classifier"""
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if isinstance(model, dict):
        checkpoint['model_state_dict'] = {
            name: m.state_dict() for name, m in model.items()
        }
    else:
        checkpoint['model_state_dict'] = model.state_dict()
    
    if evm_classifier:
        checkpoint['evm_state'] = {
            'weibull_models': evm_classifier.weibull_models,
            'domain_features': evm_classifier.domain_features,
            'tailsize': evm_classifier.tailsize,
            'threshold': evm_classifier.threshold,
            'initialized': evm_classifier.initialized
        }
    
    torch.save(checkpoint, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Incremental Learning with EVM")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--domain_list", nargs="+", required=True)
    parser.add_argument("--class_counts", type=str)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_dil")
    parser.add_argument("--eaml_path", type=str, default="")
    parser.add_argument("--docformer_path", type=str, default="")
    parser.add_argument("--model", choices=["eaml", "docformer", "both"], default="eaml")
    parser.add_argument("--ensemble_first", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--finetune_mode", choices=["head_only", "partial_finetune", "full_finetune"], 
                       default="head_only")
    parser.add_argument("--unfreeze_depth", type=int, default=2)
    parser.add_argument("--class_list_path", type=str)
    parser.add_argument("--num_classes", type=int, default=16)
    
    # EVM-specific arguments
    parser.add_argument("--use_evm", action="store_true", 
                       help="Enable EVM classifier for domain detection")
    parser.add_argument("--evm_tailsize", type=float, default=0.5,
                       help="EVM tailsize parameter")
    parser.add_argument("--evm_threshold", type=float, default=0.7,
                       help="EVM threshold for domain prediction")
    parser.add_argument("--evm_update_freq", type=int, default=5,
                       help="Update EVM every N epochs")
    
    args = parser.parse_args()
    
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    main(args)

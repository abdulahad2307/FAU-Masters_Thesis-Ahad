import argparse
import torch
import os
import time
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
        class_list = None  # fallback to auto-detect
    
    ## ---------------------- Data Loading ------------------------------------ ##
    print("=== Loading Data ===")
    data_start = time.time()
    
    # Use DIL DataLoader for domain-specific loading
    dil_loader = DILDataLoader(
        data_root=args.data_dir,
        domain_list=args.domain_list,
        batch_size=args.batch_size,
        img_size=(224, 224),
        num_workers=4
    )
    
    # Get loaders for each domain
    train_loaders = dil_loader.get_domain_loaders('train')
    val_loaders = dil_loader.get_domain_loaders('val')
    
    if not train_loaders:
        raise ValueError("No training data found. Check domain paths and directory structure.")
    
    # Get class counts
    class_counts = parse_class_counts(args.class_counts, args.domain_list, args.num_classes)
    
    print(f"Class counts: {class_counts}")
    print(f"Data loaded in {time.time() - data_start:.2f}s")
    
    ## ---------------------- Model Preparation ------------------------------- ##
    print("=== Preparing Models ===")
    
    # Prepare configuration for model loading
    model_config = {
        'strategy': f'{args.model}_only',
        'eaml_ckpt': args.eaml_path,
        'doc_ckpt': args.docformer_path,
        'dil_mode': True
    }
    
    # Handle ensemble case
    if args.ensemble_first and args.model == "both":
        model_config['strategy'] = 'pre_ensemble'
    elif args.model == "both":
        model_config['strategy'] = 'post_ensemble'
    
    # Load and prepare models
    model = prepare_dil_models(model_config, class_counts, device)
    
    # Set fine-tuning mode AFTER model preparation
    print(f"Setting finetune mode: {args.finetune_mode}")
    if isinstance(model, dict):
        for single_model in model.values():
            set_finetune_mode(single_model, mode=args.finetune_mode, 
                            encoder_unfreeze_depth=args.unfreeze_depth)
    else:
        set_finetune_mode(model, mode=args.finetune_mode, 
                         encoder_unfreeze_depth=args.unfreeze_depth)
    
    ## ---------------------- Training Setup ---------------------------------- ##
    print("=== Setting up Training ===")
    
    # Setup optimizer - collect trainable parameters
    if isinstance(model, dict):
        all_params = []
        for single_model in model.values():
            trainable_params = [p for p in single_model.parameters() if p.requires_grad]
            all_params.extend(trainable_params)
            print(f"Model {type(single_model).__name__}: {len(trainable_params)} trainable parameters")
    else:
        all_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Model {type(model).__name__}: {len(all_params)} trainable parameters")
    
    # Check if we have trainable parameters
    if len(all_params) == 0:
        print("ERROR: No trainable parameters found!")
        print("Model structure:")
        if isinstance(model, dict):
            for name, single_model in model.items():
                print(f"  {name}: {type(single_model).__name__}")
                for param_name, param in single_model.named_parameters():
                    print(f"    {param_name}: requires_grad={param.requires_grad}")
        else:
            for param_name, param in model.named_parameters():
                print(f"  {param_name}: requires_grad={param.requires_grad}")
        raise ValueError("No trainable parameters found. Check finetune_mode setting.")
    
    print(f"Total trainable parameters: {sum(p.numel() for p in all_params):,}")
    
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(args.ckpt_dir, f"{args.model}_dil.pth")
    start_epoch = load_checkpoint_dil(model, optimizer, checkpoint_path, device)
    
    ## ---------------------- Training Loop ----------------------------------- ##
    print("=== Starting Training ===")
    best_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_one_epoch_dil(
            model, train_loaders, optimizer, criterion, device
        )
        
        # Validation
        val_acc = evaluate_dil(model, val_loaders, device)
        
        # Save checkpoint
        save_checkpoint_dil(model, optimizer, epoch, checkpoint_path)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.ckpt_dir, f"{args.model}_best.pth")
            save_checkpoint_dil(model, optimizer, epoch, best_path)
            print(f"✓ New best model saved: {best_acc:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n=== Training Completed ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Incremental Learning")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root directory containing domain subdirectories")
    parser.add_argument("--domain_list", nargs="+", required=True,
                       help="List of domain names (subdirectory names)")
    parser.add_argument("--class_counts", type=str,
                       help="Domain-specific class counts (format: domain1:16,domain2:16)")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_dil")
    parser.add_argument("--eaml_path", type=str, default="")
    parser.add_argument("--docformer_path", type=str, default="")
    parser.add_argument("--model", choices=["eaml", "docformer", "both"], default="eaml")
    parser.add_argument("--ensemble_first", action="store_true", 
                       help="Use pre-ensemble (only valid with --model both)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--finetune_mode", choices=["head_only", "partial_finetune", "full_finetune"], 
                       default="head_only")
    parser.add_argument("--unfreeze_depth", type=int, default=2)
    parser.add_argument("--class_list_path", type=str,
                       help="Optional path to JSON file with class list")
    parser.add_argument("--num_classes", type=int, default=16,
                       help="Default number of classes per domain")
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)

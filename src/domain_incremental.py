#!/usr/bin/env python3
import os
import argparse
import torch
import time
from typing import Dict, List

from utils.domain_IL.dil_dataset import get_all_domains_loaders
from utils.domain_IL.dil_model_loader import prepare_dil_models
from utils.domain_IL.dil_train_utils import (
    train_one_epoch_dil,
    evaluate_dil,
    save_checkpoint_dil,
    load_checkpoint_dil,
    DILMetrics
)
from utils.domain_IL.dil_utils import (
    StandardDomainIL,
    DistillationDomainIL,
    EWC,
    extract_features
)
from utils.domain_IL.adaptive_lr import AdaptiveLR
from utils.domain_IL.evm_classifier import EVMDomainClassifier

def main():
    parser = argparse.ArgumentParser(description="Domain Incremental Learning")
    
    # Data arguments
    parser.add_argument("--data_dir", required=True,
                      help="Root directory containing domain folders")
    parser.add_argument("--domain_list", nargs="+", required=True,
                      help="List of domain names to process incrementally")
    parser.add_argument("--class_counts", type=str, required=True,
                      help="Comma-separated domain:class_count pairs")
    
    # Model arguments
    parser.add_argument("--model_name", choices=["eaml", "docformer"], 
                      default="eaml", help="Base model type")
    parser.add_argument("--base_model_path", type=str, required=True,
                      help="Path to base model checkpoint")
    
    # Training arguments
    parser.add_argument("--checkpoint_dir", default="checkpoints/dil")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    
    # DIL strategy
    parser.add_argument("--strategy", choices=["standard", "distillation"],
                      default="distillation")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    
    # Catastrophic forgetting mitigation
    parser.add_argument("--use_ewc", action="store_true")
    parser.add_argument("--lambda_ewc", type=float, default=5000.0)
    parser.add_argument("--use_exemplars", action="store_true")
    parser.add_argument("--max_exemplars", type=int, default=200)
    
    # EVM domain detection
    parser.add_argument("--use_evm", action="store_true")
    parser.add_argument("--evm_tailsize", type=float, default=0.5)
    parser.add_argument("--evm_threshold", type=float, default=0.7)
    
    # Training mode
    parser.add_argument("--training_mode", choices=["head_only", "partial", "full"],
                      default="head_only")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse class counts
    class_counts = {}
    for pair in args.class_counts.split(','):
        domain, count = pair.split(':')
        class_counts[domain] = int(count)
    
    # Initialize metrics
    metrics = DILMetrics()
    
    # Initialize strategy
    strategy = DistillationDomainIL(device, args.temperature, args.lambda_distill) \
               if args.strategy == "distillation" else StandardDomainIL(device)
    
    # Initialize EVM
    evm = EVMDomainClassifier(args.evm_tailsize, args.evm_threshold) if args.use_evm else None
    
    # Setup domains
    domains = [{"name": d, "path": os.path.join(args.data_dir, d)} 
              for d in args.domain_list]
    
    # Get data loaders
    print("Loading datasets...")
    data_loaders = get_all_domains_loaders(domains, args.batch_size)
    
    # Load and prepare base model
    print(f"Loading base model: {args.model_name}")
    model_config = {
        "strategy": args.model_name,
        "eaml_ckpt": args.base_model_path if args.model_name == "eaml" else "",
        "doc_ckpt": args.base_model_path if args.model_name == "docformer" else "",
        "dil_mode": True
    }
    model = prepare_dil_models(model_config, class_per_domain=class_counts, device=device)
    
    # Training loop for each domain
    for idx, (domain_name, loader) in enumerate(data_loaders):
        print(f"\n=== Training on domain: {domain_name} ({idx+1}/{len(data_loaders)}) ===")
        
        # Update metrics tracker
        metrics.add_domain(domain_name)
        
        # Get the current domain's data
        train_loader = loader[0] if isinstance(loader, tuple) else loader
        val_loader = loader[1] if isinstance(loader, tuple) else None
        
        # Setup for this domain
        if idx > 0:
            # Create a copy of the current model for distillation
            old_model = prepare_dil_models(model_config, class_per_domain=class_counts, device=device)
            # Load weights
            old_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.domain_list[idx-1]}.pth")
            if os.path.exists(old_ckpt_path):
                checkpoint = torch.load(old_ckpt_path, map_location=device)
                if "model_state_dict" in checkpoint:
                    old_model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    old_model.load_state_dict(checkpoint)
            old_model.eval()
            
            # Add new domain head
            model = strategy.adapt_model(model, args.domain_list[:idx], domain_name)
            
            # Setup EWC if enabled
            ewc = None
            if args.use_ewc:
                prev_domain = args.domain_list[idx-1]
                prev_loader = data_loaders[idx-1][1]
                ewc = EWC(old_model, prev_loader, device, args.lambda_ewc)
        else:
            old_model = None
            ewc = None
        
        # Setup training mode
        if args.training_mode == "head_only":
            for param in model.base_model.parameters():
                param.requires_grad = False
        elif args.training_mode == "partial":
            # Freeze most layers except last few
            for param in model.base_model.parameters():
                param.requires_grad = False
            # Unfreeze last few layers - this is model specific
            if hasattr(model.base_model, 'encoder'):
                for layer in model.base_model.encoder.layer[-2:]:  # Last 2 layers
                    for param in layer.parameters():
                        param.requires_grad = True
                
        # All parameters trainable in "full" mode (default PyTorch behavior)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr
        )
        
        # Setup learning rate scheduler
        lr_scheduler = AdaptiveLR(optimizer, base_lr=args.lr)
        
        # Resume from checkpoint if needed
        start_epoch = 0
        if args.resume:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{domain_name}.pth")
            if os.path.exists(ckpt_path):
                start_epoch = load_checkpoint_dil(model, optimizer, ckpt_path, device)
                print(f"Resuming from epoch {start_epoch}")
        
        # Training loop
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_metrics = train_one_epoch_dil(
                model, train_loader, domain_name, optimizer, 
                criterion, device, strategy, old_model, ewc
            )
            
            # Evaluate
            if val_loader:
                val_metrics = evaluate_dil(
                    model, val_loader, domain_name, device, metrics, evm
                )
                
                # Update learning rate
                lr_adjusted = lr_scheduler.step(val_metrics)
                if lr_adjusted:
                    print(f"Learning rate adjusted to {lr_scheduler.get_lr()}")
            
            # Save checkpoint
            save_checkpoint_dil(
                model, optimizer, epoch, 
                os.path.join(args.checkpoint_dir, f"{domain_name}_epoch{epoch}.pth")
            )
        
        # Final checkpoint for this domain
        save_checkpoint_dil(
            model, optimizer, args.epochs,
            os.path.join(args.checkpoint_dir, f"{domain_name}.pth")
        )
        
        # Update EVM classifier if enabled
        if args.use_evm:
            print(f"Updating EVM classifier with domain {domain_name}")
            features = extract_features(model, train_loader, device, domain_name)
            
            if idx == 0:
                # First domain, fit EVM from scratch
                evm.fit(features)
            else:
                # Incremental update for new domain
                evm.incremental_update({domain_name: features[domain_name]})
    
    print("Domain Incremental Learning completed successfully!")

if __name__ == "__main__":
    main()

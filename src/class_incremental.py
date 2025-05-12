# src/class_incremental.py
import os
import time
import torch
import torch.nn as nn
from typing import List

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig
from utils.class_IL.dataloader_utils import get_class_il_loader
from utils.class_IL.train_utils import save_checkpoint, load_checkpoint, train_one_epoch, evaluate, CILMetrics
from utils.class_IL.cil_utils import StandardIncremental, RotationAugmentedDistillation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_incremental_learning(
    data_root: str,
    class_order: List[str],
    base_model_path: str,
    model_name: str,
    checkpoint_dir: str,
    start_step: int = 0,
    batch_size: int = 8,
    lr: float = 2e-5,
    num_epochs: int = 10,
    strategy: str = "rad",
    temperature: float = 2.0,
    lambda_distill: float = 1.0
):
    print("Starting Class Incremental Learning...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize metrics tracker
    initial_classes = class_order[:start_step+1] if start_step > 0 else [class_order[0]]
    metrics = CILMetrics(initial_classes)
    
    # Initialize strategy
    if strategy == "rad":
        inc_strategy = RotationAugmentedDistillation(DEVICE, temperature, lambda_distill)
    else:
        inc_strategy = StandardIncremental(DEVICE)
    
    # Initialize best accuracy tracking
    best_acc = 0.0
    
    # Keep track of the previous model for distillation
    old_model = None
    
    for step in range(start_step, len(class_order)):
        current_classes = class_order[:step + 1]
        prev_classes = class_order[:step] if step > 0 else []
        new_classes = [class_order[step]] if step > start_step else []
        
        print(f"Step {step+1}/{len(class_order)} | Training on classes: {current_classes}")
        print(f"New classes: {new_classes}")
        
        # Update metrics with new class information
        if step > start_step:
            metrics.incremental_state_update(new_classes)
        
        # Get dataloaders
        train_loader = get_class_il_loader(
            model_type=model_name,
            data_dir=os.path.join(data_root, "train"),
            current_classes=current_classes,
            batch_size=batch_size
        )
        
        val_loader = get_class_il_loader(
            model_type=model_name,
            data_dir=os.path.join(data_root, "val"),
            current_classes=current_classes,
            batch_size=batch_size
        )
        
        # Initialize model
        if model_name == "docformer":
            config = DocFormerConfig()
            train_dataset = train_loader.dataset
            model = DocFormer(config, num_classes=len(train_dataset.class_to_idx))
            model.to(config.device)
        elif model_name == "eaml":
            model = EAMLModel(num_classes=len(current_classes))
            model.to(DEVICE)
        
        # Load base model weights if first step or adapt model for new classes
        if step == start_step:
            try:
                # Try to load the checkpoint
                checkpoint = torch.load(base_model_path, map_location=DEVICE, weights_only=True)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded base model from {base_model_path}")
            except Exception as e:
                print(f"Error loading model from {base_model_path}: {e}")
                print("Initializing model with random weights.")
        elif step > start_step:
            # Save current model as old model for distillation
            if strategy == "rad":
                old_model = type(model)(config, num_classes=len(prev_classes)) if model_name == "docformer" else EAMLModel(num_classes=len(prev_classes))
                old_model.load_state_dict(model.state_dict(), strict=False)
                old_model.to(DEVICE)
                old_model.eval()
                
                # Adapt model for new classes
                model = inc_strategy.adapt_model(model, prev_classes, new_classes)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            
            model.train()
            total_loss = 0
            all_preds, all_labels = [], []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Compute loss using the selected strategy
                loss, preds, labels = inc_strategy.compute_loss(model, batch, criterion, old_model)
                
                loss.backward()
                optimizer.step()
                
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
                total_loss += loss.item()
            
            # Evaluate
            val_metrics = evaluate(model, val_loader, DEVICE, metrics)
            epoch_time = time.time() - start_time
            
            print(f"Epoch Time: {epoch_time:.2f}s | Val Acc: {val_metrics['top1_acc']:.4f}")
            
            # Save best model based on validation accuracy
            if val_metrics['top1_acc'] > best_acc:
                best_acc = val_metrics['top1_acc']
                # Save checkpoint
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    path=os.path.join(checkpoint_dir, f"step_{step}_class_{class_order[step]}.pth")
                )
        
        # Reset best accuracy for next step
        best_acc = 0.0
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs,
        path=os.path.join(checkpoint_dir, "final_model.pth")
    )
    
    print("Incremental Learning completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--class_order', required=True,
                       help="Comma-separated class order")
    parser.add_argument('--base_model_path', required=True)
    parser.add_argument('--model_name', required=True, choices=['eaml', 'docformer'])
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default="rad", 
                       choices=["standard", "rad"])
    parser.add_argument('--temperature', type=float, default=2.0,
                       help="Temperature for knowledge distillation")
    parser.add_argument('--lambda_distill', type=float, default=1.0,
                       help="Weight for distillation loss")
    
    args = parser.parse_args()
    
    run_incremental_learning(
        data_root=args.data_dir,
        class_order=args.class_order.split(','),
        base_model_path=args.base_model_path,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        start_step=args.start_step,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        strategy=args.strategy,
        temperature=args.temperature,
        lambda_distill=args.lambda_distill
    )

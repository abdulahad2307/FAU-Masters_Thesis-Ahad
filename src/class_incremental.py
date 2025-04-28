import os
import time
import torch
import torch.nn as nn
from typing import List
#from models import load_model
from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig
from utils.class_IL.dataloader_utils import get_class_il_loader
from utils.class_IL.train_utils import save_checkpoint, load_checkpoint, train_one_epoch, evaluate, CILMetrics

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
    num_epochs: int = 10
):
    print("Starting Class Incremental Learning...")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize metrics tracker
    initial_classes = class_order[:start_step+1] if start_step > 0 else [class_order[0]]
    metrics = CILMetrics(initial_classes)

    # Initialize best accuracy tracking
    best_acc = 0.0

    for step in range(start_step, len(class_order)):
        current_classes = class_order[:step + 1]
        new_classes = [class_order[step]] if step > start_step else []
        print(f"Step {step+1}/{len(class_order)} | Training on new classes: {new_classes}")

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
        #model = load_model(model_name=model_name, num_classes=len(current_classes))
        #model.to(DEVICE)
        if model_name == "docformer":
            config = DocFormerConfig()
            train_dataset = train_loader.dataset
            model = DocFormer(config, num_classes=len(train_dataset.class_to_idx))
            model.to(config.device)
        elif model_name == "eaml":
            model = EAMLModel(num_classes=len(class_order))
        
        # Load base model weights if first step
        if step == start_step and os.path.exists(base_model_path):
            model.load_state_dict(torch.load(base_model_path, map_location=DEVICE))
            print(f"Loaded base model from {base_model_path}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()

            #train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, metrics)
            #val_acc = evaluate(model, val_loader, DEVICE)
            val_metrics = evaluate(model, val_loader, DEVICE, metrics)

            epoch_time = time.time() - start_time
            #print(f"Epoch Time: {epoch_time:.2f}s | Val Acc: {val_acc:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s | Val Acc: {val_metrics['top1_acc']:.4f}")
    
            # Save best model based on validation accuracy
            if val_metrics['top1_acc'] > best_acc:
                best_acc = val_metrics['top1_acc']
                # Save checkpoint
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    path=os.path.join(checkpoint_dir, f"{new_classes}_ep_{step}.pth")
                )
        
        # Calculate and print incremental learning gap (G_IL)
        if step > 0:
            full_model_acc = 0.15  # Replace with your full model accuracy
            current_acc = val_metrics['top1_acc']
            G_IL = (current_acc - full_model_acc) / (1 - full_model_acc)
            print(f"Incremental Learning Gap (G_IL): {G_IL:.4f}")
        
        # Reset best accuracy for next step
        best_acc = 0.0

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
        num_epochs=args.num_epochs
    )
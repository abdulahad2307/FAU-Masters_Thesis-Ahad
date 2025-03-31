import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from utils.docformer.model import DocFormer
from utils.docformer.dataset import FUNSDDataset, CORDDataset, collate_fn
from utils.docformer.trainer import DocFormerTrainer
from src.config import DocFormerConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--dataset", type=str, choices=["funsd", "cord"], required=True, help="Dataset to use")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate the model")
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    if os.path.exists(args.config):
        config = DocFormerConfig.from_json(args.config)
    else:
        config = DocFormerConfig()
    
    # Update config with command line arguments
    config.num_train_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    
    # Save config
    config.to_json(os.path.join(output_dir, "config.json"))
    
    # Load dataset
    if args.dataset == "funsd":
        train_dataset = FUNSDDataset(args.data_dir, split="train")
        eval_dataset = FUNSDDataset(args.data_dir, split="test")
    elif args.dataset == "cord":
        train_dataset = CORDDataset(args.data_dir, split="train")
        eval_dataset = CORDDataset(args.data_dir, split="test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = DocFormer(config)
    
    # Initialize trainer
    trainer = DocFormerTrainer(model, config, args.device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    if args.eval_only:
        eval_loss = trainer.evaluate(eval_loader)
        print(f"Evaluation Loss: {eval_loss:.4f}")
        return
    
    # Training loop
    best_loss = float("inf")
    for epoch in range(start_epoch, config.num_train_epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        eval_loss = trainer.evaluate(eval_loader)
        
        print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(output_dir, epoch)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            trainer.save_checkpoint(output_dir, epoch, best=True)
    
    print(f"Training complete. Best evaluation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
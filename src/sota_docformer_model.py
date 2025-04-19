import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils.docformer.dataset import RVLCDIPDataset, collate_fn
from utils.docformer.model import DocFormer
from utils.docformer.trainer import DocFormerTrainer
from utils.docformer.config import DocFormerConfig

def main():
    parser = argparse.ArgumentParser(description="DocFormer for Document Classification")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2.5e-5, help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--classes', type=str, default=None, 
                       help='Comma-separated list of classes to include (e.g. "letter,form,email")')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--resume', type=str, help='Path to model checkpoint')
    args = parser.parse_args()

    # Parse classes if provided
    class_list = [c.strip() for c in args.classes.split(',')] if args.classes else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize datasets with class filtering
    train_dataset = RVLCDIPDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='train',
        classes=class_list
    )
    
    val_dataset = RVLCDIPDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='val',
        classes=class_list
    )

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Initialize model and trainer
    config = DocFormerConfig()
    model = DocFormer(config, num_classes=len(train_dataset.class_to_idx))
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.resume}")

    trainer = DocFormerTrainer(
        model=model,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if not args.eval_only:
        # Initialize tracking variables
        min_val_loss = float('inf')
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(args.num_epochs):
            # Train for one epoch
            train_loss = trainer.train_epoch(train_loader, epoch)
            
            # Evaluate on validation set
            val_loss, val_acc = trainer.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint if validation loss improves
            is_best = val_loss < min_val_loss
            if is_best:
                min_val_loss = val_loss
                best_val_acc = val_acc
                
            trainer.save_checkpoint(args.output_dir, epoch, best=is_best)
    
    # Final evaluation
    val_loss, val_acc = trainer.evaluate(val_loader)
    print(f"\nFinal Results - Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"Best Val Loss: {min_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
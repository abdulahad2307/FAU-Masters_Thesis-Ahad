import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils.docformer.docformer_dataloader import DocFormerDataset, docformer_collate_fn
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
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--resume', type=str, help='Path to model checkpoint')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize datasets using the new DocFormerDataset
    train_dataset = DocFormerDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='train'
    )
    
    val_dataset = DocFormerDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='val'
    )

    # Initialize data loaders with docformer_collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=docformer_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=docformer_collate_fn
    )

    # Initialize model and trainer (rest of your existing code)
    config = DocFormerConfig()
    model = DocFormer(config)
    
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
        # Training loop
        for epoch in range(args.num_epochs):
            train_loss = trainer.train_epoch(train_loader, epoch)
            val_loss = trainer.evaluate(val_loader)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            trainer.save_checkpoint(args.output_dir, epoch, best=(val_loss == min_val_loss))
    
    # Evaluation
    val_loss = trainer.evaluate(val_loader)
    print(f"Final Validation Loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()
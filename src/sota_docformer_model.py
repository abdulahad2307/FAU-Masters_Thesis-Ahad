#!/usr/bin/env python3
import argparse
import os
import time
import torch
import logging
from torch.utils.data import DataLoader
from utils.docformer.dataset import RVLCDIPDataset, collate_fn
from utils.docformer.model import DocFormer
from utils.docformer.trainer import DocFormerTrainer
from utils.docformer.config import DocFormerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("docformer_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Set multiprocessing start method
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    start_time = time.time()
    logger.info("=== DocFormer Training ===")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="DocFormer for Document Understanding")
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', default='docformer_outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=2.5e-5)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--classes', help='Comma-separated list of classes')
    parser.add_argument('--resume', help='Path to model checkpoint')
    args = parser.parse_args()

    # Initialize config
    config = DocFormerConfig()
    config.batch_size = args.batch_size
    config.num_train_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.output_dir = args.output_dir
    config.print_config()

    # Parse classes
    class_list = [c.strip() for c in args.classes.split(',')] if args.classes else None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize datasets
    logger.info("Initializing datasets...")
    train_dataset = RVLCDIPDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='train',
        classes=class_list,
        config=config
    )
    
    val_dataset = RVLCDIPDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='val',
        classes=class_list,
        config=config
    )

    # Initialize data loaders
    logger.info("Initializing data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize model
    logger.info("Initializing model...")
    model = DocFormer(config, num_classes=len(train_dataset.class_to_idx))
    model.to(config.device)
    
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = DocFormerTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        eval_loader=val_loader
    )

    # Training loop
    logger.info("Starting training...")
    try:
        training_results = trainer.train(args.output_dir)
        logger.info("\n=== Training Completed ===")
        logger.info(f"Best Val Loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"Best Val Acc: {training_results['best_val_acc']:.2f}%")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    logger.info(f"\nTotal execution time: {(time.time()-start_time)/60:.2f} minutes")

if __name__ == '__main__':
    main()
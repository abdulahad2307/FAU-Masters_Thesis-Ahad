# main.py (Complete Training Script)
import argparse
import os
import time
import torch
import logging
from torch.utils.data import DataLoader
from utils.docformer.config import DocFormerConfig
from utils.docformer.dataset import RVLCDIPDataset
from utils.docformer.model import DocFormer
from utils.docformer.trainer import DocFormerTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("docformer_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="DocFormer Training")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='docformer_outputs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=2.5e-5)
    parser.add_argument('--ocr_engine', default='tesseract', choices=['tesseract', 'trocr', 'pero'])
    parser.add_argument('--phase', default='pretrain', choices=['pretrain', 'finetune'])
    args = parser.parse_args()

    config = DocFormerConfig()
    config.ocr_engine = args.ocr_engine
    config.phase = args.phase
    config.batch_size = args.batch_size
    config.num_train_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.print_config()

    # Initialize datasets
    train_dataset = RVLCDIPDataset(config, args.data_dir, 'train')
    val_dataset = RVLCDIPDataset(config, args.data_dir, 'val')
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = DocFormer(config, num_classes=len(train_dataset.class_to_idx))
    
    # Trainer
    trainer = DocFormerTrainer(model, config, train_loader, val_loader)
    
    # Training
    try:
        results = trainer.train(args.output_dir)
        logger.info(f"Training completed. Best Val Acc: {results['best_val_acc']:.2f}%")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    logger.info(f"Total execution time: {(time.time()-start_time)/60:.2f} minutes")

if __name__ == '__main__':
    main()

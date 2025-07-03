import argparse
import os
import time
import torch
import logging
from torch.utils.data import DataLoader
from utils.docformer.config import DocFormerConfig
from utils.docformer.dataset import RVLCDIPDataset, collate_fn
from utils.docformer.model import DocFormer
from utils.docformer.trainer import DocFormerTrainer
from utils.docformer.evaluator import DocFormerEvaluator

# Set optimizations for better performance
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

def parse_comma_separated_list(value):
    """Parse comma-separated string into list"""
    if not value:
        return None
    return [item.strip() for item in value.split(',')]

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def validate_args(args):
    """Validate command line arguments"""
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    
    if args.resume and not os.path.exists(args.resume):
        raise ValueError(f"Resume checkpoint {args.resume} does not exist")
    
    if args.classes and len(args.classes) < 2:
        raise ValueError("At least 2 classes required for training")

def main():
    # Set multiprocessing method for compatibility
    torch.multiprocessing.set_start_method('spawn', force=True)
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Progressive DocFormer Training")
    
    # Required arguments
    parser.add_argument('--data_dir', required=True, 
                       help='Path to dataset directory')
    
    # Training configuration - Paper specifications
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                       help='Learning rate for pre-training (Paper: 5e-5)')
    parser.add_argument('--finetune_lr', type=float, default=2.5e-5,
                       help='Learning rate for fine-tuning (Paper: 2.5e-5)')
    parser.add_argument('--batch_size', type=int, default=9, 
                       help='Batch size for pre-training (Paper: 9)')
    parser.add_argument('--finetune_batch_size', type=int, default=4,
                       help='Batch size for fine-tuning (Paper: 4)')
    parser.add_argument('--num_epochs', type=int, default=30, 
                       help='Total number of training epochs')
    
    # Progressive training epochs
    parser.add_argument('--text_epochs', type=int, default=10,
                       help='Epochs for text-only pre-training')
    parser.add_argument('--visual_epochs', type=int, default=5,
                       help='Epochs for visual pre-training')
    parser.add_argument('--final_epochs', type=int, default=15,
                       help='Epochs for final fine-tuning')
    
    # OCR engine selection
    parser.add_argument('--ocr_engine', default='trocr', 
                       choices=['tesseract', 'trocr', 'pero', 'easyocr', 'paddleocr'], 
                       help='OCR engine to use')
    
    # Model configuration
    parser.add_argument('--output_dir', default=None, 
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--max_seq_length', type=int, default=512, 
                       help='Maximum sequence length (Paper: 512)')
    parser.add_argument('--classes', type=parse_comma_separated_list, 
                       help='Comma-separated list of classes to train on')
    
    # Training stages
    parser.add_argument('--training_stage', default='text_pretrain',
                       choices=['text_pretrain', 'multimodal_pretrain', 'finetune'],
                       help='Training stage to start from')
    parser.add_argument('--phase', default='finetune',
                       choices=['pretrain', 'finetune'],
                       help='Training phase (pretrain or finetune)')
    
    # Checkpointing and evaluation
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only run evaluation on best model')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Progressive training options
    parser.add_argument('--progressive', action='store_true', default=True,
                       help='Use progressive training strategy')
    parser.add_argument('--skip_pretrain', action='store_true',
                       help='Skip pre-training and go directly to fine-tuning')
    
    # Loss weights (Paper specifications)
    parser.add_argument('--mm_mlm_weight', type=float, default=5.0,
                       help='Multi-modal MLM loss weight (Paper: λ=5)')
    parser.add_argument('--ltr_weight', type=float, default=1.0,
                       help='Learn-to-reconstruct loss weight (Paper: β=1)')
    parser.add_argument('--tdi_weight', type=float, default=5.0,
                       help='Text-describes-image loss weight (Paper: γ=5)')
    
    # Hardware optimization
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--device', default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Initialize configuration with paper specifications
    config = DocFormerConfig()
    
    # Override with command line arguments
    config.learning_rate = args.learning_rate
    config.finetune_lr = args.finetune_lr
    config.batch_size = args.batch_size
    config.finetune_bs = args.finetune_batch_size
    config.num_train_epochs = args.num_epochs
    config.ocr_engine = args.ocr_engine
    config.training_stage = args.training_stage
    config.phase = args.phase
    config.num_workers = args.num_workers
    config.use_amp = args.use_amp
    config.save_interval = args.save_interval
    
    # Progressive training epochs
    config.text_stage_epochs = args.text_epochs
    config.visual_stage_epochs = args.visual_epochs
    config.final_stage_epochs = args.final_epochs
    
    # Loss weights
    config.mm_mlm_weight = args.mm_mlm_weight
    config.ltr_weight = args.ltr_weight
    config.tdi_weight = args.tdi_weight
    
    # Set device
    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        config.device = "cpu"
        print("Using CPU")
    
    # Set output directory
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        timestamp = int(time.time())
        config.output_dir = f"docformer_outputs_{config.ocr_engine}_{timestamp}"
    
    # Set sequence length
    if args.max_seq_length:
        config.max_position_embeddings = args.max_seq_length
    
    # Setup logging
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    
    # Scale loss weights for initial stage
    config.scale_weights(config.training_stage)
    config.print_config()
    
    # Save configuration
    config.save(config.output_dir)
    
    logger.info("Initializing datasets...")
    try:
        # Initialize datasets
        train_dataset = RVLCDIPDataset(config, args.data_dir, 'train')
        val_dataset = RVLCDIPDataset(config, args.data_dir, 'val')
        test_dataset = RVLCDIPDataset(config, args.data_dir, 'test')
        
        logger.info(f"Loaded datasets - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Filter classes if specified
        if args.classes:
            logger.info(f"Filtering to classes: {args.classes}")
            train_dataset.filter_classes(args.classes)
            val_dataset.filter_classes(args.classes)
            test_dataset.filter_classes(args.classes)
        
        # Create class mappings
        train_dataset.create_class_mappings()
        val_dataset.class_to_idx = train_dataset.class_to_idx
        val_dataset.idx_to_class = train_dataset.idx_to_class
        test_dataset.class_to_idx = train_dataset.class_to_idx
        test_dataset.idx_to_class = train_dataset.idx_to_class
        
        num_classes = len(train_dataset.class_to_idx)
        logger.info(f"Number of classes: {num_classes}")
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {str(e)}")
        return 1
    
    # Create data loaders with appropriate batch sizes
    current_batch_size = config.finetune_bs if config.training_stage == 'finetune' else config.batch_size
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=current_batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config.num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.eval_batch_size,
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.eval_batch_size,
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    logger.info("Initializing model...")
    try:
        # Initialize model
        model = DocFormer(config, num_classes=num_classes)
        model.to(config.device)
        
        # Load checkpoint if resuming
        if args.resume and os.path.exists(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
            
            # Only load scaler state if it exists and scaler is available
            #if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                #if hasattr(trainer, 'scaler') and trainer.scaler is not None:
                    #trainer.scaler.load_state_dict(checkpoint['scaler_state_dict']
                    
            # Update training stage if specified in checkpoint
            if 'training_stage' in checkpoint:
                config.training_stage = checkpoint['training_stage']
                logger.info(f"Resumed from training stage: {config.training_stage}")
        
        # Print model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return 1
    
    # Evaluation only mode
    if args.evaluate_only:
        logger.info("Running evaluation only...")
        try:
            # Load best model for evaluation
            best_model_path = os.path.join(config.output_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                logger.info(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=config.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                logger.warning("Best model not found. Using current model state.")
            
            # Run evaluation
            evaluator = DocFormerEvaluator(model, config, test_loader, 
                                         list(train_dataset.class_to_idx.keys()))
            metrics = evaluator.evaluate_model(config.output_dir)
            
            # Compare with paper results (RVL-CDIP benchmark)
            paper_accuracy = 96.17  # Paper's reported accuracy
            our_accuracy = metrics['accuracy'] * 100
            logger.info(f"Paper's RVL-CDIP accuracy: {paper_accuracy:.2f}%")
            logger.info(f"Our accuracy: {our_accuracy:.2f}%")
            logger.info(f"Difference: {(our_accuracy - paper_accuracy):.2f}%")
            
            return 0
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return 1
    
    # Training mode
    logger.info("Initializing trainer...")
    try:
        trainer = DocFormerTrainer(model, config, train_loader, val_loader)
        
        logger.info("Starting training...")
        if args.progressive and not args.skip_pretrain:
            # Progressive training through all stages
            results = trainer.progressive_train(config.output_dir)
        else:
            # Standard training (single stage)
            results = trainer.train(config.output_dir)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        
        # Automatic evaluation after training
        logger.info("Starting final evaluation...")
        evaluator = DocFormerEvaluator(model, config, test_loader, 
                                     list(train_dataset.class_to_idx.keys()))
        metrics = evaluator.evaluate_model(config.output_dir)
        
        # Compare with paper results
        paper_accuracy = 96.17
        our_accuracy = metrics['accuracy'] * 100
        logger.info(f"Final Results:")
        logger.info(f"Paper's RVL-CDIP accuracy: {paper_accuracy:.2f}%")
        logger.info(f"Our test accuracy: {our_accuracy:.2f}%")
        logger.info(f"Performance gap: {(our_accuracy - paper_accuracy):.2f}%")
        
        # Save final results
        final_results = {
            'training_results': results,
            'test_metrics': metrics,
            'paper_comparison': {
                'paper_accuracy': paper_accuracy,
                'our_accuracy': our_accuracy,
                'gap': our_accuracy - paper_accuracy
            },
            'config': config.__dict__
        }
        
        import json
        with open(os.path.join(config.output_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")
    logger.info(f"Results saved to: {config.output_dir}")
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)

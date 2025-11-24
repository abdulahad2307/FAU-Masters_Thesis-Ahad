import json
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

@dataclass
class DocFormerConfig:
    # Model Architecture - Paper compliant
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    
    # OCR Configuration
    ocr_engine: str = "trocr"
    ocr_cache_dir: str = "ocr_cache"
    ocr_batch_size: int = 16
    ocr_confidence_threshold: float = 0.6
    ocr_device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Image Processing
    ocr_image_size: int = 384
    ocr_resample: int = 2
    
    # Visual Backbone
    visual_backbone: str = "resnet50"
    visual_feature_dim: int = 2048
    visual_output_dim: int = 768
    visual_pool_size: int = 16
    
    # Training Parameters - Fixed for better performance
    phase: str = "finetune"
    training_stage: str = "text_pretrain"
    learning_rate: float = 5e-5  # Paper specification
    finetune_lr: float = 2.5e-5  # Paper specification
    batch_size: int = 9          # Paper: 9 for pretraining
    finetune_bs: int = 4         # Paper: 4 for fine-tuning
    num_train_epochs: int = 30   # Total epochs
    num_workers: int = 4
    
    # Progressive Training Epochs - Increased for better learning
    text_stage_epochs: int = 10   # Increased from 3
    visual_stage_epochs: int = 10 # Increased from 2
    final_stage_epochs: int = 20  # Increased from 3
    
    # Optimization - Fixed learning rate issues
    use_amp: bool = False        # Disabled to avoid GradScaler issues
    use_gradient_checkpointing: bool = True
    max_grad_norm: float = 0.5   # Reduced for stability
    warmup_ratio: float = 0.1    # 10% warmup
    warmup_steps: int = 1000     # Fixed warmup steps as per paper
    weight_decay: float = 0.01
    save_interval: int = 5
    
    # Learning Rate Scheduler Configuration
    scheduler_type: str = "linear"  # Use linear warmup instead of OneCycleLR
    min_lr: float = 1e-6           # Minimum learning rate
    lr_decay_factor: float = 0.5   # For ReduceLROnPlateau
    lr_patience: int = 3           # Patience for LR reduction
    
    # Loss Weights - Fixed for proper progressive training
    mm_mlm_weight: float = 5.0
    ltr_weight: float = 1.0
    tdi_weight: float = 5.0
    
    # Pre-training Tasks
    pretrain_tasks: List[str] = field(default_factory=lambda: ["mm_mlm", "ltr", "tdi"])
    mm_mlm_probability: float = 0.15
    tdi_negative_probability: float = 0.2
    
    # Evaluation
    eval_batch_size: int = 8
    eval_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    
    # System
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    output_dir: str = "docformer_outputs"
    
    # Checkpoint Management
    save_total_limit: int = 3     # Keep last 3 checkpoints
    save_best_model: bool = True  # Always keep best model
    cleanup_checkpoints: bool = True  # Enable automatic cleanup

    def __post_init__(self):
        """Paper-compliant validation and setup"""
        self._validate_architecture()
        self._validate_training_params()
        self._setup_stage_specific_config()

    def _validate_architecture(self):
        assert self.visual_backbone == "resnet50", "Paper uses ResNet50"
        assert self.hidden_size == 768, "Base model config"
        assert self.num_hidden_layers == 12, "Base model config"
        assert self.num_attention_heads == 12, "Base model config"

    def _validate_training_params(self):
        assert self.num_workers > 0, "num_workers must be positive"
        assert self.warmup_ratio >= 0 and self.warmup_ratio <= 1, "Invalid warmup ratio"
        assert self.ocr_engine in ["tesseract", "trocr", "pero", "easyocr", "paddleocr"]
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.finetune_lr > 0, "Fine-tune learning rate must be positive"

    def _setup_stage_specific_config(self):
        """Setup stage-specific configurations"""
        # Ensure proper batch sizes for each stage
        if self.training_stage in ["text_pretrain", "multimodal_pretrain"]:
            self.current_batch_size = self.batch_size
            self.current_lr = self.learning_rate
        else:  # finetune
            self.current_batch_size = self.finetune_bs
            self.current_lr = self.finetune_lr

    def scale_weights(self, stage: str):
        """Adjust loss weights for progressive training stages - FIXED"""
        weight_settings = {
            "text_pretrain": (0.0, 0.0, 0.0),      # Only classification loss
            "multimodal_pretrain": (5.0, 1.0, 5.0), # All pretraining losses
            "finetune": (0.0, 0.0, 0.0)             # Only classification loss
        }
        if stage in weight_settings:
            self.mm_mlm_weight, self.ltr_weight, self.tdi_weight = weight_settings[stage]
            print(f"Updated loss weights for {stage}: MM-MLM={self.mm_mlm_weight}, LTR={self.ltr_weight}, TDI={self.tdi_weight}")
        else:
            print(f"Warning: Unknown stage {stage}, keeping current weights")

    def get_learning_rate_for_stage(self, stage: str) -> float:
        """Get appropriate learning rate for training stage"""
        lr_mapping = {
            "text_pretrain": self.learning_rate,
            "multimodal_pretrain": self.learning_rate,
            "finetune": self.finetune_lr
        }
        return lr_mapping.get(stage, self.learning_rate)

    def get_batch_size_for_stage(self, stage: str) -> int:
        """Get appropriate batch size for training stage"""
        if stage == "finetune":
            return self.finetune_bs
        else:
            return self.batch_size

    def get_epochs_for_stage(self, stage: str) -> int:
        """Get number of epochs for training stage"""
        epoch_mapping = {
            "text_pretrain": self.text_stage_epochs,
            "multimodal_pretrain": self.visual_stage_epochs,
            "finetune": self.final_stage_epochs
        }
        return epoch_mapping.get(stage, 5)

    def reset_for_stage(self, stage: str):
        """Reset configuration for new training stage"""
        self.training_stage = stage
        self.scale_weights(stage)
        self.current_batch_size = self.get_batch_size_for_stage(stage)
        self.current_lr = self.get_learning_rate_for_stage(stage)
        print(f"Reset config for stage: {stage}")
        print(f"  - Learning rate: {self.current_lr}")
        print(f"  - Batch size: {self.current_batch_size}")
        print(f"  - Epochs: {self.get_epochs_for_stage(stage)}")

    def save(self, path: str):
        """Save configuration with better error handling"""
        try:
            config_dict = asdict(self)
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(Path(path)/"config.json", "w") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {path}/config.json")
        except Exception as e:
            print(f"Failed to save configuration: {e}")

    @classmethod
    def load(cls, path: str):
        """Load configuration with better error handling"""
        try:
            with open(Path(path)/"config.json", "r") as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return cls()  # Return default config

    def print_config(self):
        """Enhanced configuration printout"""
        print("=== DocFormer Configuration ===")
        print(f"Model Architecture: {self.hidden_size}d, {self.num_hidden_layers} layers, {self.num_attention_heads} heads")
        print(f"Training Stage: {self.training_stage}")
        print(f"Learning Rates: pretrain={self.learning_rate}, finetune={self.finetune_lr}")
        print(f"Batch Sizes: pretrain={self.batch_size}, finetune={self.finetune_bs}")
        print(f"Progressive Epochs: text={self.text_stage_epochs}, visual={self.visual_stage_epochs}, final={self.final_stage_epochs}")
        print(f"OCR: engine={self.ocr_engine}, device={self.ocr_device}, cache_dir={self.ocr_cache_dir}")
        print(f"Loss Weights: MM-MLM={self.mm_mlm_weight}, LTR={self.ltr_weight}, TDI={self.tdi_weight}")
        print(f"Optimization: amp={self.use_amp}, grad_clip={self.max_grad_norm}, warmup={self.warmup_ratio}")
        print(f"System: device={self.device}, workers={self.num_workers}")
        print("==============================")

    def get_training_summary(self):
        """Get training configuration summary"""
        total_epochs = self.text_stage_epochs + self.visual_stage_epochs + self.final_stage_epochs
        return {
            "total_epochs": total_epochs,
            "stages": {
                "text_pretrain": self.text_stage_epochs,
                "multimodal_pretrain": self.visual_stage_epochs,
                "finetune": self.final_stage_epochs
            },
            "learning_rates": {
                "pretrain": self.learning_rate,
                "finetune": self.finetune_lr
            },
            "batch_sizes": {
                "pretrain": self.batch_size,
                "finetune": self.finetune_bs
            }
        }

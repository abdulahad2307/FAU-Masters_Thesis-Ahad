import json
import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DocFormerConfig:
    # Model architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    
    # OCR configuration
    ocr_engine: str = "tesseract"  # [tesseract, trocr, pero]
    ocr_cache_dir: str = "ocr_cache"
    ocr_batch_size: int = 16
    ocr_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Visual backbone
    visual_backbone: str = "resnet50"
    visual_feature_dim: int = 2048
    visual_output_dim: int = 768
    
    # Training phases
    phase: str = "pretrain"  # [pretrain, finetune]
    
    # Pre-training
    pretrain_tasks: list = ["mm_mlm", "ltr", "tdi"]
    mm_mlm_probability: float = 0.15
    tdi_negative_probability: float = 0.2
    
    # Optimization
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    batch_size: int = 8
    num_workers: int = 4
    max_grad_norm: float = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    def print_config(self):
        print("=== DocFormer Configuration ===")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("==============================")

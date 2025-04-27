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
    
    # Visual backbone
    visual_backbone: str = "resnet50"
    visual_feature_dim: int = 2048
    visual_output_dim: int = 768
    
    # Training
    learning_rate: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 8
    num_train_epochs: int = 5
    max_grad_norm: float = 1.0
    checkpoint_keep_last: int = 3
    
    # Pre-training tasks weights (λ=5, β=1, γ=5 from paper)
    mm_mlm_weight: float = 5.0  # λ
    ltr_weight: float = 1.0      # β
    tdi_weight: float = 5.0      # γ
    
    # Pre-training tasks probabilities
    mm_mlm_probability: float = 0.15
    tdi_negative_probability: float = 0.2  # 20% negative pairs
    
    # Logging
    logging_steps: int = 50
    save_steps: int = 500
    
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
import torch
import numpy as np
from typing import Dict, List, Optional

class AdaptiveLR:
    """Adaptive learning rate scheduler for class incremental learning"""
    def __init__(self, optimizer, base_lr: float = 0.001, min_lr: float = 1e-6, 
                 decay_factor: float = 0.75, patience: int = 3):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.best_acc = 0
        self.no_improvement_count = 0
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
    
    def step(self, val_acc: float) -> bool:
        """Update learning rate based on validation accuracy"""
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            
            if self.no_improvement_count >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * self.decay_factor, self.min_lr)
                
                self.no_improvement_count = 0
                return True
            
            return False
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def reset(self, task_complexity: float = 1.0):
        """Reset scheduler for a new task with optional complexity adjustment"""
        self.best_acc = 0
        self.no_improvement_count = 0
        
        # Adjust learning rate based on task complexity
        adjusted_lr = self.base_lr / (1 + task_complexity)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr

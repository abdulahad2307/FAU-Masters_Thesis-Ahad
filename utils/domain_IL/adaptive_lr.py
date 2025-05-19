import torch

class AdaptiveLR:
    """Adaptive learning rate scheduler for domain incremental learning"""
    def __init__(self, optimizer, base_lr=2e-5, min_lr=1e-7, factor=0.5, patience=3):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.best_acc = 0
        self.best_loss = float('inf')
        self.wait_count = 0
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
    
    def step(self, val_metrics):
        """
        Update learning rate based on validation performance
        Args:
            val_metrics: Dict containing 'accuracy' and/or 'loss'
        """
        updated = False
        
        # Check if performance improved
        if 'accuracy' in val_metrics and val_metrics['accuracy'] > self.best_acc:
            self.best_acc = val_metrics['accuracy']
            self.wait_count = 0
        elif 'loss' in val_metrics and val_metrics['loss'] < self.best_loss:
            self.best_loss = val_metrics['loss']
            self.wait_count = 0
        else:
            self.wait_count += 1
            
            # Reduce learning rate if no improvement for patience epochs
            if self.wait_count >= self.patience:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                    if new_lr != param_group['lr']:
                        param_group['lr'] = new_lr
                        updated = True
                
                self.wait_count = 0
        
        return updated
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def reset(self):
        """Reset scheduler for a new domain"""
        self.best_acc = 0
        self.best_loss = float('inf')
        self.wait_count = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

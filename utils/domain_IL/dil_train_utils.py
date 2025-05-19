import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

class DILMetrics:
    """Metrics tracking for Domain Incremental Learning"""
    def __init__(self):
        self.domains = []
        self.accuracies = {}
        self.losses = {}
        self.confusion_matrices = {}
        
    def add_domain(self, domain):
        """Add a new domain to track metrics for"""
        if domain not in self.domains:
            self.domains.append(domain)
            
    def update(self, domain, preds, labels, loss=None):
        """Update metrics for a domain"""
        # Convert to numpy arrays if tensors
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        # Compute accuracy
        correct = (preds == labels).sum()
        total = len(labels)
        accuracy = correct / total if total > 0 else 0
        
        # Update accuracies
        if domain not in self.accuracies:
            self.accuracies[domain] = []
        self.accuracies[domain].append(accuracy)
        
        # Update losses
        if loss is not None:
            if domain not in self.losses:
                self.losses[domain] = []
            self.losses[domain].append(loss)
            
        # Update confusion matrix
        if domain not in self.confusion_matrices:
            num_classes = max(max(labels) + 1, max(preds) + 1) if len(labels) > 0 and len(preds) > 0 else 1
            self.confusion_matrices[domain] = np.zeros((num_classes, num_classes))
            
        # Extend confusion matrix if needed
        cm = self.confusion_matrices[domain]
        max_idx = max(max(labels) + 1, max(preds) + 1) if len(labels) > 0 and len(preds) > 0 else 1
        if max_idx > cm.shape[0]:
            new_cm = np.zeros((max_idx, max_idx))
            new_cm[:cm.shape[0], :cm.shape[1]] = cm
            self.confusion_matrices[domain] = new_cm
            cm = new_cm
            
        # Update confusion matrix
        for i in range(len(labels)):
            cm[labels[i], preds[i]] += 1
            
    def get_metrics(self):
        """Get aggregated metrics"""
        metrics = {
            'domains': self.domains,
            'accuracies': {},
            'confusion_matrices': self.confusion_matrices
        }
        
        # Compute average accuracy for each domain
        for domain in self.domains:
            if domain in self.accuracies:
                metrics['accuracies'][domain] = np.mean(self.accuracies[domain])
                
        # Compute average accuracy across all domains
        all_accs = [acc for accs in self.accuracies.values() for acc in accs]
        metrics['mean_accuracy'] = np.mean(all_accs) if all_accs else 0
        
        # Compute average loss if available
        if self.losses:
            metrics['losses'] = {}
            for domain in self.domains:
                if domain in self.losses:
                    metrics['losses'][domain] = np.mean(self.losses[domain])
            
            all_losses = [loss for losses in self.losses.values() for loss in losses]
            metrics['mean_loss'] = np.mean(all_losses) if all_losses else 0
            
        return metrics

def train_one_epoch_dil(
    model, 
    dataloader, 
    domain, 
    optimizer, 
    criterion, 
    device,
    strategy=None,
    old_model=None,
    ewc=None,
    metrics=None
):
    """Train model for one epoch on domain data"""
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Training on {domain}")
    for batch in pbar:
        optimizer.zero_grad()
        
        # Handle different batch formats
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
            batch = (inputs, domain, labels)
        elif isinstance(batch, dict):
            inputs = batch
            labels = batch['labels'] if 'labels' in batch else batch['targets']
            batch = (inputs, domain, labels)
            
        # Use strategy if provided, otherwise direct forward pass
        if strategy:
            loss, preds, labels = strategy.compute_loss(model, batch, criterion, old_model, ewc)
        else:
            inputs, batch_domain, labels = batch
            
            # Move inputs to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                         {k2: v2.to(device) for k2, v2 in v.items()} 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
                
            labels = labels.to(device)
            
            # Forward pass
            logits = model(inputs, batch_domain)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Add EWC regularization if available
            if ewc is not None:
                loss += ewc.penalty(model)
                
            preds = torch.argmax(logits, dim=1)
            
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': total_correct / total_samples if total_samples > 0 else 0
        })
        
    # Compute epoch metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    # Update metrics if provided
    if metrics is not None:
        metrics.update(domain, all_preds, all_labels, avg_loss)
        
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc
    }

def evaluate_dil(
    model, 
    dataloader, 
    domain, 
    device,
    metrics=None,
    evm_classifier=None
):
    """Evaluate model on domain data"""
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating on {domain}")
        for batch in pbar:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
            elif isinstance(batch, dict):
                inputs = batch
                labels = batch['labels'] if 'labels' in batch else batch['targets']
            else:
                inputs, labels = batch[0], batch[1]
                
            # Move inputs to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                         {k2: v2.to(device) for k2, v2 in v.items()} 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
                
            labels = labels.to(device)
            
            # Domain detection with EVM if available
            if evm_classifier is not None:
                # Extract features
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(inputs, domain)
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'forward_features'):
                    features = model.base_model.forward_features(inputs)
                else:
                    features = None
                    
                if features is not None:
                    # Domain prediction
                    domain_preds = evm_classifier.predict(features.cpu().numpy())
                    # Count correct domain predictions
                    correct_domains = sum(1 for d in domain_preds if d == domain)
                    domain_acc = correct_domains / len(domain_preds) if domain_preds else 0
                    print(f"Domain detection accuracy: {domain_acc:.4f}")
            
            # Forward pass
            logits = model(inputs, domain)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Compute metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    # Update metrics if provided
    if metrics is not None:
        metrics.update(domain, all_preds, all_labels, avg_loss)
        
    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc
    }

def save_checkpoint_dil(model, optimizer, epoch, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare state dict based on model type
    if isinstance(model, dict):  # Ensemble model
        model_state = {k: v.state_dict() for k, v in model.items()}
    else:
        model_state = model.state_dict()
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    
    print(f"Checkpoint saved to {path}")

def load_checkpoint_dil(model, optimizer, path, device):
    """Load model checkpoint"""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state based on model type
    if isinstance(model, dict):  # Ensemble model
        for name, m in model.items():
            if name in checkpoint['model_state_dict']:
                m.load_state_dict(checkpoint['model_state_dict'][name])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {path} at epoch {checkpoint['epoch']}")
    
    return checkpoint['epoch']

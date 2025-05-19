import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

class DomainIncrementalStrategy:
    """Base class for domain incremental learning strategies"""
    def __init__(self, device):
        self.device = device
        
    def adapt_model(self, model, old_domains, new_domain):
        """Adapt model for new domain"""
        if not hasattr(model, 'add_domain_head'):
            raise ValueError("Model must implement add_domain_head method")
        model.add_domain_head(new_domain)
        return model
        
    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        """Compute loss with strategy-specific components"""
        inputs, domain, labels = batch
        
        # Move inputs to device based on structure
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else 
                     {k2: v2.to(self.device) for k2, v2 in v.items()} 
                     for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
            
        labels = labels.to(self.device)
        
        # Forward pass
        logits = model(inputs, domain)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Add EWC regularization if available
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_loss
            
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels

class StandardDomainIL(DomainIncrementalStrategy):
    """Standard domain incremental learning"""
    pass  # Base implementation is sufficient

class DistillationDomainIL(DomainIncrementalStrategy):
    """Domain incremental learning with knowledge distillation"""
    def __init__(self, device, temperature=2.0, lambda_distill=1.0):
        super().__init__(device)
        self.temperature = temperature
        self.lambda_distill = lambda_distill
        
    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        """Compute loss with distillation component"""
        inputs, domain, labels = batch
        
        # Move inputs to device based on structure
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else 
                     {k2: v2.to(self.device) for k2, v2 in v.items()} 
                     for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
            
        labels = labels.to(self.device)
        
        # Forward pass
        logits = model(inputs, domain)
        
        # Classification loss
        cls_loss = criterion(logits, labels)
        
        # Distillation loss if we have an old model
        if old_model is not None:
            with torch.no_grad():
                old_logits = old_model(inputs, domain)
            
            # Get soft targets from old model
            soft_targets = nn.functional.softmax(old_logits / self.temperature, dim=1)
            
            # Get soft probabilities from current model
            soft_probs = nn.functional.log_softmax(logits / self.temperature, dim=1)
            
            # Calculate distillation loss
            dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
            
            # Combined loss
            loss = cls_loss + self.lambda_distill * dist_loss
        else:
            loss = cls_loss
            
        # Add EWC regularization if available
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_loss
            
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels

class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting"""
    def __init__(self, model, dataloader, device, lambda_ewc=5000.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}  # Store parameter values
        self._fisher = {}  # Store Fisher information matrix diagonals
        
        # Compute Fisher information matrix
        self._compute_fisher(dataloader)
        
        # Store current parameter values
        for n, p in self.params.items():
            self._means[n] = p.data.clone()
    
    def _compute_fisher(self, dataloader):
        """Compute Fisher Information Matrix for parameters"""
        # Initialize Fisher information for each parameter
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        # Set model to evaluation mode
        self.model.train()
        
        # Accumulate Fisher information
        samples_count = 0
        for batch in dataloader:
            if len(batch) == 3:  # Unpacked batch
                inputs, domain, labels = batch
            else:  # Dictionary batch
                inputs = batch
                domain = batch.get('domain', 'default_domain')
                labels = batch['labels']
                
            # Move to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else 
                         {k2: v2.to(self.device) for k2, v2 in v.items()} 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
                
            labels = labels.to(self.device)
            
            samples_count += len(labels)
            
            # Forward pass
            logits = self.model(inputs, domain)
            
            # Compute log probabilities
            log_probs = nn.functional.log_softmax(logits, dim=1)
            
            # Compute gradients for Fisher information
            for i in range(len(labels)):
                self.model.zero_grad()
                # Select log probability of the target class
                log_prob = log_probs[i, labels[i]]
                log_prob.backward(retain_graph=(i < len(labels) - 1))
                
                # Accumulate Fisher information
                for n, p in self.params.items():
                    if p.grad is not None:
                        fisher[n] += p.grad.data ** 2
        
        # Normalize by number of samples
        for n in fisher.keys():
            fisher[n] /= samples_count
            
        self._fisher = fisher
    
    def penalty(self, model):
        """Compute EWC penalty for current model parameters"""
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                # Compute squared distance between current and stored parameters
                # weighted by Fisher information
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        
        return self.lambda_ewc * loss

def extract_features(model, dataloader, device, domain=None):
    """Extract features for EVM classifier"""
    model.eval()
    features = {}
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different batch formats
            if len(batch) == 3:  # Unpacked batch
                inputs, batch_domain, labels = batch
                if domain is None:
                    domain = batch_domain
            else:  # Dictionary batch
                inputs = batch
                batch_domain = batch.get('domain', domain)
                labels = batch['labels']
            
            # Move inputs to device
            if isinstance(inputs, dict):
                device_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                                {k2: v2.to(device) for k2, v2 in v.items()} 
                                for k, v in inputs.items()}
            else:
                device_inputs = inputs.to(device)
            
            # Extract features
            if hasattr(model, 'extract_features'):
                batch_features = model.extract_features(device_inputs, batch_domain)
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'forward_features'):
                batch_features = model.base_model.forward_features(device_inputs)
            elif hasattr(model, 'forward_features'):
                batch_features = model.forward_features(device_inputs)
            else:
                raise ValueError("Model must have extract_features or forward_features method")
            
            # Group features by domain
            if batch_domain not in features:
                features[batch_domain] = []
            
            features[batch_domain].append(batch_features.cpu().numpy())
    
    # Concatenate features for each domain
    for domain in features:
        features[domain] = np.vstack(features[domain])
    
    return features

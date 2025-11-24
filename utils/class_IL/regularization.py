import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting"""
    def __init__(self, model: nn.Module, dataloader, device: torch.device, lambda_ewc: float = 5000.0):
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
            samples_count += len(batch['labels'])
            
            # Forward pass
            if "images" in batch:  # EAML
                images = batch['images'].to(self.device)
                texts = {k: v.to(self.device) for k, v in batch['texts'].items()}
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(images=images, texts=texts)
                logits = outputs
            else:  # DocFormer
                inputs = {
                    'pixel_values': batch['pixel_values'].to(self.device),
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'bboxes': batch['bboxes'].to(self.device)
                }
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, task="classification")
                logits = outputs['logits']
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=1)
            
            # Compute gradients
            for i in range(len(labels)):
                self.model.zero_grad()
                # Select the log probability of the target class
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
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty for current model parameters"""
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                # Compute squared distance between current and stored parameters
                # weighted by Fisher information
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        
        return self.lambda_ewc * loss
    
    def update(self, new_model: nn.Module, dataloader):
        """Update EWC with a new model and dataset"""
        # Store old values
        old_means = self._means.copy()
        old_fisher = self._fisher.copy()
        
        # Compute new Fisher information
        self.model = new_model
        self.params = {n: p for n, p in new_model.named_parameters() if p.requires_grad}
        self._compute_fisher(dataloader)
        
        # Store new parameter values
        for n, p in self.params.items():
            self._means[n] = p.data.clone()
        
        # Merge old and new Fisher information (with equal weighting)
        for n in self._fisher.keys():
            if n in old_fisher:
                self._fisher[n] = (self._fisher[n] + old_fisher[n]) / 2

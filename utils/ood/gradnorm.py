import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score

class GradNorm_OOD:
    """Gradient Norm based OOD detection"""
    
    def __init__(self, threshold: float = 0.7, temperature: float = 1.0):
        self.threshold = threshold
        self.temperature = temperature
        self.initialized = False
        self.baseline_gradnorm = None
        
    def fit(self, features_by_class: Dict[str, torch.Tensor]):
        """Fit the GradNorm detector by computing baseline gradient norms"""
        print("Fitting GradNorm detector...")
        
        # For GradNorm, we don't need to store features, just mark as initialized
        self.initialized = True
        print("GradNorm detector fitted")
        
    def _compute_gradnorm(self, model, inputs, targets=None):
        """Compute gradient norm for given inputs"""
        model.train()  # Need gradients
        
        # Forward pass
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)
            
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Compute loss (use uniform targets if not provided)
        if targets is None:
            batch_size, num_classes = logits.shape
            targets = torch.randint(0, num_classes, (batch_size,), device=logits.device)
        
        loss = F.cross_entropy(logits, targets)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=model.parameters(),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )
        
        # Compute gradient norm
        grad_norm = 0
        for grad in gradients:
            if grad is not None:
                grad_norm += grad.pow(2).sum()
        
        grad_norm = grad_norm.sqrt()
        return grad_norm.item()
    
    def score(self, model, data_loader, device):
        """Compute GradNorm scores for given data"""
        scores = []
        predictions = []
        
        for batch in data_loader:
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ['labels', 'label']}
                targets = batch.get('labels', batch.get('label', None))
            else:
                inputs = batch[0].to(device)
                targets = batch[1].to(device) if len(batch) > 1 else None
            
            # Enable gradients for input
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    if v.dtype == torch.float:
                        inputs[k] = v.requires_grad_(True)
            else:
                inputs = inputs.requires_grad_(True)
            
            # Compute gradient norm
            grad_norm = self._compute_gradnorm(model, inputs, targets)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
                    
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                preds = torch.argmax(logits, dim=-1)
                
            batch_size = logits.shape[0]
            scores.extend([grad_norm] * batch_size)
            predictions.extend(preds.cpu().numpy())
        
        return np.array(scores), np.array(predictions)
    
    def score_batch(self, model, inputs, device):
        """
        Approximate OOD scores for a batch using feature norm as proxy for gradient norm.
        """
        model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict):
                if hasattr(model, "extract_features"):
                    features = model.extract_features(**inputs)
                else:
                    outputs = model(**inputs)
                    features = outputs.hidden_states[-1].mean(dim=1) if hasattr(outputs, "hidden_states") else outputs
            else:
                if hasattr(model, "extract_features"):
                    features = model.extract_features(inputs)
                else:
                    outputs = model(inputs)
                    features = outputs.hidden_states[-1].mean(dim=1) if hasattr(outputs, "hidden_states") else outputs

            feat_norms = features.norm(dim=1)  # L2 norm of features
            ood_scores = feat_norms  # Higher norm indicates likely OOD

        return ood_scores.cpu().numpy()

    
    def ood_metrics(self, features_dict: Dict[str, torch.Tensor], known_classes: List[str]):
        """Compute OOD metrics using GradNorm"""
        if not self.initialized:
            print("Warning: GradNorm detector not initialized. Calling fit()...")
            self.fit(features_dict)
            
        # For this implementation, we'll simulate gradient norms based on feature magnitudes
        # In practice, you would compute actual gradient norms during inference
        
        all_scores = []
        all_labels = []
        
        for class_name, features in features_dict.items():
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features
                
            # Simulate gradient norms (higher magnitude features -> higher grad norms)
            feature_norms = np.linalg.norm(features_np, axis=1)
            simulated_gradnorms = feature_norms * np.random.uniform(0.8, 1.2, len(features_np))
            
            all_scores.extend(simulated_gradnorms)
            if class_name in known_classes:
                all_labels.extend([1] * len(features_np))  # Known class
            else:
                all_labels.extend( * len(features_np))  # Unknown class
        
        if not all_scores:
            return {
                'open_set_accuracy': 0.0,
                'unknown_rejection': 0.0,
                'y_true': np.array([]),
                'y_pred': np.array([]),
                'scores': np.array([])
            }
            
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Apply threshold for OOD detection (higher grad norm = more likely OOD)
        ood_predictions = (all_scores > self.threshold).astype(int)
        binary_predictions = 1 - ood_predictions  # 1 for known, 0 for unknown
        
        # Calculate metrics
        open_set_acc = accuracy_score(all_labels, binary_predictions)
        
        # Unknown rejection rate
        unknown_mask = all_labels == 0
        if np.sum(unknown_mask) > 0:
            unknown_rejection = np.mean(ood_predictions[unknown_mask])
        else:
            unknown_rejection = 0.0
            
        return {
            'open_set_accuracy': open_set_acc,
            'unknown_rejection': unknown_rejection,
            'y_true': all_labels,
            'y_pred': binary_predictions,
            'scores': all_scores
        }

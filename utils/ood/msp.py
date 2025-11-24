import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import roc_auc_score, accuracy_score

class MSP_OOD:
    """Maximum Softmax Probability (MSP) for OOD detection"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.initialized = False
        
    def fit(self, features_by_class: Dict[str, torch.Tensor]):
        """Fit the MSP detector (no training needed for MSP)"""
        self.initialized = True
        print(f"MSP detector initialized with threshold: {self.threshold}")
        
    def score(self, model, data_loader, device):
        """Compute MSP scores for given data"""
        model.eval()
        scores = []
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    outputs = model(**inputs)
                else:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                probs = F.softmax(logits, dim=-1)
                max_probs, preds = torch.max(probs, dim=-1)
                
                scores.extend(max_probs.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
                
        return np.array(scores), np.array(predictions)
    
    def score_batch(self, model, inputs, device):
        """
        Compute OOD scores for a single batch using maximum softmax probability.
        Returns the inverse of max softmax probability as OOD score.
        """
        import torch.nn.functional as F
        model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            probs = F.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            ood_scores = 1.0 - max_probs

        return ood_scores.cpu().numpy()

    
    def ood_metrics(self, features_dict: Dict[str, torch.Tensor], known_classes: List[str]):
        """Compute OOD metrics using MSP"""
        if not self.initialized:
            print("Warning: MSP detector not initialized. Calling fit()...")
            self.fit(features_dict)
            
        # Extract features and labels
        all_features = []
        all_labels = []
        
        for class_name, features in features_dict.items():
            all_features.append(features)
            if class_name in known_classes:
                all_labels.extend([1] * len(features))  # Known class
            else:
                all_labels.extend( * len(features))  # Unknown class
        
        if not all_features:
            return {
                'open_set_accuracy': 0.0,
                'unknown_rejection': 0.0,
                'y_true': np.array([]),
                'y_pred': np.array([]),
                'scores': np.array([])
            }
            
        all_features = torch.cat(all_features, dim=0)
        all_labels = np.array(all_labels)
        
        # Compute softmax probabilities (assuming features are logits)
        with torch.no_grad():
            probs = F.softmax(all_features, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0].cpu().numpy()
            predictions = torch.argmax(probs, dim=-1).cpu().numpy()
        
        # Apply threshold for OOD detection
        ood_predictions = (max_probs < self.threshold).astype(int)
        binary_predictions = 1 - ood_predictions  # 1 for known, 0 for unknown
        
        # Calculate metrics
        open_set_acc = accuracy_score(all_labels, binary_predictions)
        
        # Unknown rejection rate (how well we reject unknown samples)
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
            'scores': max_probs
        }

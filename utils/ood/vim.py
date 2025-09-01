import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score
from scipy.linalg import svd

class VIM_OOD:
    """
    ViM: Virtual-logit Matching based Out-of-Distribution Detection (CVPR 2022)
    - Fit on in-distribution features and logits to estimate mean, principal subspace, and scaling.
    - For any test input, augment logits with a virtual logit computed from the norm of the feature residual
      (distance from feature subspace), then use the softmax probability of the virtual class as OOD score.
    """
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.initialized = False
        self.u = None             # Principal directions (basis vectors), shape [d, d_sub]
        self.mu = None            # Mean feature vector, shape [d]
        self.alpha = None         # Scaling/weighting factor for virtual logit
        self.id_logit_mean = None # Mean of maximum logit for ID training set
        self.id_vim_mean = None   # Mean norm of residual for ID training set

    def fit(self, features_by_class: Dict[str, torch.Tensor], logits_by_class: Dict[str, torch.Tensor] = None, d_sub: int = 256):
        """
        features_by_class: dict of {class_name: N x D tensor}
        logits_by_class:   dict of {class_name: N x C tensor} (optional, used to estimate logit statistics)
        d_sub:             number of principal components for the feature subspace
        """
        print("Fitting ViM OOD detector...")
        # Aggregating all features (N x D)
        features = []
        for feats in features_by_class.values():
            features.append(feats if isinstance(feats, torch.Tensor) else torch.tensor(feats))
        features = torch.cat(features, dim=0)
        device = features.device
        self.mu = features.mean(dim=0).to(device)
        feats_centered = features - self.mu

        # SVD to get top principal directions
        u, s, vh = torch.linalg.svd(feats_centered, full_matrices=False)
        self.u = vh[:d_sub, :].t().to(device)  # D x d_sub

        # Produce max-logit and virtual-logit statistics for scaling alpha
        if logits_by_class is not None:
            logits = []
            for logs in logits_by_class.values():
                logits.append(logs if isinstance(logs, torch.Tensor) else torch.tensor(logs))
            logits = torch.cat(logits, dim=0)
            max_logits = logits.max(dim=1)[0]
            self.id_logit_mean = max_logits.mean().item()
        else:
            self.id_logit_mean = None

        # Project to subspace and compute virtual logit components
        feats_proj = feats_centered @ self.u @ self.u.t()
        feats_residual = feats_centered - feats_proj
        vim_scores = feats_residual.norm(dim=1)
        self.id_vim_mean = vim_scores.mean().item()

        # Scaling factor to match virtual logit and normal logit scales
        if self.id_logit_mean is not None and self.id_vim_mean != 0:
            self.alpha = self.id_logit_mean / self.id_vim_mean
        else:
            self.alpha = 1.0
        self.initialized = True
        print(f"ViM fit: mu.shape={self.mu.shape}, u.shape={self.u.shape}, alpha={self.alpha:.4f}")

    def _virtual_logit(self, feats: torch.Tensor):
        
        device = feats.device
        mu = self.mu.to(device)
        u = self.u.to(device)
        # feats: [N, D]
        feats = feats - mu
        feats_proj = feats @ u @ u.t()
        residual = feats - feats_proj
        vim = residual.norm(dim=1)
        virtual_logit = self.alpha * vim
        return virtual_logit

    def score(self, model, data_loader, device):
        """
        Returns:
          virtual_probs: Numpy, softmax probability for the virtual (OOD) class
          pred:          Numpy, model argmax predictions for ID classes
        """
        model.eval()
        features = []
        logits = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    batch_x = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    if hasattr(model, 'extract_features'):  # extract penultimate layer features
                        features_out = model.extract_features(**batch_x)
                        logits_out = model(**batch_x)
                    else:  # fallback: assume output has features/logits
                        logits_out = model(**batch_x)
                        features_out = logits_out.hidden_states[-1].mean(dim=1) if hasattr(logits_out, "hidden_states") else logits_out
                    if hasattr(logits_out, 'logits'):
                        logits_out = logits_out.logits
                else:
                    inputs = batch[0].to(device)
                    if hasattr(model, 'extract_features'):
                        features_out = model.extract_features(inputs)
                        logits_out = model(inputs)
                    else:
                        logits_out = model(inputs)
                        features_out = logits_out.hidden_states[-1].mean(dim=1) if hasattr(logits_out, "hidden_states") else logits_out
                    if hasattr(logits_out, 'logits'):
                        logits_out = logits_out.logits

                features.append(features_out.detach())
                logits.append(logits_out.detach())

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)

        virtual_logits = self._virtual_logit(features).unsqueeze(1)  # shape [N,1]
        aug_logits = torch.cat([logits, virtual_logits], dim=1)  # Append OOD "virtual" logit

        probs = F.softmax(aug_logits, dim=1)
        virtual_probs = probs[:, -1].cpu().numpy()
        preds = torch.argmax(probs[:, :-1], dim=1).cpu().numpy()
        return virtual_probs, preds
    
    def score_batch(self, model, inputs, device, texts=None):
        """
        Compute OOD scores (virtual class softmax probabilities) for a single batch.
        Args:
            model: The trained model.
            inputs: Batch inputs, dict or tensor, matching model forward requirements.
            device: Torch device.
            texts: feature extracted texts.
        Returns:
            virtual_probs: numpy array of OOD scores per sample in batch.
        """
        model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict):
                if hasattr(model, "extract_features"):
                    if texts is not None:
                        features = model.extract_features(**inputs, texts=texts)
                        outputs = model(**inputs, texts=texts)
                    else:
                        features = model.extract_features(**inputs)
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                    features = outputs.hidden_states[-1].mean(dim=1) if hasattr(outputs, "hidden_states") else outputs
            else:
                if hasattr(model, "extract_features"):
                    if texts is not None:
                        features = model.extract_features(inputs, texts=texts)
                    else:
                        features = model.extract_features(inputs)
                    outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    features = outputs.hidden_states[-1].mean(dim=1) if hasattr(outputs, "hidden_states") else outputs

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            virtual_logits = self._virtual_logit(features).unsqueeze(1)
            aug_logits = torch.cat([logits, virtual_logits], dim=1)

            probs = torch.nn.functional.softmax(aug_logits, dim=1)
            virtual_probs = probs[:, -1].cpu().numpy()

        return virtual_probs

    def ood_metrics(self, features_dict: Dict[str, torch.Tensor], known_classes: List[str]):
        """
        features_dict: dict {class_name: N x D feature tensor}
        known_classes: list of class names for in-distribution
        Returns stats including softmax probability of virtual logit
        """
        if not self.initialized:
            raise RuntimeError("ViM_OOD: call fit() before ood_metrics()!")

        # Creates N x D features and labels vector
        all_features, all_labels = [], []
        for class_name, feats in features_dict.items():
            feats = feats if isinstance(feats, torch.Tensor) else torch.tensor(feats)
            all_features.append(feats)
            all_labels.extend([1 if class_name in known_classes else 0] * feats.size(0))
        features = torch.cat(all_features, dim=0)
        all_labels = np.array(all_labels)

        # Computes virtual logits and softmax probs
        if hasattr(self, "logit_dim"):
            num_classes = self.logit_dim
        else:
            num_classes = len(known_classes)
        dummy_logits = torch.zeros(features.shape[0], num_classes)  # (N, C)
        virtual_logits = self._virtual_logit(features).unsqueeze(1)
        aug_logits = torch.cat([dummy_logits, virtual_logits], dim=1)
        probs = F.softmax(aug_logits, dim=1)
        virtual_probs = probs[:, -1].cpu().numpy()
        # OOD classification: virtual_prob >= threshold → OOD (0), else ID (1)
        ood_predictions = (virtual_probs >= self.threshold).astype(int)
        binary_predictions = 1 - ood_predictions  # 1: known, 0: unknown

        open_set_acc = accuracy_score(all_labels, binary_predictions)
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
            'scores': virtual_probs
        }

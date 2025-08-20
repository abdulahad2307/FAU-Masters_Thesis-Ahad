import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

class IncrementalStrategy:
    """Base class for incremental learning strategies"""
    def __init__(self, device):
        self.device = device

    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model for new classes"""
        raise NotImplementedError

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        """Compute loss with strategy-specific components"""
        raise NotImplementedError

class StandardIncremental(IncrementalStrategy):
    """Standard incremental learning without forgetting mitigation"""
    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model architecture for new classes"""
        if model_name == "docformer" and hasattr(model, 'classifier'):
            # Save old classifier weights
            old_classifier = model.classifier.weight.data.clone()
            old_bias = model.classifier.bias.data.clone() if model.classifier.bias is not None else None
            
            # Initialize new classifier
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, new_num_classes)
            
            # Copying old weights
            with torch.no_grad():
                model.classifier.weight.data[:old_num_classes] = old_classifier
                if old_bias is not None and model.classifier.bias is not None:
                    model.classifier.bias.data[:old_num_classes] = old_bias
                    
        elif model_name == "eaml":
            # Saving old classifier weights
            old_image_classifier = model.image_classifier.weight.data.clone()
            old_image_bias = model.image_classifier.bias.data.clone()
            old_text_classifier = model.text_classifier.weight.data.clone()
            old_text_bias = model.text_classifier.bias.data.clone()
            old_fusion_classifier = model.fusion_classifier.weight.data.clone()
            old_fusion_bias = model.fusion_classifier.bias.data.clone()
            
            # Initializing new classifiers
            model.image_classifier = nn.Linear(model.image_classifier.in_features, new_num_classes)
            model.text_classifier = nn.Linear(model.text_classifier.in_features, new_num_classes)
            model.fusion_classifier = nn.Linear(model.fusion_classifier.in_features, new_num_classes)
            
            # Copy overlapping weights safely
            with torch.no_grad():
                num_copy_img = min(old_num_classes, old_image_classifier.size(0), model.image_classifier.weight.data.size(0))
                model.image_classifier.weight.data[:num_copy_img] = old_image_classifier[:num_copy_img]
                model.image_classifier.bias.data[:num_copy_img] = old_image_bias[:num_copy_img]

                num_copy_txt = min(old_num_classes, old_text_classifier.size(0), model.text_classifier.weight.data.size(0))
                model.text_classifier.weight.data[:num_copy_txt] = old_text_classifier[:num_copy_txt]
                model.text_classifier.bias.data[:num_copy_txt] = old_text_bias[:num_copy_txt]

                num_copy_fus = min(old_num_classes, old_fusion_classifier.size(0), model.fusion_classifier.weight.data.size(0))
                model.fusion_classifier.weight.data[:num_copy_fus] = old_fusion_classifier[:num_copy_fus]
                model.fusion_classifier.bias.data[:num_copy_fus] = old_fusion_bias[:num_copy_fus]

                    
        return model

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            texts = batch['texts']
            # Moving text tensors to device
            texts = {k: v.to(self.device) for k, v in texts.items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = model(images=images, texts=texts)
            logits = outputs
        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bboxes': batch['bboxes'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            
        # Classification loss
        loss = criterion(logits, labels)
        
        # Adding EWC regularization if available
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_loss
            
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

class DistillationIncremental(IncrementalStrategy):
    """Incremental learning with knowledge distillation"""
    def __init__(self, device, temperature=2.0, lambda_distill=1.0):
        super().__init__(device)
        self.temperature = temperature
        self.lambda_distill = lambda_distill

    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model architecture for new classes"""
        return StandardIncremental(self.device).adapt_model(model, old_num_classes, new_num_classes, model_name)

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        """Compute loss with distillation component"""
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            texts = batch['texts']
            # Moving text tensors to device
            texts = {k: v.to(self.device) for k, v in texts.items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = model(images=images, texts=texts)
            logits = outputs
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(images=images, texts=texts)
                    old_logits = old_outputs
                old_class_count = old_logits.size(1)
                print(f"Student logits shape: {logits.shape}")
                print(f"Teacher logits shape: {old_logits.shape}")
                print(f"Old class count: {old_class_count}")
                    
                # Only applying distillation to old classes
                old_class_count = old_logits.size(1)
                
                # Getting soft targets from old model
                soft_targets = nn.functional.softmax(old_logits / self.temperature, dim=1)
                
                # Getting soft probabilities from current model (only for old classes)
                soft_probs = nn.functional.log_softmax(logits[:, :old_class_count] / self.temperature, dim=1)
                
                # Calculating distillation loss
                dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
                
                # Combined loss
                loss = cls_loss + self.lambda_distill * dist_loss
            else:
                loss = cls_loss

                print(f"Classification loss: {cls_loss.item()}, Distillation loss: {dist_loss.item() if old_model is not None else 'N/A'}, EWC loss: {ewc_loss.item() if ewc is not None else 'N/A'}")

                
        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bboxes': batch['bboxes'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(**inputs, task="classification")
                    old_logits = old_outputs['logits']
                    
                # Only applying distillation to old classes
                old_class_count = old_logits.size(1)
                
                # Getting soft targets from old model
                soft_targets = nn.functional.softmax(old_logits / self.temperature, dim=1)
                
                # Getting soft probabilities from current model (only for old classes)
                soft_probs = nn.functional.log_softmax(logits[:, :old_class_count] / self.temperature, dim=1)
                
                # Calculating distillation loss
                dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
                
                # Combined loss
                loss = cls_loss + self.lambda_distill * dist_loss
            else:
                loss = cls_loss
        
        # Adding EWC regularization if available
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_loss
            
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

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

class ExemplarManager:
    """Manages exemplars for replay-based class incremental learning"""
    def __init__(self, 
                 max_exemplars=200, 
                 max_per_class=20, 
                 selection_strategy="herding"):
        self.exemplars = {}
        self.max_exemplars = max_exemplars
        self.max_per_class = max_per_class
        self.selection_strategy = selection_strategy
        self.feature_extractor = None
        
    def set_feature_extractor(self, model):
        """Set the feature extractor model for herding selection"""
        self.feature_extractor = model
        
    def update(self, dataset, class_name, model=None):
        """Update exemplar set with examples from a new class"""
        if self.selection_strategy == "herding" and model is not None:
            self.exemplars[class_name] = self._select_herding(dataset, class_name, model)
        else:
            self.exemplars[class_name] = self._select_random(dataset, class_name)
            
        # Reduce exemplar set if needed
        self._balance_exemplar_set()
        
    def _select_random(self, dataset, class_name):
        """Select random exemplars for a class"""
        import random
        class_samples = [s for s in dataset.samples if s[2] == class_name]
        if len(class_samples) <= self.max_per_class:
            return class_samples
        return random.sample(class_samples, self.max_per_class)
    
    def _select_herding(self, dataset, class_name, model):
        """Select exemplars using herding (closest to class mean)"""
        class_samples = [s for s in dataset.samples if s[2] == class_name]
        if len(class_samples) <= self.max_per_class:
            return class_samples
            
        # Extract features for all samples of this class
        features = []
        labels = []
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for sample in class_samples:
                img_path, tokens, _ = sample
                image = dataset.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                if hasattr(model, 'extract_features'):
                    feature = model.extract_features(
                        images=image,
                        texts={
                            'input_ids': tokens["input_ids"].unsqueeze(0).to(device),
                            'attention_mask': tokens["attention_mask"].unsqueeze(0).to(device)
                        }
                    )
                else:
                    outputs = model(images=image, texts={
                        'input_ids': tokens["input_ids"].unsqueeze(0).to(device),
                        'attention_mask': tokens["attention_mask"].unsqueeze(0).to(device)
                    })
                    feature = outputs
                features.append(feature.cpu().numpy())
                labels.append(sample)
        
        # Compute class mean
        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        
        # Select samples closest to the mean (herding selection)
        selected_indices = []
        selected_features = []
        for _ in range(min(self.max_per_class, len(features))):
            if len(selected_features) == 0:
                # Initialize with the closest sample to the mean
                distances = np.linalg.norm(features - class_mean, axis=1)
                idx = np.argmin(distances)
            else:
                # Select the sample that makes the selected set's mean closest to the class mean
                current_mean = np.mean(np.array(selected_features), axis=0)
                candidate_means = np.array([
                    (current_mean * len(selected_features) + features[i]) / (len(selected_features) + 1)
                    for i in range(len(features)) if i not in selected_indices
                ])
                distances = np.linalg.norm(candidate_means - class_mean, axis=1)
                remaining_indices = [i for i in range(len(features)) if i not in selected_indices]
                idx = remaining_indices[np.argmin(distances)]
                
            selected_indices.append(idx)
            selected_features.append(features[idx])
            
        return [labels[i] for i in selected_indices]
    
    def _balance_exemplar_set(self):
        """Balance exemplar set to ensure fair representation of all classes"""
        total_exemplars = sum(len(exems) for exems in self.exemplars.values())
        if total_exemplars <= self.max_exemplars:
            return
            
        # Reduce exemplars per class evenly
        target_per_class = self.max_exemplars // len(self.exemplars)
        remainder = self.max_exemplars % len(self.exemplars)
        for i, class_name in enumerate(self.exemplars.keys()):
            target = target_per_class + (1 if i < remainder else 0)
            if len(self.exemplars[class_name]) > target:
                self.exemplars[class_name] = self.exemplars[class_name][:target]
                
    def get_exemplar_dataset(self, transform=None):
        """Convert exemplars to a dataset-like format"""
        all_exemplars = []
        for class_name, examples in self.exemplars.items():
            all_exemplars.extend(examples)
        return all_exemplars

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

def extract_features(model, dataloader, device, max_samples_per_class=None):
    """
    Extract features for EVM classifier in a model-agnostic way, with optional per-class sample limit.
    Returns: Dict[class_name, np.ndarray of features]
    """
    from collections import defaultdict
    import random

    model.eval()
    features = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            if "images" in batch:  # EAML
                images = batch['images'].to(device)
                texts = {k: v.to(device) for k, v in batch['texts'].items()}
                labels = batch['labels'].to(device)

                # Prefer model.extract_features if available
                if hasattr(model, 'extract_features'):
                    batch_features = model.extract_features(images=images, texts=texts)
                else:
                    outputs = model(images=images, texts=texts, return_features=True)
                    batch_features = outputs['fused_feat']

            else:  # DocFormer
                inputs = {
                    'pixel_values': batch['pixel_values'].to(device),
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'bboxes': batch['bboxes'].to(device)
                }
                labels = batch['labels'].to(device)
                outputs = model(**inputs, task="classification")
                batch_features = outputs['features'] if 'features' in outputs else outputs['logits']

            # Group features by class
            for i, label in enumerate(labels.cpu().numpy()):
                class_name = dataloader.dataset.current_classes[label]
                features[class_name].append(batch_features[i].cpu().numpy())

    # If max_samples_per_class is set, randomly sample up to that many per samples class
    for class_name in features:
        if max_samples_per_class is not None and len(features[class_name]) > max_samples_per_class:
            features[class_name] = random.sample(features[class_name], max_samples_per_class)
        features[class_name] = np.vstack(features[class_name])

    return features


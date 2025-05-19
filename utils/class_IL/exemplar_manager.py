import torch
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image
import random
import os

class ExemplarManager:
    """Manages exemplars for replay-based class incremental learning"""
    def __init__(self, max_exemplars=200, max_per_class=20, selection_strategy="herding"):
        self.exemplars = {}  # class_name -> list of examples
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
                            'input_ids': tokens['input_ids'].unsqueeze(0).to(device),
                            'attention_mask': tokens['attention_mask'].unsqueeze(0).to(device)
                        }
                    )
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

import os
import torch
from typing import List, Dict, Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class DILDataLoader:
    """
    Domain Incremental Learning DataLoader
    Handles multiple domains with separate train/val/test splits
    """
    def __init__(self, data_root: str, domain_list: List[str], 
                 batch_size: int = 16, img_size: tuple = (224, 224), 
                 num_workers: int = 4):
        self.data_root = data_root
        self.domain_list = domain_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        
        # Standard transforms for document images
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validate domains
        self._validate_domains()
        
    def _validate_domains(self):
        """Validate that all domain directories exist"""
        for domain in self.domain_list:
            domain_path = os.path.join(self.data_root, domain)
            if not os.path.isdir(domain_path):
                raise ValueError(f"Domain directory not found: {domain_path}")
            print(f"✓ Found domain: {domain}")
    
    def get_domain_loaders(self, phase: str = 'train') -> Dict[str, DataLoader]:
        """
        Get DataLoader for each domain
        
        Args:
            phase: 'train', 'val', or 'test'
            
        Returns:
            Dictionary mapping domain names to DataLoaders
        """
        loaders = {}
        
        for domain in self.domain_list:
            domain_phase_path = os.path.join(self.data_root, domain, phase)
            
            if not os.path.exists(domain_phase_path):
                print(f"Warning: {phase} directory not found for domain {domain}")
                continue
                
            try:
                dataset = datasets.ImageFolder(domain_phase_path, transform=self.transform)
                
                if len(dataset) == 0:
                    print(f"Warning: No images found in {domain_phase_path}")
                    continue
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(phase == 'train'),
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=(phase == 'train')
                )
                
                loaders[domain] = loader
                print(f"✓ Created {phase} loader for {domain}: {len(dataset)} samples")
                
            except Exception as e:
                print(f"Error creating loader for domain {domain}: {e}")
                continue
        
        return loaders
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get number of classes for each domain"""
        class_counts = {}
        
        for domain in self.domain_list:
            train_path = os.path.join(self.data_root, domain, 'train')
            if os.path.exists(train_path):
                try:
                    dataset = datasets.ImageFolder(train_path)
                    class_counts[domain] = len(dataset.classes)
                    print(f"Domain {domain}: {class_counts[domain]} classes")
                except:
                    class_counts[domain] = 0
            else:
                class_counts[domain] = 0
                
        return class_counts

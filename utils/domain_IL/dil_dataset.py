import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from typing import Dict, List, Tuple, Union, Optional

def get_domain_transforms(img_size: Tuple[int, int] = (224, 224)):
    """Standard image transformation for all domains"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class DomainDataset(Dataset):
    """Dataset for a specific domain"""
    def __init__(self, data_path, transform=None, domain_name=None):
        self.data_path = data_path
        self.transform = transform
        self.domain_name = domain_name
        
        # Load dataset
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
            
        self.dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, self.domain_name, label

def get_domain_loader(domain_name: str, data_path: str, batch_size: int, 
                      img_size=(224, 224), is_train=True, val_split=0.2):
    """
    Creating data loaders for the given domain
    Args:
        domain_name: Name of the domain
        data_path: Path to the domain data
        batch_size: Batch size
        img_size: Input image size
        is_train: Whether to create train loader
        val_split: Validation split ratio if creating both train and val loaders
    Returns:
        train_loader, val_loader (or single loader if is_train is specified)
    """
    transform = get_domain_transforms(img_size)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    # Create dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    # Split dataset if needed
    if is_train and val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    else:
        # Create single loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=4,
            pin_memory=True
        )
        
        return loader

def get_all_domains_loaders(domains: List[Dict], batch_size: int, 
                           img_size=(224, 224), is_train=True, val_split=0.2):
    """
    Getting loaders for all domains in DIL
    Args:
        domains: List of domain dicts, each with 'name' and 'path' keys
        batch_size: Batch size
        img_size: Input image size
        is_train: Whether to create train loaders
        val_split: Validation split ratio if is_train is True
    Returns:
        List of (domain_name, loader) or (domain_name, (train_loader, val_loader)) tuples
    """
    loaders = []
    
    for domain in domains:
        domain_name = domain["name"]
        data_path = domain["path"]
        
        # Get loader(s) for this domain
        domain_loaders = get_domain_loader(
            domain_name=domain_name,
            data_path=data_path,
            batch_size=batch_size,
            img_size=domain.get("input_size", img_size),
            is_train=is_train,
            val_split=val_split
        )
        
        loaders.append((domain_name, domain_loaders))
        
    return loaders

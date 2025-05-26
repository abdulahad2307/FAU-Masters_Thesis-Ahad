import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict

def get_domain_transforms(img_size: Tuple[int, int] = (224, 224)):
    """Standard image transformation for all domains."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean/std
                             std=[0.229, 0.224, 0.225])
    ])

def get_domain_loader(domain_name: str, data_path: str, batch_size: int, img_size=(224, 224), is_train=True):
    """Create a data loader for a specific domain folder."""
    transform = get_domain_transforms(img_size)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True
    )

def get_all_domains_loaders(domains: List[Dict], batch_size: int, img_size=(224, 224)):
    """Get loaders for all domain folders."""
    loaders = []
    for domain in domains:
        loader = get_domain_loader(
            domain_name=domain["name"],
            data_path=domain["path"],
            batch_size=batch_size,
            img_size=domain.get("input_size", img_size)
        )
        loaders.append((domain["name"], loader))
    return loaders

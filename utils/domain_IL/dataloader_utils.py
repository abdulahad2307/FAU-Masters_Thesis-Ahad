from typing import Optional, Dict
from torch.utils.data import DataLoader as TorchDataLoader

from utils.dataloader import (
    EAML_DataLoader,
    DataLoader as DocFormerLoader
)

def get_dataloaders(model_type: str,
                    data_dir: str,
                    batch_size: int = 16,
                    num_workers: int = 4,
                    class_list: Optional[list] = None,
                    img_size: int = 224,
                    mode: str = "train") -> Dict[str, TorchDataLoader]:
    """
    Load data loaders for domain-incremental learning.
    
    Args:
        model_type: 'eaml' or 'docformer'
        data_dir: Root directory of dataset with subfolders train/val/test
        batch_size: Batch size
        num_workers: Workers for loading
        class_list: Optional class list
        img_size: Image resize size
        mode: One of 'train', 'val', 'test', or 'all'
        
    Returns:
        Dict of dataloaders
    """
    loaders = {}

    if model_type == "eaml":
        loader = EAML_DataLoader(data_dir, batch_size, num_workers, img_size, class_list)
    elif model_type == "docformer":
        loader = DocFormerLoader(data_dir, batch_size, num_workers, img_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if mode in ["train", "all"]:
        loaders["train"] = loader.get_loader("train", shuffle=True)
    if mode in ["val", "all"]:
        loaders["val"] = loader.get_loader("val", shuffle=False)
    if mode in ["test", "all"]:
        loaders["test"] = loader.get_loader("test", shuffle=False)

    return loaders

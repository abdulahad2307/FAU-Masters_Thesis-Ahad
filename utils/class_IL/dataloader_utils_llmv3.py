import os
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import torch
from transformers import BertTokenizer
from utils.llmv3.llmv3_data_loader import get_dataloaders, CILLayoutLMv3Dataset, layoutlmv3_cil_collate_fn

common_transform = transforms.Compose([
    transforms.Resize((229, 229)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class ExemplarDatasetWrapper(Dataset):
    def __init__(self, exemplar_samples, transform=None, max_length=512, bbox_style="rect"):
        self.samples = exemplar_samples
        self.transform = transform
        self.max_length = max_length
        self.bbox_style = bbox_style

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tokens, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": torch.zeros((self.max_length, 4), dtype=torch.long),
            "labels": torch.tensor(tokens.get("label", -1), dtype=torch.long)
        }

def get_class_il_loader(
    model_type: str,
    data_dir: str,
    current_classes: List[str],
    batch_size: int = 16,
    num_workers: int = 0,
    ocr_data: Optional[str] = None,
    max_length: int = 512,         
    bbox_style: str = "rect",
    exemplar_dataset: Optional[List] = None
) -> DataLoader:
    if model_type == "layoutlmv3":
        main_dataset = CILLayoutLMv3Dataset(
            image_dir=data_dir,
            ocr_tensor_file=ocr_data,
            current_classes=current_classes,
            max_length=max_length,
            bbox_style=bbox_style
        )
        if exemplar_dataset and len(exemplar_dataset) > 0:
            exemplar_ds = ExemplarDatasetWrapper(exemplar_dataset, transform=common_transform, max_length=max_length, bbox_style=bbox_style)
            combined_dataset = ConcatDataset([main_dataset, exemplar_ds])
        else:
            combined_dataset = main_dataset
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=layoutlmv3_cil_collate_fn
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

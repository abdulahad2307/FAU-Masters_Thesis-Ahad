import os
import json
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from transformers import BertTokenizer
from PIL import Image
import torch
from typing import Optional, Dict, List
import numpy as np
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALB_AVAILABLE = True
except ImportError:
    ALB_AVAILABLE = False
import warnings

class EAML_Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None, class_list: Optional[List[str]] = None, 
                 ocr_json_path=None, img_size=224):  # Add img_size parameter
        self.data_dir = data_dir
        self.transform = transform
        self.class_list = sorted(class_list) if class_list else None
        self.samples = []
        self.ocr_json_path = ocr_json_path
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        print("Class to index mapping:", self.class_to_idx)
        self.idx_to_class = {}
        self.img_size = img_size  # Store img_size for fallback
        self._build_class_mappings()
        if self.ocr_json_path:
            with open(self.ocr_json_path, "r", encoding="utf-8") as f:
                self.ocr_texts = json.load(f)
        else:
            self.ocr_texts = None
        self._load_samples()
        self._verify_labels()
        if not self.samples:
            raise ValueError(f"No valid samples found for specified classes: {class_list}")

    def _build_class_mappings(self):
        if self.class_list is None:
            raise ValueError("Class list must be provided")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_list)}

    def _verify_labels(self):
        valid_samples = []
        invalid_samples = 0
        for sample in self.samples:
            _, _, label = sample
            if label in self.class_to_idx:
                valid_samples.append(sample)
            else:
                print(f"Label mismatch: {label} not in class_to_idx")
                invalid_samples += 1
        if invalid_samples > 0:
            print(f"Warning: Found {invalid_samples} samples with invalid labels")
        self.samples = valid_samples

    def _load_samples(self):
        print(f"Loading dataset from: {self.data_dir}")
        print(f"Filtering for classes: {self.class_list}")
        class_set = set(self.class_list) if self.class_list else set()
        valid_samples = 0
        skipped_samples = 0
        for root, _, files in os.walk(self.data_dir):
            label = os.path.basename(root)
            if label not in class_set:
                continue
            for img_file in files:
                if img_file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, img_file)
                    try:
                        if self.ocr_texts is not None:
                            extracted_text = self.ocr_texts.get(img_path, "")
                        else:
                            extracted_text = ""
                        if not extracted_text.strip():
                            skipped_samples += 1
                            continue
                        tokenized_text = self.tokenizer(
                            extracted_text,
                            padding="max_length",
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        )
                        self.samples.append((img_path, tokenized_text, label))
                        valid_samples += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        skipped_samples += 1
                        continue
        print(f"Loaded {valid_samples} valid samples")
        print(f"Skipped {skipped_samples} samples due to errors")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, text_data, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    # Fallback transform uses self.img_size
                    fallback = transforms.Compose([
                        transforms.Resize((self.img_size, self.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    image = fallback(image)
                    warnings.warn(f"Transform failed for {img_path}: {str(e)}. Used fallback.")
            return {
                'image': image,
                'text': {
                    'input_ids': text_data['input_ids'].squeeze(0),
                    'attention_mask': text_data['attention_mask'].squeeze(0)
                },
                'label': torch.tensor(self.class_to_idx[label])
            }
        except Exception as e:
            print(f"Error loading sample {img_path}: {str(e)}")
            raise

def eaml_collate_fn(batch: List[Dict]) -> Dict:
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'texts': {
            'input_ids': torch.stack([item['text']['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['text']['attention_mask'] for item in batch])
        },
        'labels': torch.stack([item['label'] for item in batch])
    }

class EAML_DataLoader:
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4,
                 img_size: int = 224, class_list: Optional[List[str]] = None,
                 ocr_json_path=None):
        if not class_list:
            raise ValueError("Class list cannot be empty")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.class_list = sorted(class_list)
        self.ocr_json_path = ocr_json_path
        self.transform_train = self._build_transform(train=True)
        self.transform_eval = self._build_transform(train=False)

    def _build_transform(self, train=True):
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        
        if train:
            aug_transforms = [
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    shear=10
                )
            ]
            
            if ALB_AVAILABLE:
                # Define albumentations cutout (using CoarseDropout)
                def albumentations_cutout(img):
                    img_np = np.array(img)
                    # Corrected: Define aug inside the function
                    
                    aug = A.CoarseDropout(
                        max_holes=1, 
                        max_height=32, 
                        max_width=32, 
                        fill_value=0, 
                        p=1.0
                    )
                    """
                    aug = A.CoarseDropout(
                        max_holes=1,
                        min_holes=1,  
                        max_height=32,
                        min_height=32,
                        max_width=32,
                        min_width=32,
                        fill_value=0,
                        p=1.0
                    )
                    """
                    return Image.fromarray(aug(image=img_np)['image'])
                aug_transforms.insert(0, transforms.Lambda(albumentations_cutout))
            
            # Combine augmentations with base transforms
            return transforms.Compose(aug_transforms + base_transforms)
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def get_loader(self, split: str, shuffle: bool = True):
        dataset_path = os.path.join(self.data_dir, split)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Split directory does not exist: {dataset_path}")
        transform = self.transform_train if split == 'train' else self.transform_eval
        dataset = EAML_Dataset(
            data_dir=dataset_path,
            transform=transform,
            class_list=self.class_list,
            ocr_json_path=self.ocr_json_path
        )
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=eaml_collate_fn
        )

def load_class_list(class_list_path: Optional[str] = None) -> Optional[List[str]]:
    if class_list_path and os.path.exists(class_list_path):
        with open(class_list_path) as f:
            return json.load(f)
    return None

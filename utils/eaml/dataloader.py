import os
import json
import warnings
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torchvision import transforms
from transformers import BertTokenizer
from typing import Optional, List, Dict

class EAML_Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        class_list: Optional[List[str]] = None,
        ocr_data_path: Optional[str] = None,
        img_size: int = 224,
        handle_empty_text: str = "exclude",
        fallback_text: str = "[EMPTY]"
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.class_list = sorted(class_list) if class_list else None
        self.img_size = img_size
        self.handle_empty_text = handle_empty_text
        self.fallback_text = fallback_text

        if not self.class_list:
            raise ValueError("class_list must be provided and non-empty")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ocr_texts, self.ocr_tokenized = self._load_ocr_data(ocr_data_path)
        self.samples = []
        self._load_samples()
        self._verify_labels()
        if not self.samples:
            raise ValueError(f"No valid samples found for specified classes: {self.class_list}")

    def _load_ocr_data(self, ocr_path):
        if ocr_path is None:
            return {}, False
        if not os.path.exists(ocr_path):
            raise FileNotFoundError(f"OCR data file not found: {ocr_path}")
        if ocr_path.endswith(".json"):
            with open(ocr_path, "r", encoding="utf-8") as f:
                return json.load(f), False
        elif ocr_path.endswith((".pt", ".pth")):
            loaded = torch.load(ocr_path, map_location="cpu")
            # Heuristic: check if values are dict with 'input_ids'
            first_val = next(iter(loaded.values()))
            if isinstance(first_val, dict) and "input_ids" in first_val:
                return loaded, True
            else:
                raise ValueError("Tensor OCR file must be dict[img]->{'input_ids','attention_mask'}")
        else:
            raise ValueError(f"Unsupported OCR data format: {ocr_path}")

    def _is_text_empty(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return True
        return len(text.strip()) == 0

    def _get_ocr_entry(self, img_path: str):
        # Try exact path, then filename
        if img_path in self.ocr_texts:
            return self.ocr_texts[img_path]
        filename = os.path.basename(img_path)
        if filename in self.ocr_texts:
            return self.ocr_texts[filename]
        # Try endswith matching
        for key in self.ocr_texts.keys():
            if img_path.endswith(key) or key.endswith(filename):
                return self.ocr_texts[key]
        return None

    def _load_samples(self):
        print(f"Loading dataset from: {self.data_dir}")
        print(f"Filtering for classes: {self.class_list}")
        valid_samples = 0
        skipped_empty = 0
        skipped_errors = 0
        classes_set = set(self.class_list)
        for root, _, files in os.walk(self.data_dir):
            label = os.path.basename(root)
            if label not in classes_set:
                continue
            for img_file in files:
                if img_file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, img_file)
                    try:
                        ocr_entry = self._get_ocr_entry(img_path)
                        if ocr_entry is None:
                            skipped_errors += 1
                            continue
                        if self.ocr_tokenized:
                            # Already tokenized
                            tokenized = {
                                "input_ids": ocr_entry["input_ids"],
                                "attention_mask": ocr_entry["attention_mask"]
                            }
                        else:
                            # String, needs tokenization
                            text = ocr_entry
                            if self._is_text_empty(text):
                                if self.handle_empty_text == "exclude":
                                    skipped_empty += 1
                                    continue
                                elif self.handle_empty_text == "fallback":
                                    text = self.fallback_text
                            tokenized = self.tokenizer(
                                text,
                                padding="max_length",
                                truncation=True,
                                max_length=128,
                                return_tensors="pt"
                            )
                            tokenized = {
                                "input_ids": tokenized["input_ids"].squeeze(0),
                                "attention_mask": tokenized["attention_mask"].squeeze(0)
                            }
                        self.samples.append((img_path, tokenized, label))
                        valid_samples += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        skipped_errors += 1
        print(f"Loaded {valid_samples} valid samples")
        print(f"Skipped {skipped_empty} samples due to empty text")
        print(f"Skipped {skipped_errors} samples due to errors")

    def _verify_labels(self):
        valid_samples = []
        invalid_count = 0
        for sample in self.samples:
            _, _, label = sample
            if label in self.class_to_idx:
                valid_samples.append(sample)
            else:
                invalid_count += 1
        if invalid_count > 0:
            print(f"Warning: Found {invalid_count} samples with invalid labels")
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text_data, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    fallback_transform = transforms.Compose([
                        transforms.Resize((self.img_size, self.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                    ])
                    image = fallback_transform(image)
                    warnings.warn(f"Transform failed for {img_path}: {e}. Used fallback transform.")
            return {
                "image": image,
                "text": {
                    "input_ids": text_data["input_ids"],
                    "attention_mask": text_data["attention_mask"]
                },
                "label": torch.tensor(self.class_to_idx[label]),
                "img_path": img_path
            }
        except Exception as e:
            print(f"Error loading sample {img_path}: {e}")
            raise

def eaml_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "texts": {
            "input_ids": torch.stack([item["text"]["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["text"]["attention_mask"] for item in batch])
        },
        "labels": torch.stack([item["label"] for item in batch]),
        "img_paths": [item["img_path"] for item in batch]
    }

class EAML_DataLoader:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
        img_size: int = 224,
        class_list: Optional[List[str]] = None,
        ocr_data_path: Optional[str] = None,
        handle_empty_text: str = "exclude",
        fallback_text: str = "[EMPTY]"
    ):
        if not class_list:
            raise ValueError("class_list cannot be empty")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.class_list = sorted(class_list)
        self.ocr_data_path = ocr_data_path
        self.handle_empty_text = handle_empty_text
        self.fallback_text = fallback_text
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
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ]
            return transforms.Compose(aug_transforms + base_transforms)
        else:
            return transforms.Compose(base_transforms)

    def get_loader(self, split: str, shuffle: bool = True):
        dataset_path = os.path.join(self.data_dir, split)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Split directory does not exist: {dataset_path}")
        transform = self.transform_train if split == "train" else self.transform_eval
        dataset = EAML_Dataset(
            data_dir=dataset_path,
            transform=transform,
            class_list=self.class_list,
            ocr_data_path=self.ocr_data_path,
            img_size=self.img_size,
            handle_empty_text=self.handle_empty_text,
            fallback_text=self.fallback_text
        )
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=eaml_collate_fn,
            pin_memory=True
        )

def load_class_list(class_list_path: Optional[str] = None) -> Optional[List[str]]:
    if class_list_path and os.path.exists(class_list_path):
        with open(class_list_path) as f:
            return json.load(f)
    return None

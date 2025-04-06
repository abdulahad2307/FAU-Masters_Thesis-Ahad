# utils/dataloader.py
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torchvision.datasets import ImageFolder
from transformers import BertTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
from typing import Optional, Union, List, Dict

class EAML_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.texts = []
        self.labels = []
        
        # Initialize OCR components
        self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.load_samples()

    def extract_text_from_image(self, image):
        """Use TROCR for text extraction instead of Tesseract"""
        pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self.ocr_model.generate(pixel_values)
        return self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def load_samples(self):
        """Modified to use TROCR directly without text files"""
        print(f"📂 Scanning dataset directory: {self.data_dir}")
        num_files = 0
        
        for root, _, files in os.walk(self.data_dir):
            for img_file in files:
                if img_file.endswith((".tif", ".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, img_file)
                    label = os.path.basename(root)
                    num_files += 1
                    
                    # Process image and extract text
                    image = Image.open(img_path).convert("RGB")
                    extracted_text = self.extract_text_from_image(image)
                    
                    # Tokenize text
                    tokenized_text = self.tokenizer(
                        extracted_text, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=128, 
                        return_tensors="pt"
                    )
                    
                    self.image_paths.append(img_path)
                    self.texts.append(tokenized_text)
                    self.labels.append(label)
        
        print(f"✅ Found {num_files} images, Loaded {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.texts[idx], self.labels[idx]

class DocFormerDataset(Dataset):
    """DocFormer-compatible dataset (unchanged from your version)"""
    def __init__(self, data_dir, tokenizer_name="bert-base-uncased", max_seq_length=512, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.split = split
        
        self.samples = []
        self.class_names = []
        self._load_samples()
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(self.class_names)))}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'class_name': class_name
                        })
                        self.class_names.append(class_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        pixel_values = self.transform(image)
        
        input_ids = torch.tensor([self.tokenizer.cls_token_id] + 
                               [self.tokenizer.pad_token_id] * (self.max_seq_length - 2) +
                               [self.tokenizer.sep_token_id])[:self.max_seq_length]
        
        attention_mask = torch.tensor([1] + [0] * (self.max_seq_length - 2) + [1])[:self.max_seq_length]
        
        bboxes = torch.tensor([[0, 0, image.width, 0, image.width, image.height, 0, image.height]] * self.max_seq_length)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bboxes': bboxes,
            'label': torch.tensor(self.class_to_idx[sample['class_name']])
        }

class DataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=224, 
                 dataset_type="all", classes=None, use_eaml=False, use_docformer=False):
        if use_eaml and use_docformer:
            raise ValueError("Cannot use both EAML and DocFormer modes simultaneously")
            
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.classes = classes
        self.use_eaml = use_eaml
        self.use_docformer = use_docformer

        # Configure transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406] if use_docformer else [0.5], 
                std=[0.229, 0.224, 0.225] if use_docformer else [0.5]
            )
        ])

    def load_dataset(self, dataset_type):
        dataset_path = os.path.join(self.data_dir, dataset_type)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset directory does not exist: {dataset_path}")

        if self.use_eaml:
            return EAML_Dataset(dataset_path, transform=self.transform)
        elif self.use_docformer:
            return DocFormerDataset(
                data_dir=self.data_dir,
                split=dataset_type,
                max_seq_length=512
            )
        else:
            return ImageFolder(root=dataset_path, transform=self.transform)

    def get_data_loader(self, dataset_type):
        dataset = self.load_dataset(dataset_type)
        
        if self.use_docformer:
            return TorchDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(dataset_type == "train"),
                num_workers=self.num_workers,
                collate_fn=self.docformer_collate_fn
            )
        else:
            return TorchDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(dataset_type == "train"),
                num_workers=self.num_workers
            )

    @staticmethod
    def docformer_collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'bboxes': torch.stack([x['bboxes'] for x in batch]),
            'labels': torch.stack([x['label'] for x in batch])
        }

    def load_data(self):
        loaders = {}
        if self.dataset_type in ["train", "all"]:
            loaders["train"] = self.get_data_loader("train")
        if self.dataset_type in ["val", "all"]:
            loaders["val"] = self.get_data_loader("val")
        if self.dataset_type in ["test", "all"]:
            loaders["test"] = self.get_data_loader("test")
        return loaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified DataLoader for EAML and DocFormer")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dataset_type", type=str, default="all")
    parser.add_argument("--use_eaml", action="store_true")
    parser.add_argument("--use_docformer", action="store_true")
    
    args = parser.parse_args()
    
    loader = DataLoader(**vars(args))
    data_loaders = loader.load_data()
    print(f"Initialized loaders for: {list(data_loaders.keys())}")
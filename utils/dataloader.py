import os
import json
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from transformers import BertTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from typing import Optional, Dict, List, Union

# ==================== EAML Components ====================
class EAML_Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None, class_list: Optional[List[str]] = None):
        """
        EAML Dataset with OCR text extraction and class list support
        
        Args:
            data_dir: Path to dataset directory
            transform: Image transformations
            class_list: List of class names
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_list = class_list
        
        # Initializing OCR components (lazy loading)
        self.ocr_processor = None
        self.ocr_model = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        self._load_samples()
        self._build_class_mappings()

    def _build_class_mappings(self):
        """
        Build class to index mappings
        """
        if self.class_list is None:
            # Inferring classes from directory structure
            self.class_list = sorted({label for _, _, label in self.samples})
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def _initialize_ocr(self):
        """
        Lazy initialization of OCR components
        """
        if self.ocr_processor is None:
            self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            # Freezing OCR model
            for param in self.ocr_model.parameters():
                param.requires_grad = False
            self.ocr_model.eval()

    def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using TrOCR
        """
        self._initialize_ocr()
        pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self.ocr_model.generate(pixel_values)
        return self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _load_samples(self):
        """
        Load samples from directory structure
        """
        print(f"Loading EAML dataset from: {self.data_dir}")
        
        for root, _, files in os.walk(self.data_dir):
            for img_file in files:
                if img_file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, img_file)
                    label = os.path.basename(root)
                    
                    try:
                        image = Image.open(img_path).convert("RGB")
                        extracted_text = self._extract_text_from_image(image)
                        
                        tokenized_text = self.tokenizer(
                            extracted_text,
                            padding="max_length",
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        )
                        
                        self.samples.append((img_path, tokenized_text, label))
                    except Exception as e:
                        print(f"⚠️ Error processing {img_path}: {e}")
                        continue
        
        print(f"Loaded {len(self.samples)} samples with {len(set(self.class_list if self.class_list else []))} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, text_data, label = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.class_to_idx[label]
        
        return {
            'image': image,
            'text': {
                'input_ids': text_data['input_ids'].squeeze(0),
                'attention_mask': text_data['attention_mask'].squeeze(0)
            },
            'label': torch.tensor(label_idx)
        }

def eaml_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for EAML dataset
    """
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
                 img_size: int = 224, class_list: Optional[List[str]] = None):
        """
        DataLoader specifically for EAML model
        
        Args:
            data_dir: Root directory containing train/val/test subdirectories
            batch_size: Batch size
            num_workers: Number of workers for data loading
            img_size: Size for image resizing
            class_list: Optional list of class names
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.class_list = class_list
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_loader(self, split: str = 'train', shuffle: bool = True) -> TorchDataLoader:
        """
        Getting DataLoader for specific split
        
        Args:
            split: One of 'train', 'val', or 'test'
            shuffle: Whether to shuffle the data
        Returns:
            Configured DataLoader
        """
        dataset_path = os.path.join(self.data_dir, split)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset directory does not exist: {dataset_path}")
        
        dataset = EAML_Dataset(
            data_dir=dataset_path,
            transform=self.transform,
            class_list=self.class_list
        )
        
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=eaml_collate_fn
        )

    def get_class_mappings(self) -> Dict:
        """
        Get class to index mappings
        """
        sample_dataset = EAML_Dataset(
            data_dir=os.path.join(self.data_dir, 'train'),
            transform=self.transform,
            class_list=self.class_list
        )
        return {
            'class_to_idx': sample_dataset.class_to_idx,
            'idx_to_class': sample_dataset.idx_to_class,
            'classes': sample_dataset.class_list
        }

# ==================== DocFormer Components ====================
class DocFormerDataset(Dataset):
    """
    DocFormer-compatible dataset
    """
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

def docformer_collate_fn(batch):
    """
    Collate function for DocFormer dataset
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bboxes': torch.stack([x['bboxes'] for x in batch]),
        'labels': torch.stack([x['label'] for x in batch])
    }

class DataLoader:
    """
    Original DocFormer DataLoader implementation"""
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

    def get_loader(self, split='train', shuffle=True):
        dataset = DocFormerDataset(
            data_dir=self.data_dir,
            split=split
        )
        
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=docformer_collate_fn
        )

    def get_class_mappings(self):
        sample_dataset = DocFormerDataset(
            data_dir=self.data_dir,
            split='train'
        )
        return {
            'class_to_idx': sample_dataset.class_to_idx,
            'idx_to_class': sample_dataset.idx_to_class,
            'classes': list(sample_dataset.class_to_idx.keys())
        }

# ==================== Utility Functions ====================
def load_class_list(class_list_path: Optional[str] = None) -> Optional[List[str]]:
    """
    Load class list from JSON file
    
    Args:
        class_list_path: Path to JSON file containing class list
    Returns:
        List of class names or None if file not found
    """
    if class_list_path and os.path.exists(class_list_path):
        with open(class_list_path) as f:
            return json.load(f)
    return None
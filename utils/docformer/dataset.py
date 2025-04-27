import os
import logging
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class RVLCDIPDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name="microsoft/layoutlm-base-uncased", 
                 max_seq_length=512, split="train", classes=None, config=None):
        self.data_dir = os.path.join(data_dir, split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.split = split
        self.config = config
        self.target_classes = set([c.lower() for c in classes]) if classes else None
        self.device = torch.device(config.device if config else "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize OCR in main process only
        self._init_ocr()
        
        # Load samples
        self.samples = []
        self.class_names = []
        self._load_samples()
        
        # Create label mappings
        unique_classes = sorted(set(self.class_names))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        self.idx_to_class = {idx: idx for idx in range(len(unique_classes))}  # Simplified for debugging

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _init_ocr(self):
        """Initialize OCR model in main process only"""
        try:
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            self.ocr_model.to(self.device)
            self.ocr_model.eval()
            
            # Freeze OCR model
            for param in self.ocr_model.parameters():
                param.requires_grad = False
                
            logger.info("OCR model initialized on %s", self.device)
        except Exception as e:
            logger.error("Failed to initialize OCR model: %s", str(e))
            raise

    def _load_samples(self):
        """Load all samples from directory structure"""
        for class_name in os.listdir(self.data_dir):
            if self.target_classes and class_name.lower() not in self.target_classes:
                continue
                
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'class_name': class_name.lower()
                        })
                        self.class_names.append(class_name.lower())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Remove .to(self.device) here - keep on CPU
            pixel_values = self.transform(image)
            
            # Extract OCR data
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt")  # Keep on CPU
                outputs = self.ocr_model.generate(**inputs.to(self.device))  # Only model needs GPU
                text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Tokenize text (keep on CPU)
            encoding = self.tokenizer(text, return_tensors="pt", padding='max_length', 
                                    truncation=True, max_length=self.max_seq_length)
            
            # Create dummy bboxes (keep on CPU)
            bboxes = torch.zeros((self.max_seq_length, 8), dtype=torch.float32)
            
            return {
                'pixel_values': pixel_values,  # CPU
                'input_ids': encoding['input_ids'].squeeze(0),  # CPU
                'attention_mask': encoding['attention_mask'].squeeze(0),  # CPU
                'bboxes': bboxes,  # CPU
                'label': torch.tensor(self.class_to_idx[sample['class_name']], dtype=torch.long)  # CPU
            }
        except Exception as e:
            logger.error("Error processing sample %s: %s", sample['image_path'], str(e))
            empty_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)  # CPU
            return {
                'pixel_values': empty_tensor,
                'input_ids': torch.zeros(self.max_seq_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_seq_length, dtype=torch.long),
                'bboxes': torch.zeros((self.max_seq_length, 8), dtype=torch.float32),
                'label': torch.tensor(0, dtype=torch.long)
            }
def collate_fn(batch):
    """Custom collate function ensuring all tensors are on same device"""
    device = batch[0]['pixel_values'].device
    
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bboxes': torch.stack([x['bboxes'] for x in batch]),
        'labels': torch.stack([x['label'] for x in batch])
    }
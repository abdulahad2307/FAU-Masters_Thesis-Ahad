import os
import logging
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
from typing import List, Dict, Optional
from ocr_engine import OCRWrapper

logger = logging.getLogger(__name__)

class RVLCDIPDataset(Dataset):
    def __init__(self, config, data_dir: str, split: str = "train"):
        self.config = config
        self.data_dir = os.path.join(data_dir, split)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ocr_engine = OCRWrapper(config)
        
        # Initialize OCR cache
        self.ocr_cache = {}
        if not os.path.exists(config.ocr_cache_dir):
            os.makedirs(config.ocr_cache_dir, exist_ok=True)
            
        # Load samples with parallel processing
        self.samples = self._load_samples()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        samples = []
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        samples.append({
                            'img_path': img_path,
                            'class': class_dir.lower()
                        })
        return samples

    def _get_ocr_results(self, img_path: str) -> Dict:
        """Get OCR results with caching"""
        cache_path = os.path.join(self.config.ocr_cache_dir, 
                                os.path.basename(img_path) + ".pt")
        
        if os.path.exists(cache_path):
            return torch.load(cache_path)
            
        img = Image.open(img_path).convert('RGB')
        ocr_result = self.ocr_engine.process_batch([img])[0]
        
        # Convert to tensor and save
        ocr_result['bboxes'] = torch.tensor(ocr_result['bboxes'], dtype=torch.float32)
        torch.save(ocr_result, cache_path)
        
        return ocr_result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            # Load image and OCR results
            img = Image.open(sample['img_path']).convert('RGB')
            ocr_data = self._get_ocr_results(sample['img_path'])
            
            # Process image
            pixel_values = self.transform(img).to(self.config.device)
            
            # Tokenize text with dynamic padding
            encoding = self.tokenizer(
                ocr_data['text'],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.config.max_position_embeddings
            )
            
            # Process bounding boxes
            if len(ocr_data['bboxes']) > 0:
                bboxes = torch.zeros((self.config.max_position_embeddings, 8), 
                                   dtype=torch.float32)
                valid_bboxes = ocr_data['bboxes'][:self.config.max_position_embeddings]
                bboxes[:len(valid_bboxes)] = valid_bboxes
            else:
                bboxes = torch.zeros((self.config.max_position_embeddings, 8), 
                                   dtype=torch.float32)
                
            return {
                'pixel_values': pixel_values,
                'input_ids': encoding['input_ids'].squeeze(0).to(self.config.device),
                'attention_mask': encoding['attention_mask'].squeeze(0).to(self.config.device),
                'bboxes': bboxes.to(self.config.device),
                'label': torch.tensor(self.class_to_idx[sample['class']], 
                                    dtype=torch.long).to(self.config.device)
            }
        except Exception as e:
            logger.error(f"Error processing {sample['img_path']}: {str(e)}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        return {
            'pixel_values': torch.zeros((3, 224, 224), dtype=torch.float32).to(self.config.device),
            'input_ids': torch.zeros(self.config.max_position_embeddings, dtype=torch.long).to(self.config.device),
            'attention_mask': torch.zeros(self.config.max_position_embeddings, dtype=torch.long).to(self.config.device),
            'bboxes': torch.zeros((self.config.max_position_embeddings, 8), dtype=torch.float32).to(self.config.device),
            'label': torch.tensor(0, dtype=torch.long).to(self.config.device)
        }

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
from typing import List, Dict, Optional
from utils.docformer.ocr_engine import UnifiedOCREngine

class RVLCDIPDataset(Dataset):
    def __init__(self, config, data_dir: str, split: str = "train"):
        self.config = config
        self.data_dir = os.path.join(data_dir, split)
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Ensure OCR cache directory exists
        os.makedirs(config.ocr_cache_dir, exist_ok=True)
        
        # Initialize OCR engine
        self.ocr_engine = UnifiedOCREngine(config)
        
        # Load samples
        self.samples = self._load_samples()
        self.class_names = [sample['class'] for sample in self.samples]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.samples)} samples from {split} split")

    def _load_samples(self):
        """Load samples from directory structure"""
        samples = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return samples
            
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
        # Create cache filename
        img_name = os.path.basename(img_path)
        cache_filename = f"{self.config.ocr_engine}_{img_name}.pt"
        cache_path = os.path.join(self.config.ocr_cache_dir, cache_filename)
        
        # Load from cache if exists
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location='cpu')
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {str(e)}")
        
        # Process with OCR
        try:
            img = Image.open(img_path).convert('RGB')
            ocr_result = self.ocr_engine.process_batch([img])[0]
            
            # Convert bboxes to tensor
            if len(ocr_result['bboxes']) > 0:
                # Ensure we have 8 coordinates per bbox
                processed_bboxes = []
                for bbox in ocr_result['bboxes']:
                    if len(bbox) == 4:  # x1, y1, x2, y2
                        x1, y1, x2, y2 = bbox
                        bbox_8 = [x1, y1, x2, y1, x2, y2, x1, y2]
                    elif len(bbox) >= 8:  # Already 8 coordinates
                        bbox_8 = bbox[:8]
                    else:
                        bbox_8 = [0, 0, 0, 0, 0, 0, 0, 0]  # Default
                    processed_bboxes.append(bbox_8)
                
                ocr_result['bboxes'] = torch.tensor(processed_bboxes, dtype=torch.float32)
            else:
                ocr_result['bboxes'] = torch.zeros((0, 8), dtype=torch.float32)
            
            # Save to cache
            torch.save(ocr_result, cache_path)
            return ocr_result
            
        except Exception as e:
            print(f"OCR processing failed for {img_path}: {str(e)}")
            return {"text": "[UNK]", "bboxes": torch.zeros((0, 8), dtype=torch.float32)}

    def filter_classes(self, target_classes):
        """Filter samples to only include specified classes"""
        if not target_classes:
            return
            
        target_classes_set = set([cls.lower() for cls in target_classes])
        filtered_samples = []
        filtered_class_names = []
        
        for sample in self.samples:
            if sample['class'] in target_classes_set:
                filtered_samples.append(sample)
                filtered_class_names.append(sample['class'])
        
        self.samples = filtered_samples
        self.class_names = filtered_class_names
        print(f"Filtered to {len(self.samples)} samples from {len(target_classes)} classes")
    
    def create_class_mappings(self):
        """Create class to index mappings"""
        unique_classes = sorted(set(self.class_names))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        print(f"Created mappings for {len(unique_classes)} classes: {list(unique_classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            # Load and transform image
            img = Image.open(sample['img_path']).convert('RGB')
            pixel_values = self.transform(img)
            
            # Get OCR results
            ocr_data = self._get_ocr_results(sample['img_path'])
            
            # Process text
            text = ocr_data['text'] if ocr_data['text'] and ocr_data['text'] != "[UNK]" else ""
            
            # Add special tokens for BERT
            if text:
                text = "[CLS] " + text + " [SEP]"
            else:
                text = "[CLS] [SEP]"
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.config.max_position_embeddings
            )
            
            # Process bounding boxes
            max_seq_len = self.config.max_position_embeddings
            bboxes = torch.zeros((max_seq_len, 8), dtype=torch.float32)
            
            if len(ocr_data['bboxes']) > 0:
                # Add [CLS] bbox (full page)
                cls_bbox = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
                bboxes[0] = cls_bbox
                
                # Add OCR bboxes
                valid_bboxes = ocr_data['bboxes'][:max_seq_len-2]  # Reserve space for [CLS] and [SEP]
                if len(valid_bboxes) > 0:
                    bboxes[1:len(valid_bboxes)+1] = valid_bboxes
                
                # Add [SEP] bbox (same as [CLS])
                if len(valid_bboxes) + 1 < max_seq_len:
                    bboxes[len(valid_bboxes)+1] = cls_bbox
            else:
                # No OCR bboxes, use [CLS] bbox for all positions
                cls_bbox = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
                bboxes[0] = cls_bbox  # [CLS]
                bboxes[1] = cls_bbox  # [SEP]
            
            return {
                'pixel_values': pixel_values,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'bboxes': bboxes,
                'labels': torch.tensor(self.class_to_idx[sample['class']], dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        """Return empty sample for error cases"""
        max_seq_len = self.config.max_position_embeddings
        cls_bbox = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        
        # Create bboxes tensor
        bboxes = torch.zeros((max_seq_len, 8), dtype=torch.float32)
        bboxes[0] = cls_bbox  # [CLS] token bbox
        bboxes[1] = cls_bbox  # [SEP] token bbox
        
        # Create attention mask for [CLS] and [SEP] only
        attention_mask = torch.zeros(max_seq_len, dtype=torch.long)
        attention_mask[0] = 1  # [CLS]
        attention_mask[1] = 1  # [SEP]
        
        # Create input_ids for [CLS] and [SEP]
        input_ids = torch.zeros(max_seq_len, dtype=torch.long)
        input_ids[0] = self.tokenizer.cls_token_id
        input_ids[1] = self.tokenizer.sep_token_id
        
        return {
            'pixel_values': torch.zeros((3, 224, 224), dtype=torch.float32),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bboxes': bboxes,
            'labels': torch.tensor(0, dtype=torch.long)
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bboxes': torch.stack([x['bboxes'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

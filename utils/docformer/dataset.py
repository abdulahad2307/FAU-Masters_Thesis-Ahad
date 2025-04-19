import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, TrOCRProcessor
from torchvision import transforms

class RVLCDIPDataset(Dataset):
    """
    Dataset class for RVL-CDIP documents with TrOCR for text extraction
    """
    def __init__(self, data_dir, tokenizer_name="bert-base-uncased", max_seq_length=512, split="train", classes=None):
        self.data_dir = os.path.join(data_dir, split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten", 
            use_fast=True,
            do_resize=True,
            do_normalize=True,
            size=224
        )
        self.max_seq_length = max_seq_length
        self.split = split
        
        # Use provided classes or all classes if None
        self.target_classes = set([c.lower() for c in classes]) if classes else None
        
        # Collect all samples
        self.samples = []
        self.class_names = []
        self._load_samples()
        
        # Create label mappings
        unique_classes = sorted(set(self.class_names))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(unique_classes)}
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        """Load all samples from the directory structure, filtering for target classes if specified"""
        for class_name in os.listdir(self.data_dir):
            # Skip classes not in our target list if specified
            if self.target_classes and class_name.lower() not in self.target_classes:
                continue
                
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'class_name': class_name.lower()  # Ensure consistent case
                        })
                        self.class_names.append(class_name.lower())

    def __len__(self):
        return len(self.samples)

    def _extract_text_and_bboxes(self, image):
        """Extract text and bounding boxes using TrOCR"""
        # Process image with the processor
        inputs = self.processor(images=image, return_tensors="pt")
        # Return dummy text and whole image bbox
        return ["document"], [[0, 0, image.width, 0, image.width, image.height, 0, image.height]]


    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Process image
        pixel_values = self.transform(image)
        
        # Extract OCR data (simplified for DocFormer)
        words, bboxes = self._extract_text_and_bboxes(image)
        
        # Initialize lists for tokenized data
        input_ids = []
        bbox_tensors = []
        attention_mask = []
        
        # Add [CLS] token
        input_ids.append(self.tokenizer.cls_token_id)
        bbox_tensors.append([0]*8)  # Dummy bbox for special tokens
        attention_mask.append(1)
        
        # Process each word
        for word, bbox in zip(words, bboxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            
            # Extend bbox for each token
            for _ in word_tokens:
                bbox_tensors.append(bbox)
                attention_mask.append(1)
            input_ids.extend(token_ids)
        
        # Add [SEP] token
        input_ids.append(self.tokenizer.sep_token_id)
        bbox_tensors.append([0]*8)
        attention_mask.append(1)
        
        # Convert to tensors and normalize bboxes
        bboxes = torch.tensor(bbox_tensors, dtype=torch.float32)
        
        # Normalize coordinates to [0,1] range
        bboxes[:, 0::2] /= image.width   # Normalize x coordinates (0,2,4,6)
        bboxes[:, 1::2] /= image.height  # Normalize y coordinates (1,3,5,7)
        
        # Clamp to ensure no values outside [0,1]
        bboxes = torch.clamp(bboxes, 0, 1)
        
        # Truncate/pad sequences
        input_ids = input_ids[:self.max_seq_length]
        attention_mask = attention_mask[:self.max_seq_length]
        bboxes = bboxes[:self.max_seq_length]
        
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        bboxes = torch.cat([
            bboxes,
            torch.zeros((padding_length, 8), dtype=torch.float32)
        ])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'bboxes': bboxes,  # Now properly normalized [0,1]
            'label': torch.tensor(self.class_to_idx[sample['class_name']])
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bboxes': torch.stack([x['bboxes'] for x in batch]),
        'labels': torch.stack([x['label'] for x in batch])
    }
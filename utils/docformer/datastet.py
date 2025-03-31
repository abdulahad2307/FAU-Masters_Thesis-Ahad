import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class DocFormerDataset(Dataset):
    """
    Base dataset class for DocFormer
    """
    def __init__(self, data_dir, tokenizer_name="microsoft/layoutlm-base-uncased", max_seq_length=512, split="train"):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.split = split
        
        # Load annotations
        with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        image_path = os.path.join(self.data_dir, "images", annotation["image_path"])
        image = Image.open(image_path).convert("RGB")
        
        # Text Tokenization and get bounding boxes
        words = annotation["words"]
        bboxes = annotation["bboxes"]
        
        # Tokenizing words and align bounding boxes
        tokens = []
        token_bboxes = []
        for word, bbox in zip(words, bboxes):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_bboxes.extend([bbox] * len(word_tokens))
        
        # Truncate if too long
        if len(tokens) > self.max_seq_length - 2:  # Account for [CLS] and [SEP]
            tokens = tokens[:self.max_seq_length - 2]
            token_bboxes = token_bboxes[:self.max_seq_length - 2]
        
        # Adding special tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        token_bboxes = [[0, 0, 0, 0, 0, 0, 0, 0]] + token_bboxes + [[0, 0, 0, 0, 0, 0, 0, 0]]
        
        # Converting to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Creating attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad to max length
        padding_length = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_bboxes = token_bboxes + [[0, 0, 0, 0, 0, 0, 0, 0]] * padding_length
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        bboxes = torch.tensor(token_bboxes, dtype=torch.float32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bboxes": bboxes,
            "image": image,
            "words": words,
            "labels": torch.tensor(annotation.get("labels", []), dtype=torch.long)
        }

class FUNSDDataset(DocFormerDataset):
    """Dataset for FUNSD form understanding task"""
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # For FUNSD, we need to add entity labels
        annotation = self.annotations[idx]
        
        # Align labels with tokens
        labels = []
        current_label = 0
        for word, label in zip(annotation["words"], annotation["labels"]):
            word_tokens = self.tokenizer.tokenize(word)
            labels.extend([label] * len(word_tokens))
        
        # Truncate if needed
        if len(labels) > self.max_seq_length - 2:
            labels = labels[:self.max_seq_length - 2]
        
        # Add labels for [CLS] and [SEP]
        labels = [0] + labels + [0]
        
        # Pad
        padding_length = self.max_seq_length - len(labels)
        labels = labels + [0] * padding_length
        
        data["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return data

class CORDDataset(DocFormerDataset):
    """Dataset for CORD receipts dataset"""
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # For CORD, we need to add receipt item labels
        annotation = self.annotations[idx]
        
        # Align labels with tokens
        labels = []
        for word, label in zip(annotation["words"], annotation["labels"]):
            word_tokens = self.tokenizer.tokenize(word)
            labels.extend([label] * len(word_tokens))
        
        # Truncate if needed
        if len(labels) > self.max_seq_length - 2:
            labels = labels[:self.max_seq_length - 2]
        
        # Add labels for [CLS] and [SEP]
        labels = [0] + labels + [0]
        
        # Pad
        padding_length = self.max_seq_length - len(labels)
        labels = labels + [0] * padding_length
        
        data["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return data

def collate_fn(batch):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    bboxes = torch.stack([item["bboxes"] for item in batch])
    
    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bboxes": bboxes,
            "labels": labels
        }
    else:
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bboxes": bboxes
        }
import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def save_checkpoint(model, optimizer, epoch, path):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path, device):
    """Load training checkpoint"""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # Handle both model types
        if "images" in batch:  # EAML
            inputs = {
                'images': batch['images'].to(device),
                'input_ids': batch['texts']['input_ids'].to(device),
                'attention_mask': batch['texts']['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'bboxes': batch['bboxes'].to(device)
            }
            labels = batch['labels'].to(device)
            outputs = model(**inputs, task="classification")
        
        # Extract logits from model output (which is a dictionary)
        logits = outputs['logits']
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item()
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Train Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")
    return acc

def evaluate(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if "images" in batch:  # EAML
                inputs = {
                    'images': batch['images'].to(device),
                    'input_ids': batch['texts']['input_ids'].to(device),
                    'attention_mask': batch['texts']['attention_mask'].to(device)
                }
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
            else:  # DocFormer
                inputs = {
                    'pixel_values': batch['pixel_values'].to(device),
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'bboxes': batch['bboxes'].to(device)
                }
                labels = batch['labels'].to(device)
                outputs = model(**inputs, task="classification")
            
            # Extract logits from model output
            logits = outputs['logits']
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Evaluation Accuracy: {acc:.4f}")
    return acc
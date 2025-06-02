import os
import torch
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

class CILMetrics:
    def __init__(self, class_names):
        self.num_classes = len(class_names)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.current_state = 0
        self.class_mapping = {0: class_names}

    def update(self, preds, labels):
        for p, l in zip(preds, labels):
            if 0 <= l < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[l, p] += 1
            else:
                print(f"Warning: Label {l} or prediction {p} out of bounds (num_classes={self.num_classes})")

    def get_metrics(self):
        metrics = {}
        total = np.sum(self.confusion_matrix)
        metrics['top1_acc'] = np.trace(self.confusion_matrix) / total if total > 0 else 0
        if self.current_state > 0:
            prev_classes = []
            for s in range(self.current_state):
                prev_classes.extend(self.class_mapping[s])
            n_prev = len(prev_classes)
            if n_prev > 0:
                past_correct = np.trace(self.confusion_matrix[:n_prev, :n_prev])
                past_total = np.sum(self.confusion_matrix[:n_prev, :])
                metrics['past_acc'] = past_correct / past_total if past_total > 0 else 0
                new_correct = np.trace(self.confusion_matrix[n_prev:, n_prev:])
                new_total = np.sum(self.confusion_matrix[n_prev:, :])
                metrics['new_acc'] = new_correct / new_total if new_total > 0 else 0
                metrics['e(p,p)'] = np.sum(self.confusion_matrix[:n_prev, :n_prev]) - past_correct
                metrics['e(p,n)'] = np.sum(self.confusion_matrix[:n_prev, n_prev:])
                metrics['e(n,p)'] = np.sum(self.confusion_matrix[n_prev:, :n_prev])
                metrics['e(n,n)'] = np.sum(self.confusion_matrix[n_prev:, n_prev:]) - new_correct
        return metrics

    def incremental_state_update(self, new_classes):
        self.current_state += 1
        self.class_mapping[self.current_state] = new_classes
        old_size = self.num_classes
        new_size = old_size + len(new_classes)
        new_matrix = np.zeros((new_size, new_size))
        new_matrix[:old_size, :old_size] = self.confusion_matrix
        self.confusion_matrix = new_matrix
        self.num_classes = new_size

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']

def train_one_epoch_cil(model, dataloader, optimizer, criterion, device, metrics):
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
        if isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
        total_loss += loss.item()
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Train Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")

    epoch_metrics = metrics.get_metrics()
    print(f"Train Loss: {total_loss/len(dataloader):.4f} | Acc: {epoch_metrics['top1_acc']:.4f}")
    return epoch_metrics

def evaluate(model, dataloader, device, metrics,full_model_acc, evm=None, use_evm=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                if "images" in batch:  # EAML
                    images = batch['images'].to(device)
                    texts = {k: v.to(device) for k, v in batch['texts'].items()}
                    labels = batch['labels'].to(device)
                    outputs = model(images=images, texts=texts)
                    logits = outputs
                else:  # DocFormer
                    inputs = {
                        'pixel_values': batch['pixel_values'].to(device),
                        'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device),
                        'bboxes': batch['bboxes'].to(device)
                    }
                    labels = batch['labels'].to(device)
                    outputs = model(**inputs, task="classification")
                    logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch during evaluation: {e}")
                continue
    acc = accuracy_score(all_labels, all_preds)
    eval_metrics = metrics.get_metrics()
    g_il = None
    if full_model_acc is not None and 'top1_acc' in eval_metrics:
        g_il = (eval_metrics['top1_acc']- full_model_acc) / (1 - full_model_acc)
        print(f"Incremental Learning Gap (G_IL): {g_il:.4f}")
    print("\nEvaluation Results:")
    print(f"Total Accuracy: {eval_metrics.get('top1_acc', acc):.4f}")
    if 'past_acc' in eval_metrics:
        print(f"Previous Classes Accuracy: {eval_metrics['past_acc']:.4f}")
        print(f"New Classes Accuracy: {eval_metrics['new_acc']:.4f}")
    if g_il is not None:
        print(f"Incremental Learning Gap (G_IL): {g_il:.4f}")
    return eval_metrics

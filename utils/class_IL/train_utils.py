import os
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

class CILMetrics:
    def __init__(self, num_classes):
        self.num_classes = len(num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.current_state = 0
        self.class_mapping = {0: num_classes}
    """
    def update(self, preds, labels):
        for p, l in zip(preds, labels):
            self.confusion_matrix[l][p] += 1
    """ 
    def update(self, preds, labels):
        # Ensure preds and labels are within bounds
        for p, l in zip(preds, labels):
            if 0 <= l < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[l, p] += 1  # Use tuple indexing
            else:
                print(f"Warning: Label {l} or prediction {p} out of bounds (num_classes={self.num_classes})")
      
    def get_metrics(self):
        metrics = {}
        
        # Overall accuracy
        metrics['top1_acc'] = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
        
        if self.current_state > 0:
            # Get all classes up to previous state
            prev_classes = []
            for s in range(self.current_state):
                prev_classes.extend(self.class_mapping[s])
            n_prev = len(prev_classes)
            
            if n_prev > 0:  # Only if we have past classes
                # Past class metrics
                past_correct = np.trace(self.confusion_matrix[:n_prev, :n_prev])
                past_total = np.sum(self.confusion_matrix[:n_prev, :])
                metrics['past_acc'] = past_correct / past_total if past_total > 0 else 0
                
                # New class metrics
                new_correct = np.trace(self.confusion_matrix[n_prev:, n_prev:])
                new_total = np.sum(self.confusion_matrix[n_prev:, :])
                metrics['new_acc'] = new_correct / new_total if new_total > 0 else 0
                
                # Error types
                metrics['e(p,p)'] = np.sum(self.confusion_matrix[:n_prev, :n_prev]) - past_correct
                metrics['e(p,n)'] = np.sum(self.confusion_matrix[:n_prev, n_prev:])
                metrics['e(n,p)'] = np.sum(self.confusion_matrix[n_prev:, :n_prev])
                metrics['e(n,n)'] = np.sum(self.confusion_matrix[n_prev:, n_prev:]) - new_correct
        
        return metrics
    
    """
    def incremental_state_update(self, new_classes):
        self.current_state += 1
        self.class_mapping[self.current_state] = new_classes
        # Expand confusion matrix for new classes
        new_total_classes = self.num_classes + len(new_classes)
        new_matrix = np.zeros((new_total_classes, new_total_classes))
        new_matrix[:self.num_classes, :self.num_classes] = self.confusion_matrix
        self.confusion_matrix = new_matrix
        self.num_classes = new_total_classes
    """

    def incremental_state_update(self, new_classes):
        self.current_state += 1
        self.class_mapping[self.current_state] = new_classes
        
        # Expand confusion matrix for new classes
        old_size = self.num_classes
        new_size = old_size + len(new_classes)
        
        # Create new larger matrix
        new_matrix = np.zeros((new_size, new_size))
        
        # Copy old values
        new_matrix[:old_size, :old_size] = self.confusion_matrix
        
        # Update
        self.confusion_matrix = new_matrix
        self.num_classes = new_size

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

def train_one_epoch(model, dataloader, optimizer, criterion, device, metrics):
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
        metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
        total_loss += loss.item()
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Train Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")

    epoch_metrics = metrics.get_metrics()
    print(f"Train Loss: {total_loss/len(dataloader):.4f} | Acc: {epoch_metrics['top1_acc']:.4f}")
    return epoch_metrics
    #return acc
    """
def evaluate(model, dataloader, device, metrics):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                if "images" in batch:  # EAML
                    inputs = {
                        'images': batch['images'].to(device),
                        'texts': batch['texts']  # Pass as dictionary
                    }
                    # Move text tensors to device
                    inputs['texts'] = {k: v.to(device) for k, v in inputs['texts'].items()}
                    
                    labels = batch['labels'].to(device)
                    outputs = model(**inputs)
                    logits = outputs  # For EAML, outputs are logits
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
                
                # Store predictions and labels for overall accuracy
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Safely update metrics with bounds checking
                p_np = preds.cpu().numpy()
                l_np = labels.cpu().numpy()
                
                # Filter out-of-bounds values
                valid_indices = np.logical_and(
                    np.logical_and(p_np >= 0, p_np < metrics.num_classes),
                    np.logical_and(l_np >= 0, l_np < metrics.num_classes)
                )
                
                if not np.all(valid_indices):
                    invalid_count = np.sum(~valid_indices)
                    print(f"Warning: Found {invalid_count} out-of-bounds indices. "
                          f"Max pred: {np.max(p_np)}, Max label: {np.max(l_np)}, "
                          f"Num classes: {metrics.num_classes}")
                
                # Only update with valid indices
                if np.any(valid_indices):
                    metrics.update(p_np[valid_indices], l_np[valid_indices])
            
            except Exception as e:
                print(f"Error processing batch during evaluation: {e}")
                continue
    
    try:
        # Calculate overall accuracy using sklearn
        acc = accuracy_score(all_labels, all_preds)
        print(f"Evaluation Accuracy: {acc:.4f}")
        
        # Get metrics from confusion matrix
        eval_metrics = metrics.get_metrics()
        
        print("\nEvaluation Metrics:")
        print(f"- Top1 Accuracy: {eval_metrics.get('top1_acc', 0):.4f}")
        
        if 'past_acc' in eval_metrics:
            print(f"- Past Classes Accuracy: {eval_metrics['past_acc']:.4f}")
            print(f"- New Classes Accuracy: {eval_metrics['new_acc']:.4f}")
            print(f"- Past->Past Errors: {eval_metrics['e(p,p)']}")
            print(f"- Past->New Errors: {eval_metrics['e(p,n)']}")
            print(f"- New->Past Errors: {eval_metrics['e(n,p)']}")
            print(f"- New->New Errors: {eval_metrics['e(n,n)']}")
        
        return eval_metrics
    except Exception as e:
        print(f"Error calculating evaluation metrics: {e}")
        # Return basic metrics if confusion matrix calculation fails
        return {'top1_acc': acc if 'acc' in locals() else 0}
        """
    #return acc

def evaluate(model, dataloader, device, metrics):
    """Evaluate model performance with robust error handling"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                if "images" in batch:  # EAML
                    images = batch['images'].to(device)
                    texts = {k: v.to(device) for k, v in batch['texts'].items()}
                    labels = batch['labels'].to(device)
                    
                    # Forward pass for EAML
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
                
                # Store predictions and labels for overall accuracy
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Safely update metrics with bounds checking
                p_np = preds.cpu().numpy()
                l_np = labels.cpu().numpy()
                
                # Filter out-of-bounds values
                valid_indices = np.logical_and(
                    np.logical_and(p_np >= 0, p_np < metrics.num_classes),
                    np.logical_and(l_np >= 0, l_np < metrics.num_classes)
                )
                
                if not np.all(valid_indices):
                    invalid_count = np.sum(~valid_indices)
                    print(f"Warning: Found {invalid_count} out-of-bounds indices.")
                
                # Only update with valid indices
                if np.any(valid_indices):
                    metrics.update(p_np[valid_indices], l_np[valid_indices])
            
            except Exception as e:
                print(f"Error processing batch during evaluation: {e}")
                continue
    
    # Calculate overall accuracy
    acc = accuracy_score(all_labels, all_preds)
    
    # Get detailed metrics
    eval_metrics = metrics.get_metrics()
    
    # Calculate G_IL (Incremental Learning Gap)
    # Assuming full model accuracy is 0.85 (replace with actual value)
    full_model_acc = 0.85  # Replace with your full model accuracy
    g_il = None
    if 'top1_acc' in eval_metrics:
        current_acc = eval_metrics['top1_acc']
        g_il = (current_acc - full_model_acc) / (1 - full_model_acc)
    
    # Print metrics in the desired format
    print("\nEvaluation Results:")
    print(f"Total Accuracy (Previous + New Classes): {eval_metrics.get('top1_acc', acc):.4f}")
    
    if 'past_acc' in eval_metrics:
        print(f"Previous Classes Accuracy: {eval_metrics['past_acc']:.4f}")
        print(f"New Classes Accuracy: {eval_metrics['new_acc']:.4f}")
    
    if g_il is not None:
        print(f"Incremental Learning Gap (G_IL): {g_il:.4f}")
    
    # Print additional metrics for debugging
    if 'past_acc' in eval_metrics:
        print("\nDetailed Error Analysis:")
        print(f"- Past->Past Errors: {eval_metrics['e(p,p)']}")
        print(f"- Past->New Errors: {eval_metrics['e(p,n)']}")
        print(f"- New->Past Errors: {eval_metrics['e(n,p)']}")
        print(f"- New->New Errors: {eval_metrics['e(n,n)']}")
    
    return eval_metrics

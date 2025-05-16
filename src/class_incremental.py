import os
import time
import torch
import torch.nn as nn
from typing import List
import copy

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig
from utils.class_IL.dataloader_utils import get_class_il_loader
from utils.class_IL.train_utils import save_checkpoint, load_checkpoint, train_one_epoch, evaluate, CILMetrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IncrementalStrategy:
    """Base class for incremental learning strategies"""
    def __init__(self, device):
        self.device = device
        
    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model for new classes"""
        raise NotImplementedError
        
    def compute_loss(self, model, batch, criterion, old_model=None):
        """Compute loss with strategy-specific components"""
        raise NotImplementedError

class StandardIncremental(IncrementalStrategy):
    """Standard incremental learning without forgetting mitigation"""
    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model architecture for new classes"""
        if model_name == "docformer" and hasattr(model, 'classifier'):
            # Save old classifier weights
            old_classifier = model.classifier.weight.data.clone()
            old_bias = model.classifier.bias.data.clone() if model.classifier.bias is not None else None
            
            # Initialize new classifier
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, new_num_classes)
            
            # Copy old weights
            with torch.no_grad():
                model.classifier.weight.data[:old_num_classes] = old_classifier
                if old_bias is not None and model.classifier.bias is not None:
                    model.classifier.bias.data[:old_num_classes] = old_bias
        
        elif model_name == "eaml":
            # Save old classifier weights
            old_image_classifier = model.image_classifier.weight.data.clone()
            old_image_bias = model.image_classifier.bias.data.clone()
            
            old_text_classifier = model.text_classifier.weight.data.clone()
            old_text_bias = model.text_classifier.bias.data.clone()
            
            old_fusion_classifier = model.fusion_classifier.weight.data.clone()
            old_fusion_bias = model.fusion_classifier.bias.data.clone()
            
            # Initialize new classifiers
            model.image_classifier = nn.Linear(model.image_classifier.in_features, new_num_classes)
            model.text_classifier = nn.Linear(model.text_classifier.in_features, new_num_classes)
            model.fusion_classifier = nn.Linear(model.fusion_classifier.in_features, new_num_classes)
            
            # Copy old weights
            with torch.no_grad():
                model.image_classifier.weight.data[:old_num_classes] = old_image_classifier
                model.image_classifier.bias.data[:old_num_classes] = old_image_bias
                
                model.text_classifier.weight.data[:old_num_classes] = old_text_classifier
                model.text_classifier.bias.data[:old_num_classes] = old_text_bias
                
                model.fusion_classifier.weight.data[:old_num_classes] = old_fusion_classifier
                model.fusion_classifier.bias.data[:old_num_classes] = old_fusion_bias
        
        return model
        
    def compute_loss(self, model, batch, criterion, old_model=None):
        # Handle both model types
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            texts = batch['texts']
            # Move text tensors to device
            texts = {k: v.to(self.device) for k, v in texts.items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = model(images=images, texts=texts)
            logits = outputs
        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bboxes': batch['bboxes'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels

class DistillationIncremental(IncrementalStrategy):
    """Incremental learning with knowledge distillation"""
    def __init__(self, device, temperature=2.0, lambda_distill=1.0):
        super().__init__(device)
        self.temperature = temperature
        self.lambda_distill = lambda_distill
        
    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model architecture for new classes"""
        return StandardIncremental(self.device).adapt_model(model, old_num_classes, new_num_classes, model_name)
        
    def compute_loss(self, model, batch, criterion, old_model=None):
        """Compute loss with distillation component"""
        # Handle both model types
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            texts = batch['texts']
            # Move text tensors to device
            texts = {k: v.to(self.device) for k, v in texts.items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = model(images=images, texts=texts)
            logits = outputs
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(images=images, texts=texts)
                    old_logits = old_outputs
                
                # Only apply distillation to old classes
                old_class_count = old_logits.size(1)
                
                # Get soft targets from old model
                soft_targets = nn.functional.softmax(old_logits / self.temperature, dim=1)
                # Get soft probabilities from current model (only for old classes)
                soft_probs = nn.functional.log_softmax(logits[:, :old_class_count] / self.temperature, dim=1)
                
                # Calculate distillation loss
                dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
                
                # Combined loss
                loss = cls_loss + self.lambda_distill * dist_loss
            else:
                loss = cls_loss
                
        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bboxes': batch['bboxes'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(**inputs, task="classification")
                    old_logits = old_outputs['logits']
                
                # Only apply distillation to old classes
                old_class_count = old_logits.size(1)
                
                # Get soft targets from old model
                soft_targets = nn.functional.softmax(old_logits / self.temperature, dim=1)
                # Get soft probabilities from current model (only for old classes)
                soft_probs = nn.functional.log_softmax(logits[:, :old_class_count] / self.temperature, dim=1)
                
                # Calculate distillation loss
                dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
                
                # Combined loss
                loss = cls_loss + self.lambda_distill * dist_loss
            else:
                loss = cls_loss
        
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels

def run_incremental_learning(
    data_root: str,
    class_order: List[str],
    base_model_path: str,
    model_name: str,
    checkpoint_dir: str,
    start_step: int = 0,
    batch_size: int = 8,
    lr: float = 2e-5,
    num_epochs: int = 10,
    strategy: str = "distillation",
    temperature: float = 2.0,
    lambda_distill: float = 1.0
):
    print("Starting Class Incremental Learning...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize metrics tracker
    initial_classes = class_order[:start_step+1] if start_step > 0 else [class_order[0]]
    metrics = CILMetrics(initial_classes)
    
    # Initialize strategy
    if strategy == "distillation":
        inc_strategy = DistillationIncremental(DEVICE, temperature, lambda_distill)
    else:
        inc_strategy = StandardIncremental(DEVICE)
    
    # Initialize best accuracy tracking
    best_acc = 0.0
    
    # Keep track of the previous model for distillation
    old_model = None
    
    for step in range(start_step, len(class_order)):
        current_classes = class_order[:step + 1]
        prev_classes = class_order[:step] if step > 0 else []
        new_class = [class_order[step]] if step > start_step else []
        
        print(f"Step {step+1}/{len(class_order)} | Training on classes: {current_classes}")
        print(f"New classes: {new_class}")
        
        # Update metrics with new class information
        if step > start_step:
            metrics.incremental_state_update(new_class)
        
        # Get dataloaders
        train_loader = get_class_il_loader(
            model_type=model_name,
            data_dir=os.path.join(data_root, "train"),
            current_classes=current_classes,
            batch_size=batch_size
        )
        
        val_loader = get_class_il_loader(
            model_type=model_name,
            data_dir=os.path.join(data_root, "val"),
            current_classes=current_classes,
            batch_size=batch_size
        )
        
        # Initialize model with the correct number of classes
        if model_name == "docformer":
            config = DocFormerConfig()
            train_dataset = train_loader.dataset
            num_classes = len(train_dataset.class_to_idx)
            model = DocFormer(config, num_classes=num_classes)
            model.to(config.device)
        elif model_name == "eaml":
            if step == start_step:
                # For first step, use number of classes from base model
                model = EAMLModel(num_classes=len(initial_classes))
            else:
                # For subsequent steps, adapt model for all classes seen so far
                model = EAMLModel(num_classes=len(current_classes))
            model.to(DEVICE)
        
        # Load base model weights if first step
        if step == start_step:
            try:
                # Try to load the checkpoint
                checkpoint = torch.load(base_model_path, map_location=DEVICE, weights_only=True)
                if "model_state_dict" in checkpoint:
                    # Handle size mismatch in classifier layers
                    if model_name == "eaml":
                        # Load everything except classifier layers
                        pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items() 
                                          if not any(x in k for x in ['classifier', 'image_classifier', 'text_classifier', 'fusion_classifier'])}
                        model_dict = model.state_dict()
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)
                    else:
                        # For DocFormer, similar approach
                        pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items() 
                                          if 'classifier' not in k}
                        model_dict = model.state_dict()
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)
                else:
                    # Direct state dict
                    if model_name == "eaml":
                        pretrained_dict = {k: v for k, v in checkpoint.items() 
                                          if not any(x in k for x in ['classifier', 'image_classifier', 'text_classifier', 'fusion_classifier'])}
                        model_dict = model.state_dict()
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)
                    else:
                        pretrained_dict = {k: v for k, v in checkpoint.items() 
                                          if 'classifier' not in k}
                        model_dict = model.state_dict()
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)
                
                print(f"Loaded base model from {base_model_path}")
            except Exception as e:
                print(f"Error loading model from {base_model_path}: {e}")
                print("Initializing model with random weights.")
        elif step > start_step:
            # Save current model as old model for distillation
            if strategy == "distillation":
                if model_name == "docformer":
                    old_model = DocFormer(config, num_classes=len(prev_classes))
                else:
                    old_model = EAMLModel(num_classes=len(prev_classes))
                
                # Get previous checkpoint
                prev_step_path = os.path.join(checkpoint_dir, f"step_{step-1}_class_{class_order[step-1]}.pth")
                if os.path.exists(prev_step_path):
                    prev_checkpoint = torch.load(prev_step_path, map_location=DEVICE)
                    if "model_state_dict" in prev_checkpoint:
                        old_model.load_state_dict(prev_checkpoint["model_state_dict"])
                    else:
                        old_model.load_state_dict(prev_checkpoint)
                
                old_model.to(DEVICE)
                old_model.eval()
                
                # Adapt model for new classes
                model = inc_strategy.adapt_model(model, len(prev_classes), len(current_classes), model_name)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            
            model.train()
            total_loss = 0
            all_preds, all_labels = [], []
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                try:
                    # Compute loss using the selected strategy
                    loss, preds, labels = inc_strategy.compute_loss(model, batch, criterion, old_model)
                    
                    loss.backward()
                    optimizer.step()
                    
                    all_preds.extend(preds.detach().cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                    
                    metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
                    total_loss += loss.item()
                    
                    # Clear cache periodically to avoid OOM
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA OOM in batch {batch_idx}. Skipping batch and clearing cache.")
                        torch.cuda.empty_cache()
                        if optimizer.param_groups[0]['lr'] > 1e-6:
                            optimizer.param_groups[0]['lr'] *= 0.9
                            print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
                        continue
                    else:
                        raise e
            
            # Evaluate
            try:
                val_metrics = evaluate(model, val_loader, DEVICE, metrics)
                epoch_time = time.time() - start_time
                
                print(f"Epoch Time: {epoch_time:.2f}s | Val Acc: {val_metrics['top1_acc']:.4f}")
                
                # Save best model based on validation accuracy
                if val_metrics['top1_acc'] > best_acc:
                    best_acc = val_metrics['top1_acc']
                    # Save checkpoint
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        path=os.path.join(checkpoint_dir, f"step_{step}_class_{class_order[step]}.pth")
                    )
            except Exception as e:
                print(f"Error during evaluation: {e}")
                # Save model anyway to avoid losing progress
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    path=os.path.join(checkpoint_dir, f"step_{step}_class_{class_order[step]}_backup.pth")
                )
        
        # Reset best accuracy for next step
        best_acc = 0.0
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs,
        path=os.path.join(checkpoint_dir, "final_model.pth")
    )
    
    print("Incremental Learning completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--class_order', required=True,
                       help="Comma-separated class order")
    parser.add_argument('--base_model_path', required=True)
    parser.add_argument('--model_name', required=True, choices=['eaml', 'docformer'])
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default="distillation", 
                       choices=["standard", "distillation"])
    parser.add_argument('--temperature', type=float, default=2.0,
                       help="Temperature for knowledge distillation")
    parser.add_argument('--lambda_distill', type=float, default=1.0,
                       help="Weight for distillation loss")
    
    args = parser.parse_args()
    
    run_incremental_learning(
        data_root=args.data_dir,
        class_order=args.class_order.split(','),
        base_model_path=args.base_model_path,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        start_step=args.start_step,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        strategy=args.strategy,
        temperature=args.temperature,
        lambda_distill=args.lambda_distill
    )

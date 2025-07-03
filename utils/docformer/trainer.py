import torch
import torch.nn as nn
import logging
import os
import glob
import time
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import json
import math

class DocFormerTrainer:
    def __init__(self, model, config, train_loader, eval_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = torch.device(config.device)
        
        # Progressive training stages
        self.stages = ['text_pretrain', 'multimodal_pretrain', 'finetune']
        self.current_stage = config.training_stage
        self.stage_epochs = {
            'text_pretrain': config.text_stage_epochs,
            'multimodal_pretrain': config.visual_stage_epochs,
            'finetune': config.final_stage_epochs
        }
        
        # Initialize training components
        self._init_optimizer()
        self._init_scheduler()
        
        # Fixed GradScaler initialization - disable for problematic stages
        if config.use_amp and config.training_stage != "multimodal_pretrain":
            self.scaler = GradScaler(device='cuda', enabled=True)
            self.use_scaler = True
        else:
            self.scaler = None
            self.use_scaler = False
            
        # Training state
        self.global_step = 0
        self.best_metrics = {'loss': float('inf'), 'acc': 0.0}
        self.nan_counter = 0
        print(f"Trainer initialized for stage: {self.current_stage}")
        print(f"Mixed precision enabled: {self.use_scaler}")

        self.max_checkpoints_to_keep = 1

    def _init_optimizer(self):
        """Initialize optimizer with proper parameter grouping"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config.num_train_epochs
        self.warmup_steps = int(self.config.warmup_ratio * total_steps)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy='cos'
        )

    def _adjust_model_for_stage(self, stage):
        """Adjust model parameters based on training stage"""
        # Update scaler based on stage
        if stage == "multimodal_pretrain":
            # Disable mixed precision for problematic multimodal stage
            self.scaler = None
            self.use_scaler = False
            #print("Disabled mixed precision for multimodal pretraining")
        else:
            if self.config.use_amp:
                self.scaler = GradScaler(device='cuda', enabled=True)
                self.use_scaler = True
            
        if stage == "text_pretrain":
            for param in self.model.visual_backbone.parameters():
                param.requires_grad = False
            for param in self.model.text_embeddings.parameters():
                param.requires_grad = True
            if hasattr(self.model, 'classifier') and self.model.classifier:
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            print("Frozen visual backbone for text pretraining")
            
        elif stage == "multimodal_pretrain":
            for param in self.model.visual_backbone.parameters():
                param.requires_grad = True
            print("Unfrozen visual backbone for multimodal pretraining")
            
        elif stage == "finetune":
            for param in self.model.parameters():
                param.requires_grad = True
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.finetune_lr
            print("Unfrozen all components for fine-tuning")

    def _compute_loss(self, outputs, batch):
        #print(f"Stage: {self.current_stage}")
        #print(f"Outputs keys: {outputs.keys()}")
        #print(f"Labels shape: {batch['labels'].shape}")
        
        # Check if direct loss is available
        if 'loss' in outputs and outputs['loss'] is not None:
            loss = outputs['loss']
            if not loss.requires_grad:
                loss = loss.requires_grad_(True)
            return loss
        
        # Initialize loss tensor on correct device
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Text pretraining stage
        if self.current_stage == "text_pretrain":
            if 'logits' in outputs and outputs['logits'] is not None:
                cls_loss = nn.CrossEntropyLoss()(outputs['logits'], batch['labels'])
                return cls_loss
            else:
                print("Warning: No logits found in text pretraining")
                return torch.tensor(1e-8, device=device, requires_grad=True)
        
        # Multimodal pretraining stage
        elif self.current_stage == "multimodal_pretrain":
            loss_computed = False
            
            # MM-MLM Loss
            if ('mm_mlm_logits' in outputs and outputs['mm_mlm_logits'] is not None 
                and self.config.mm_mlm_weight > 0):
                try:
                    mm_mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                        outputs['mm_mlm_logits'].view(-1, outputs['mm_mlm_logits'].size(-1)),
                        batch['input_ids'].view(-1)
                    )
                    total_loss = total_loss + self.config.mm_mlm_weight * mm_mlm_loss
                    loss_computed = True
                except Exception as e:
                    print(f"MM-MLM loss computation failed: {e}")
            
            # LTR Loss
            if ('ltr_output' in outputs and outputs['ltr_output'] is not None 
                and self.config.ltr_weight > 0):
                try:
                    ltr_loss = nn.MSELoss()(outputs['ltr_output'], batch['pixel_values'])
                    total_loss = total_loss + self.config.ltr_weight * ltr_loss
                    loss_computed = True
                except Exception as e:
                    print(f"LTR loss computation failed: {e}")
            
            # TDI Loss
            if ('tdi_logits' in outputs and outputs['tdi_logits'] is not None 
                and self.config.tdi_weight > 0):
                try:
                    tdi_labels = torch.ones(batch['input_ids'].size(0), device=device)
                    tdi_loss = nn.BCEWithLogitsLoss()(outputs['tdi_logits'].squeeze(), tdi_labels)
                    total_loss = total_loss + self.config.tdi_weight * tdi_loss
                    loss_computed = True
                except Exception as e:
                    print(f"TDI loss computation failed: {e}")
            
            # If no loss was computed, return a small dummy loss
            if not loss_computed:
                print("Warning: No valid losses computed in multimodal pretraining")
                #return torch.tensor(1e-8, device=device, requires_grad=True)
            
            return total_loss
        
        # Fine-tuning stage
        elif self.current_stage == "finetune":
            if 'logits' in outputs and outputs['logits'] is not None:
                cls_loss = nn.CrossEntropyLoss()(outputs['logits'], batch['labels'])
                return cls_loss
        
        # Fallback: return small dummy loss instead of None
        print(f"Warning: Fallback loss for stage {self.current_stage}")
        return torch.tensor(1e-8, device=device, requires_grad=True)


    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [{self.current_stage}]")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                # Forward pass
                task = "pretrain" if self.current_stage in ["text_pretrain", "multimodal_pretrain"] else "finetune"
                outputs = self.model(**batch, task=task)
                loss = self._compute_loss(outputs, batch)
                
                # CRITICAL: Check if loss is None before tensor operations
                if loss is None:
                    print(f"Skipping batch {batch_idx}: loss is None")
                    continue
                    
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch {batch_idx}: invalid loss {loss.item()}")
                    continue
                    
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                valid_batches += 1
                self.global_step += 1
                
                # Update progress bar
                if self.scheduler and hasattr(self.scheduler, 'get_last_lr'):
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.config.learning_rate
                    
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                    'batch': f"{batch_idx+1}/{len(self.train_loader)}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
                
        return total_loss / valid_batches if valid_batches > 0 else float('inf')






    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.use_scaler:
                    with autocast(device_type='cuda', enabled=True):
                        task = "pretrain" if self.current_stage in ["text_pretrain", "multimodal_pretrain"] else "finetune"
                        outputs = self.model(**batch, task=task)
                        loss = self._compute_loss(outputs, batch)
                else:
                    task = "pretrain" if self.current_stage in ["text_pretrain", "multimodal_pretrain"] else "finetune"
                    outputs = self.model(**batch, task=task)
                    loss = self._compute_loss(outputs, batch)
                    
                total_loss += loss.item()
                
                if 'logits' in outputs and 'labels' in batch:
                    preds = outputs['logits'].argmax(dim=-1)
                    correct += (preds == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
                    
        metrics = {
            'loss': total_loss / len(self.eval_loader),
            'acc': correct / total * 100 if total > 0 else 0
        }
        return metrics

    def _manage_checkpoints(self, current_epoch, current_stage):
        """Keep only the last N checkpoints and the best model"""
        checkpoint_pattern = os.path.join(self.config.output_dir, f"checkpoint_{current_stage}_epoch*.pt")
        all_checkpoints = glob.glob(checkpoint_pattern)
        
        if len(all_checkpoints) <= self.max_checkpoints_to_keep:
            return
        
        # Sort checkpoints by epoch number
        checkpoint_info = []
        for ckpt_path in all_checkpoints:
            try:
                # Extract epoch number from filename
                filename = os.path.basename(ckpt_path)
                epoch_num = int(filename.split('epoch')[1].split('.pt')[0])
                checkpoint_info.append((epoch_num, ckpt_path))
            except (ValueError, IndexError):
                continue
        
        # Sort by epoch number (newest first)
        checkpoint_info.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only the last N checkpoints
        checkpoints_to_delete = checkpoint_info[self.max_checkpoints_to_keep:]
        
        for epoch_num, ckpt_path in checkpoints_to_delete:
            try:
                os.remove(ckpt_path)
                print(f"Deleted old checkpoint: {os.path.basename(ckpt_path)}")
            except OSError as e:
                print(f"Could not delete {ckpt_path}: {e}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with automatic cleanup"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config.__dict__,
            'training_stage': self.current_stage
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        else:
            checkpoint['scaler_state_dict'] = None
        
        # Save regular checkpoint
        filename = f"checkpoint_{self.current_stage}_epoch{epoch}.pt"
        filepath = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filename}")
        
        # Save best model (always keep this)
        if is_best:
            best_path = os.path.join(self.config.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss: {self.best_metrics['loss']:.4f}")
        
        # Clean up old checkpoints
        self._manage_checkpoints(epoch, self.current_stage)

    def progressive_train(self, output_dir):
        """Progressive training through all stages"""
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        
        all_results = {}
        
        for stage in self.stages:
            print(f"\n=== Starting {stage} training ===")
            self.current_stage = stage
            self.config.training_stage = stage
            self.config.scale_weights(stage)
            self._adjust_model_for_stage(stage)
            
            epochs = self.stage_epochs[stage]
            stage_best_loss = float('inf')
            
            for epoch in range(epochs):
                train_loss = self.train_epoch(epoch)
                eval_metrics = self.evaluate()
                
                if eval_metrics['loss'] < self.best_metrics['loss']:
                    self.best_metrics = eval_metrics.copy()
                    self.save_checkpoint(epoch, is_best=True)
                    stage_best_loss = eval_metrics['loss']
                    
                if (epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(epoch)
                
                print(
                    f"Stage: {stage} | Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {eval_metrics['loss']:.4f} | "
                    f"Val Acc: {eval_metrics['acc']:.2f}%"
                )
                    
            all_results[stage] = {
                'best_loss': stage_best_loss,
                'final_acc': eval_metrics['acc']
            }
            
            stage_checkpoint_path = os.path.join(output_dir, f"checkpoint_{stage}.pt")
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'training_stage': stage,
                'metrics': all_results[stage]
            }
            torch.save(checkpoint, stage_checkpoint_path)
                    
        total_time = time.time() - start_time
        print(f"\nProgressive training completed in {total_time/60:.2f} minutes")
        
        return {
            'best_val_acc': self.best_metrics['acc'],
            'best_val_loss': self.best_metrics['loss'],
            'stage_results': all_results,
            'total_time': total_time
        }

    def train(self, output_dir):
        """Standard training (single stage)"""
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        
        self._adjust_model_for_stage(self.current_stage)
        
        for epoch in range(self.config.num_train_epochs):
            train_loss = self.train_epoch(epoch)
            eval_metrics = self.evaluate()
            
            if eval_metrics['loss'] < self.best_metrics['loss']:
                self.best_metrics = eval_metrics.copy()
                self.save_checkpoint(epoch, is_best=True)
                
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            
            print(
                f"Epoch {epoch+1}/{self.config.num_train_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {eval_metrics['loss']:.4f} | "
                f"Val Acc: {eval_metrics['acc']:.2f}%"
            )
            
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        
        return {
            'best_val_acc': self.best_metrics['acc'],
            'best_val_loss': self.best_metrics['loss'],
            'total_time': total_time
        }

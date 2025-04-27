import os
import time
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DocFormerTrainer:
    def __init__(self, model, config, train_loader=None, eval_loader=None):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.checkpoint_files = []
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.mm_mlm_losses = []
        self.ltr_losses = []
        self.tdi_losses = []
        
        # Initialize components
        self._init_loss_functions()
        self._init_optimizer()
        self._init_scheduler()
        
        logger.info(f"Trainer initialized on device: {self.device}")

    def _init_loss_functions(self):
        """Initialize loss functions"""
        self.mm_mlm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.ltr_loss_fn = torch.nn.SmoothL1Loss()
        self.tdi_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.classification_loss_fn = torch.nn.CrossEntropyLoss()

    def _init_optimizer(self):
        """Initialize optimizer with weight decay"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        num_training_steps = self.config.num_train_epochs * len(self.train_loader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        task_losses = [0.0, 0.0, 0.0]  # MM-MLM, LTR, TDI
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Ensure all tensors are on correct device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                bboxes=batch['bboxes'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                task="pretrain"
            )
            
            # Calculate losses
            mm_mlm_loss = self.mm_mlm_loss_fn(
                outputs['mm_mlm_logits'].view(-1, self.model.text_embeddings.config.vocab_size),
                batch['input_ids'].view(-1)
            ) * self.config.mm_mlm_weight
            
            ltr_loss = self.ltr_loss_fn(
                outputs['ltr_output'],
                batch['pixel_values']
            ) * self.config.ltr_weight
            
            tdi_loss = self.tdi_loss_fn(
                outputs['tdi_logits'].squeeze(),
                torch.ones(batch['input_ids'].size(0), device=self.device)
            ) * self.config.tdi_weight
            
            total_loss = mm_mlm_loss + ltr_loss + tdi_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += total_loss.item()
            task_losses[0] += mm_mlm_loss.item()
            task_losses[1] += ltr_loss.item()
            task_losses[2] += tdi_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        self.mm_mlm_losses.append(task_losses[0] / len(self.train_loader))
        self.ltr_losses.append(task_losses[1] / len(self.train_loader))
        self.tdi_losses.append(task_losses[2] / len(self.train_loader))
        
        logger.info(
            f"Epoch {epoch+1} Train Loss: {avg_loss:.4f} | "
            f"MM-MLM: {self.mm_mlm_losses[-1]:.4f} | "
            f"LTR: {self.ltr_losses[-1]:.4f} | "
            f"TDI: {self.tdi_losses[-1]:.4f}"
        )
        
        return avg_loss

    def evaluate(self, loader=None) -> Tuple[float, float, Dict]:
        """Evaluate model performance"""
        loader = loader or self.eval_loader
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    bboxes=batch['bboxes'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels'],
                    task="classification"
                )
                
                total_loss += outputs['loss'].item()
                _, predicted = torch.max(outputs['logits'], 1)
                total_correct += (predicted == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * total_correct / total_samples
        
        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        
        logger.info(f"Validation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy, {}

    def save_metrics_plots(self, output_dir: str):
        """Save training metrics plots"""
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(self.val_accs)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        # Pre-training tasks plot
        plt.subplot(2, 2, 3)
        plt.plot(self.mm_mlm_losses, label='MM-MLM')
        plt.plot(self.ltr_losses, label='LTR')
        plt.plot(self.tdi_losses, label='TDI')
        plt.title('Pre-training Tasks Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved training metrics plots to {plot_path}")

    def train(self, output_dir: str) -> Dict:
        """Main training loop"""
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(self.config.num_train_epochs):
            epoch_start = time.time()
            
            # Train and evaluate
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc, _ = self.evaluate()
            
            # Update best metrics
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.save_checkpoint(output_dir, epoch, best=True)
            
            # Periodic checkpointing
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(output_dir, epoch)
            
            logger.info(
                f"Epoch {epoch+1} completed in {time.time()-epoch_start:.2f}s | "
                f"Best Val Loss: {self.best_val_loss:.4f}"
            )
        
        # Final evaluation and save
        final_val_loss, final_val_acc, _ = self.evaluate()
        self.save_checkpoint(output_dir, self.config.num_train_epochs - 1)
        self.save_metrics_plots(output_dir)
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'final_val_loss': final_val_loss,
            'final_val_acc': final_val_acc
        }

    def save_checkpoint(self, path: str, epoch: int, best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }
        
        if best:
            torch.save(checkpoint, os.path.join(path, "docformer_best.pt"))
            logger.info(f"Saved best model checkpoint at epoch {epoch+1}")
        else:
            torch.save(checkpoint, os.path.join(path, f"docformer_epoch_{epoch+1}.pt"))
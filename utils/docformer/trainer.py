import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

class DocFormerTrainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device)
        self.scaler = GradScaler(enabled=config.use_amp)
        self.criterion = nn.CrossEntropyLoss()
        self.global_step = 0

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-8)
        self.total_steps = len(train_loader) * config.num_train_epochs
        self.warmup_steps = int(0.1 * self.total_steps)

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(0.0, float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.best_val_loss = float('inf')
        self.early_stop_patience = config.early_stop_patience
        self.no_improve_epochs = 0
        self.model.to(self.device)
        self.config.output_dir = getattr(config, 'output_dir', 'outputs')
        os.makedirs(self.config.output_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}")
        for batch in progress_bar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.optimizer.zero_grad()
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(**batch, task="finetune")
                logits = outputs['logits']
                loss = self.criterion(logits, batch['labels'])
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total_loss += loss.item()
            desc = f"Epoch {epoch + 1} Loss: {loss.item():.4f}"
            progress_bar.set_description(desc)
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Eval"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(**batch, task="finetune")
                logits = outputs['logits']
                loss = self.criterion(logits, batch['labels'])
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {'loss': avg_loss, 'acc': accuracy * 100}

    def train(self, output_dir):
        self.model.to(self.device)
        best_model_path = None
        for epoch in range(self.config.num_train_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            val_loss = val_metrics['loss']
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['acc']:.2f}%")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                self.save_checkpoint(epoch, is_best=True)
                best_model_path = f"{output_dir}/best_model.pt"
            else:
                self.no_improve_epochs += 1
                if self.no_improve_epochs >= self.early_stop_patience:
                    print(f"Early stopping triggered after {self.early_stop_patience} epochs with no improvement.")
                    break
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
        return best_model_path

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config.__dict__,
        }
        filename = f"checkpoint_epoch{epoch+1}.pt"
        path = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.config.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch+1}")

import torch
from tqdm import tqdm
import torch.nn as nn

class DocFormerEvaluator:
    def __init__(self, model, config, test_loader, idx_to_class):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.idx_to_class = idx_to_class
        self.device = torch.device(config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.eval()

    def evaluate_model(self):
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test Eval"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(**batch, task="finetune")
                logits = outputs['logits']
                loss = self.criterion(logits, batch['labels'])
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

import torch
from torchmetrics import Accuracy
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    accur = Accuracy(task="multiclass", num_classes=model.classifier.out_features).to(device)
    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k,v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        logits = model(**inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        accur.update(preds, labels)

        loop.set_postfix(loss=loss.item(), accuracy=accur.compute().item())
    avg_loss = total_loss / len(dataloader)
    acc = accur.compute().item()
    return avg_loss, acc


def val_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    accur = Accuracy(task="multiclass", num_classes=model.classifier.out_features).to(device)
    loop = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in loop:
            inputs = {k: v.to(device) for k,v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            accur.update(preds, labels)

            loop.set_postfix(loss=loss.item(), accuracy=accur.compute().item())
    avg_loss = total_loss / len(dataloader)
    acc = accur.compute().item()
    return avg_loss, acc

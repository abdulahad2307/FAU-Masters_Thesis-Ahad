import torch
from torchmetrics import Accuracy

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    accur = Accuracy(num_classes=model.config.num_labels).to(device)
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k,v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        accur.update(preds, labels)
    avg_loss = total_loss / len(dataloader)
    acc = accur.compute().item()
    return avg_loss, acc

def val_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    accur = Accuracy(num_classes=model.config.num_labels).to(device)
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k,v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            accur.update(preds, labels)
    avg_loss = total_loss / len(dataloader)
    acc = accur.compute().item()
    return avg_loss, acc

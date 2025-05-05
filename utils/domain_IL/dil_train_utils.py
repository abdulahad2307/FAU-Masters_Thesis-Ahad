import torch
import os
import time
from tqdm import tqdm

def set_finetune_mode(model, mode="head_only", encoder_unfreeze_depth=1):
    if mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif mode == "partial_finetune":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        if hasattr(model, "encoder"):
            encoder_blocks = list(model.encoder.layer)
            for layer in encoder_blocks[-encoder_unfreeze_depth:]:
                for param in layer.parameters():
                    param.requires_grad = True
    elif mode == "full_finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

def save_checkpoint_dil(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict() if not isinstance(model, dict) else {
            k: v.state_dict() for k, v in model.items()
        },
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint_dil(model, optimizer, path, device):
    if not os.path.exists(path):
        print("No checkpoint found, training from scratch.")
        return 0

    checkpoint = torch.load(path, map_location=device)
    if isinstance(model, dict):
        for name, submodel in model.items():
            submodel.load_state_dict(checkpoint['model_state_dict'][name])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resuming from epoch {checkpoint['epoch']+1}")
    return checkpoint['epoch'] + 1

def train_one_epoch_dil(model, dataloader, optimizer, criterion, device):
    model.train() if not isinstance(model, dict) else [m.train() for m in model.values()]
    total_loss = 0
    total_correct = 0
    total_samples = 0

    start = time.time()
    loop = tqdm(dataloader, desc="DIL Training", leave=False)

    for batch in loop:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        if 'texts' in batch:
            input_ids = batch['texts']['input_ids'].to(device)
            attention_mask = batch['texts']['attention_mask'].to(device)
        else:
            input_ids, attention_mask = None, None

        optimizer.zero_grad()

        if isinstance(model, dict):  # post-ensemble
            outputs = sum([
                m(images, input_ids=input_ids, attention_mask=attention_mask)
                for m in model.values()
            ]) / len(model)
        else:
            outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    end = time.time()
    print(f"Epoch Training Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f} | Time: {end - start:.2f}s")
    return avg_loss, avg_acc

def evaluate_dil(model, dataloader, device):
    model.eval() if not isinstance(model, dict) else [m.eval() for m in model.values()]
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            if 'texts' in batch:
                input_ids = batch['texts']['input_ids'].to(device)
                attention_mask = batch['texts']['attention_mask'].to(device)
            else:
                input_ids, attention_mask = None, None

            if isinstance(model, dict):  # post-ensemble
                outputs = sum([
                    m(images, input_ids=input_ids, attention_mask=attention_mask)
                    for m in model.values()
                ]) / len(model)
            else:
                outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)

            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

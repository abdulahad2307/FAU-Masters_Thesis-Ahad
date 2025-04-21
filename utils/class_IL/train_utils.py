import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#Fine-tuning IL modes
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
        raise ValueError(f"Unknown finetune mode: {mode}")

#Optimizer with different LR for backbone/head
def create_optimizer(model, base_lr=1e-4, head_lr=1e-3):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    return torch.optim.Adam([
        {'params': backbone_params, 'lr': base_lr},
        {'params': head_params, 'lr': head_lr}
    ])

def save_checkpoint(model, optimizer, epoch, stage_id, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(path, f"stage_{stage_id}.pt"))

def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print("No checkpoint found. Starting fresh.")
        return 0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint: {path} | Resuming from epoch {checkpoint['epoch']+1}")
    return checkpoint['epoch'] + 1

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch in loop:
        optimizer.zero_grad()

        if "images" in batch:  # EAML
            images = batch["images"].to(device)
            input_ids = batch["texts"]["input_ids"].to(device)
            attention_mask = batch["texts"]["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(images, input_ids, attention_mask)

        else:  # DocFormer
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bboxes = batch["bboxes"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, bbox=bboxes)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)
    print(f"Train — Loss: {total_loss:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1

#Evaluation
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            if "images" in batch:
                images = batch["images"].to(device)
                input_ids = batch["texts"]["input_ids"].to(device)
                attention_mask = batch["texts"]["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)
            else:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bboxes = batch["bboxes"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, bbox=bboxes)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)
    print(f"Eval — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1

def train_model(model, train_loaders_by_stage, val_loader, device, args, stage_ids=None):
    """
    Trains a single model across IL stages
    """
    if stage_ids is None:
        stage_ids = range(len(train_loaders_by_stage))

    optimizer = create_optimizer(model, base_lr=args.base_lr, head_lr=args.head_lr)
    criterion = torch.nn.CrossEntropyLoss()

    for stage in stage_ids:
        print(f"Training Stage {stage + 1}/{len(train_loaders_by_stage)}")

        set_finetune_mode(model, args.finetune_mode, encoder_unfreeze_depth=args.encoder_unfreeze_depth)
        for epoch in range(args.epochs_per_stage):
            train_one_epoch(model, train_loaders_by_stage[stage], optimizer, criterion, device)
            evaluate(model, val_loader, device)

        save_checkpoint(model, optimizer, epoch, stage_id=stage, path=args.checkpoint_dir)

    return model


def ensemble_predict(models, dataloader, device, strategy="average"):
    """
    Generate ensemble predictions from multiple models.
    strategy: 'average' or 'vote'
    """
    for m in models:
        m.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            logits_list = []
            for model in models:
                if "images" in batch:
                    images = batch["images"].to(device)
                    input_ids = batch["texts"]["input_ids"].to(device)
                    attention_mask = batch["texts"]["attention_mask"].to(device)
                    output = model(images, input_ids=input_ids, attention_mask=attention_mask)
                else:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    bboxes = batch["bboxes"].to(device)
                    output = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, bbox=bboxes)
                logits_list.append(output)

            stacked = torch.stack(logits_list)
            if strategy == "average":
                avg_logits = stacked.mean(dim=0)
                preds = torch.argmax(avg_logits, dim=1)
            elif strategy == "vote":
                preds = torch.mode(torch.argmax(stacked, dim=2), dim=0).values
            else:
                raise ValueError("Unknown strategy")

            all_preds.extend(preds.cpu().tolist())
    return all_preds


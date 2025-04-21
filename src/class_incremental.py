import os
import time
import torch
import torch.nn as nn
from utils.class_IL.dataloader_utils import get_dataloaders
from utils.class_IL.train_utils import train_one_epoch, evaluate
from models import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}, starting fresh.")
        return 0
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}")
    return checkpoint['epoch']

def run_incremental_learning(
        data_root,
        class_order,
        base_model_path,
        model_name,
        checkpoint_dir,
        start_step=0,
        batch_size=8,
        lr=2e-5,
        num_epochs=10
    ):

    print("Starting Class Incremental Learning...")
    seen_classes = []

    for step in range(start_step, len(class_order)):
        current_classes = class_order[:step + 1]
        new_classes = [class_order[step]]
        print(f"Step {step+1}/{len(class_order)} | Training on new classes: {new_classes}")

        train_loader = get_dataloaders(data_root, "train", new_classes, batch_size=batch_size)
        val_loader = get_dataloaders(data_root, "val", current_classes, batch_size=batch_size)
        test_loader = get_dataloaders(data_root, "test", current_classes, batch_size=batch_size)

        model = load_model(model_name=model_name, num_classes=len(current_classes))
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pth")
        start_epoch = load_checkpoint(model, optimizer, ckpt_path)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, num_epochs):
            print(f"Training Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()

            train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_acc = evaluate(model, val_loader, DEVICE)

            epoch_time = time.time() - start_time
            print(f"Epoch Time: {epoch_time:.2f}s | Val Acc: {val_acc:.4f}")

            save_checkpoint(model, optimizer, epoch + 1, ckpt_path)

        print(f"Finished Training Step {step+1}. Now Evaluating on TEST Set...")
        test_acc = evaluate(model, test_loader, DEVICE)
        print(f"Step {step+1} Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    # Define order of classes for class-IL
    CLASS_ORDER = [
        'letter', 'form', 'email', 'handwritten', 'advertisement',
        'scientific report', 'invoice', 'presentation', 'questionnaire',
        'resume', 'memo', 'news article', 'budget', 'legal contract',
        'academic paper', 'menu'
    ]

    run_incremental_learning(
        data_root="/path/to/your/data",
        class_order=CLASS_ORDER,
        base_model_path="checkpoints/pretrained/EAML.pth",
        model_name="eaml",  # or "docformer"
        checkpoint_dir="checkpoints/class_IL/eaml",
        start_step=11,  # trained on 11, continue from 12th
        batch_size=8,
        lr=2e-5,
        num_epochs=5
    )

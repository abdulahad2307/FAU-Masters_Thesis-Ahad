import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import EAML_Dataset
from utils.eaml.eaml_model import EAMLModel
from tqdm import tqdm

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = {
        "input_ids": torch.stack([item["text"]["input_ids"].squeeze() for item in batch]),
        "attention_mask": torch.stack([item["text"]["attention_mask"].squeeze() for item in batch])
    }
    labels = torch.tensor([item["label"] for item in batch])
    return images, texts, labels

def train(model, dataloader, optimizer, criterion, device, epochs):
    model.to(device)
    model.ocr_model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, texts, labels in progress:
            images = images.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress.set_postfix({
                'loss': total_loss/(total/dataloader.batch_size),
                'acc': 100*correct/total
            })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Initialize dataset and model
    dataset = EAML_Dataset(os.path.join(args.data_dir, "train"))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    model = EAMLModel(num_classes=len(dataset.classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    train(model, dataloader, optimizer, criterion, args.device, args.epochs)

if __name__ == "__main__":
    main()
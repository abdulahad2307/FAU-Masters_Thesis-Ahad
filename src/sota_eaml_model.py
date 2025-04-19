import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import EAML_Dataset
from utils.eaml.eaml_model import EAMLModel
from tqdm import tqdm
import warnings

# Suppress the specific transformers warnings
warnings.filterwarnings("ignore", message="Config of the encoder.*")
warnings.filterwarnings("ignore", message="Config of the decoder.*")
warnings.filterwarnings("ignore", message="Some weights of.*")

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])  # images are first element
    texts = {
        "input_ids": torch.stack([item[1]["input_ids"].squeeze(0) for item in batch]),
        "attention_mask": torch.stack([item[1]["attention_mask"].squeeze(0) for item in batch])
    }
    labels = torch.tensor([item[2] for item in batch])  # labels are third element
    return images, texts, labels

def train(model, dataloader, optimizer, criterion, device, epochs):
    model.to(device)
    
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
    parser.add_argument("--device", default="cuda", help="Device to use (must be 'cuda')")
    args = parser.parse_args()

    # ==== Enforce GPU-only training ====
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA (GPU) is required but not available. Check your PyTorch installation and GPU drivers.")

    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

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
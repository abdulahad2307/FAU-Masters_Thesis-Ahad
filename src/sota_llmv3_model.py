import argparse
import os
import torch
from utils.llmv3.data_loader import get_dataloaders
from utils.llmv3.model_loader import load_model
from utils.llmv3.train_utils import train_epoch, val_epoch
from utils.llmv3.eval_utils import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="LayoutLMv3 Document Classification")
    parser.add_argument("--dataset", type=str, choices=["rvl_cdip", "tobacco3482", "docbank", "publaynet"], required=True)
    parser.add_argument("--ocr_tensor_dir", type=str, required=True, help="Directory with pre-extracted OCR tensors")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with document images")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        ocr_tensor_dir=args.ocr_tensor_dir,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    device = torch.device(args.device)
    model = load_model(num_labels=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, device)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f} / Val loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, f"layoutlmv3_{args.dataset}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training completed. Starting evaluation on validation set.")
    evaluate(model, val_loader, device)

if __name__ == "__main__":
    main()

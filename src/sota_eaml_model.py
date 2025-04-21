import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import EAML_DataLoader, load_class_list
import json
from utils.eaml.eaml_model import EAMLModel
from tqdm import tqdm

class EAMLTrainer:
    def __init__(self, model, device=None, learning_rate=1e-4):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU requested but not available. Check your GPU configuration.")
            
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            images = batch['images'].to(self.device)
            texts = {
                'input_ids': batch['texts']['input_ids'].to(self.device),
                'attention_mask': batch['texts']['attention_mask'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
                
            self.optimizer.zero_grad()
            outputs = self.model(images, texts)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress.set_postfix({
                'loss': total_loss/(total/dataloader.batch_size),
                'acc': 100*correct/total
            })
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                texts = {
                    'input_ids': batch['texts']['input_ids'].to(self.device),
                    'attention_mask': batch['texts']['attention_mask'].to(self.device)
                }
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(images, texts)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss
    
    def save_checkpoint(self, output_dir, epoch, best=False):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"eaml_cp_ep{epoch+1}.pt" if not best else "eaml_best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(output_dir, filename))

def main():
    parser = argparse.ArgumentParser(description="EAML for Document Classification")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--classes', nargs='+', default=[], 
                   help='Space-separated list of class names')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--resume', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], 
                       help="Force device selection (default: auto-detect)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    class_list = args.classes

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU requested but not available. Check your GPU configuration.")
    
    device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if class_list is None:
        raise ValueError("Class list not found. Please provide via --class_list or ensure classes.json exists in data directory")

    eaml_loader = EAML_DataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_list=class_list
    )
    
    train_loader = eaml_loader.get_loader('train')
    val_loader = eaml_loader.get_loader('val', shuffle=False)

    model = EAMLModel(num_classes=len(class_list))
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.resume}")

    trainer = EAMLTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate
    )

    if not args.eval_only:
        best_val_loss = float('inf')
        for epoch in range(args.num_epochs):
            train_loss = trainer.train_epoch(train_loader, epoch)
            val_loss = trainer.evaluate(val_loader)
            
            # Saving checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            trainer.save_checkpoint(args.output_dir, epoch, best=is_best)
    
    val_loss = trainer.evaluate(val_loader)
    print(f"Final Validation Loss: {val_loss:.4f}")

    with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
        json.dump(class_list, f)

if __name__ == "__main__":

    if torch.cuda.is_available():
        print(f"CUDA is available with {torch.cuda.device_count()} GPU(s)")
        print(f"First GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available - falling back to CPU")

    main()
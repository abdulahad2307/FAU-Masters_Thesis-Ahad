import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models

from utils.dataloader import DataLoader

OUTPUT_DIR = "/home/woody/iwi5/iwi5280h/saved_models"

class BaselineModel:
    def __init__(self, model_name, num_classes, optimizer_name, learning_rate, device="cuda"):
        """
        Initializes the baseline model.

        Parameters:
        model_name: Name of the model ('resnet50' or 'densenet121').
        num_classes: Number of output classes.
        learning_rate: Learning rate for optimizer.
        device: Device ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        ## Loading models
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        else:
            raise ValueError("Unsupported model. Choose 'resnet50' or 'densenet121'.")

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_name = optimizer_name
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif self.optimizer_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer. Choose 'adam', 'sgd', or 'adamw'.")

        # Learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

    def train(self, train_loader, val_loader, epochs=10):
        """
        Train the model.

        Parameters:
        train_loader: loads the training data
        val_loader: loads the validation data
        epochs: number of epochs
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

            # Validation
            val_accuracy= self.validate(val_loader)
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model_save_path = os.path.join(OUTPUT_DIR, f"{self.model_name}_{self.optimizer}.pth")
        print(model_save_path)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")


    def validate(self, data_loader):
        """Validate and Test the model.
        
        Parameters:
        data_loader: loads the data
        """
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def test(self, test_loader):
        """
        Testing the model using the same validation function
        """
        test_accuracy= self.validate(test_loader)
        return test_accuracy
    
    def save_training_log(model_name, optimizer_name, epochs, batch_size, num_classes, learning_rate, test_accuracy):
        """
        Saving thetraining details to a CSV log file.
        """
        log_data = {
            "Model": model_name,
            "Optimizer": optimizer_name,
            "Epochs": epochs,
            "Batch Size": batch_size,
            "Num Classes": num_classes,
            "Learning Rate": learning_rate,
            "Test Accuracy (%)": f"{test_accuracy:.2f}",
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        log_file = "logs/training_log.csv"
        os.makedirs("logs", exist_ok=True)

        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = df.append(log_data, ignore_index=True)
        else:
            df = pd.DataFrame([log_data])

        df.to_csv(log_file, index=False)
        print(f"Training log saved at {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Model for Document Classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to organized dataset directory")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "densenet121"], default="resnet50",
                        help="Model to use (resnet50 or densenet121)")
    parser.add_argument("--num_classes", type=int, default=16, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd", "adamw"], default="adam", help="Optimizer to use")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to use")

    args = parser.parse_args()

    ## Loading the dataset
    data_loader = DataLoader(data_dir=args.data_dir, batch_size=args.batch_size, dataset_type="all")
    loaders = data_loader.load_data()

    ## Initializing and training model
    model = BaselineModel(model_name=args.model_name, num_classes=args.num_classes, optimizer_name=args.optimizer, learning_rate=args.learning_rate, device=args.device)
    model.train(loaders["train"], loaders["val"], epochs=args.epochs)
    test_accuracy = model.test(loaders["test"])
    print(f"Test Accuracy: {test_accuracy:.2f}%")


    save_training_log(args.model_name, args.optimizer, args.epochs, args.batch_size, args.num_classes, 
                      args.learning_rate, test_accuracy)

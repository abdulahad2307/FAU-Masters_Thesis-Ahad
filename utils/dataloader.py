import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.datasets import ImageFolder

class DataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=224, dataset_type="all", classes=None):
        """
        Initializing the dataset loader.

        Parameters:
        data_dir: Path to the organized dataset directory.
        batch_size: Number of samples per batch.
        num_workers: Number of worker threads for data loading.
        img_size: Image size (for resizing).
        dataset_type: One of ['train', 'val', 'test', 'all'].
        classes: List of specific classes to load (default: all classes).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.classes = classes

        # Defining image transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load datasets
        self.train_dataset = self.load_dataset("train")
        self.val_dataset = self.load_dataset("val")
        self.test_dataset = self.load_dataset("test")

    def load_dataset(self, dataset_type):
        """
        Loads a dataset based on the type ('train', 'val', 'test').
        """
        dataset_path = os.path.join(self.data_dir, dataset_type)

        if not os.path.exists(dataset_path):
            raise ValueError(f"{dataset_type} directory does not exist: {dataset_path}")

        # Apply transformations and load dataset using ImageFolder
        dataset = ImageFolder(root=dataset_path, transform=self.transform)

        # If specific classes are provided, filter the dataset
        if self.classes is not None:
            class_indices = [dataset.class_to_idx[class_name] for class_name in self.classes if class_name in dataset.class_to_idx]
            dataset.samples = [sample for sample in dataset.samples if sample[1] in class_indices]
            dataset.targets = [target for target in dataset.targets if target in class_indices]

        return dataset

    def get_data_loader(self, dataset_type):
        """
        This function returns a DataLoader object for the specified dataset type (train, val, test).
        """
        if dataset_type == "train":
            dataset = self.train_dataset
        elif dataset_type == "val":
            dataset = self.val_dataset
        elif dataset_type == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        # Ensures correct DataLoader initialization with shuffle only for training
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=(dataset_type == "train"), num_workers=self.num_workers)

    def load_data(self):
        """
        Loads train, val, test, or all data based on dataset_type.
        """
        loaders = {}

        if self.dataset_type in ["train", "all"]:
            loaders["train"] = self.get_data_loader("train")
        if self.dataset_type in ["val", "all"]:
            loaders["val"] = self.get_data_loader("val")
        if self.dataset_type in ["test", "all"]:
            loaders["test"] = self.get_data_loader("test")

        return loaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom DataLoader for Document Classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to organized dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for resizing")
    parser.add_argument("--dataset_type", type=str, choices=["train", "val", "test", "all"], default="all",
                        help="Dataset split to load (train, val, test, or all)")
    parser.add_argument("--classes", nargs="+", default=None, help="List of class names to load (default: all classes)")

    args = parser.parse_args()

    loader = DataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        dataset_type=args.dataset_type,
        classes=args.classes
    )

    data_loaders = loader.load_data()
    print(f"Data Loaders initialized for: {list(data_loaders.keys())}")

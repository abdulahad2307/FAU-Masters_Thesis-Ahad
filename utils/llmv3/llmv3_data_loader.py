import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from pathlib import Path
import torch.nn.functional as F

class DocumentDataset(Dataset):
    def __init__(self, image_dir, ocr_tensor_dir, classes, max_length=512, transform=None):
        self.image_dir = Path(image_dir)
        self.ocr_tensor_dir = Path(ocr_tensor_dir)
        self.classes = classes
        self.max_length = max_length
        self.transform = transform
        self.samples = []
        for c in classes:
            img_paths = list((self.image_dir / c).glob("*.*"))
            for img_path in img_paths:
                tensor_path = self.ocr_tensor_dir / c / (img_path.stem + ".pt")
                if tensor_path.exists():
                    self.samples.append((img_path, tensor_path, c))
        self.class2idx = {c:i for i,c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tensor_path, cls = self.samples[idx]
        image = read_image(str(img_path)).float() / 255.0
        if self.transform:
            image = self.transform(image)
        ocr_tensor = torch.load(tensor_path)
        for k in ["input_ids", "bbox", "attention_mask"]:
            t = ocr_tensor[k]
            if t.size(0) > self.max_length:
                ocr_tensor[k] = t[:self.max_length]
            elif t.size(0) < self.max_length:
                pad_len = self.max_length - t.size(0)
                pad_shape = [pad_len] + list(t.shape[1:])
                pad_tensor = torch.zeros(pad_shape, dtype=t.dtype)
                ocr_tensor[k] = torch.cat((t, pad_tensor), dim=0)

        return {
            "pixel_values": image,
            "input_ids": ocr_tensor["input_ids"],
            "bbox": ocr_tensor["bbox"],
            "attention_mask": ocr_tensor["attention_mask"],
            "labels": torch.tensor(self.class2idx[cls], dtype=torch.long)
        }

def get_dataloaders(dataset_name, ocr_tensor_dir, image_dir, batch_size=8, max_length=512):
    if dataset_name == "rvl_cdip":
        classes = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report', 'scientific_publication', 'specification', 'questionnaire', 'resume', 'memo', 'invoice', 'news_article', 'budget', 'presentation', 'budget']
    elif dataset_name == "tobacco3482":
        classes = ['advertisement', 'email', 'form', 'letter','memo','news_article','Note','Report','resume','scientific_report'] 
    elif dataset_name == "docbank":
        classes = ['abstract', 'caption', 'equation', 'figure', 'list', 'paragraph', 'reference', 'table', 'title']
    elif dataset_name == "publaynet":
        classes = ['text', 'title', 'list', 'table', 'figure']
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    dataset = DocumentDataset(image_dir, ocr_tensor_dir, classes, max_length)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, len(classes)

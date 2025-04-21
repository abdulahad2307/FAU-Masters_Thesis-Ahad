import os
from typing import List, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from transformers import BertTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

# ---- Common Transform ----
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ========================== EAML Dataset for Class IL ==========================
class EAMLClassILDataset(Dataset):
    def __init__(self, data_dir: str, current_classes: List[str], transform=None):
        self.data_dir = data_dir
        self.current_classes = current_classes
        self.transform = transform

        self.samples = []
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self._ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self._ocr_model.eval()
        for p in self._ocr_model.parameters():
            p.requires_grad = False

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.current_classes)}
        self._load_samples()

    def _extract_text(self, image: Image.Image) -> str:
        with torch.no_grad():
            pixel_values = self._ocr_processor(image, return_tensors="pt").pixel_values
            generated_ids = self._ocr_model.generate(pixel_values)
            return self._ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _load_samples(self):
        for class_name in os.listdir(self.data_dir):
            if class_name not in self.current_classes:
                continue
            class_dir = os.path.join(self.data_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    img_path = os.path.join(class_dir, fname)
                    try:
                        image = Image.open(img_path).convert("RGB")
                        text = self._extract_text(image)
                        tokens = self.tokenizer(
                            text, padding="max_length", truncation=True,
                            max_length=128, return_tensors="pt"
                        )
                        self.samples.append((img_path, tokens, class_name))
                    except Exception as e:
                        print(f"Skipping {img_path} due to: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, tokens, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "text": {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0)
            },
            "label": torch.tensor(self.class_to_idx[label])
        }


def eaml_collate_fn(batch):
    return {
        "images": torch.stack([x["image"] for x in batch]),
        "texts": {
            "input_ids": torch.stack([x["text"]["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["text"]["attention_mask"] for x in batch])
        },
        "labels": torch.stack([x["label"] for x in batch])
    }


# ========================== DocFormer Dataset for Class IL ==========================
class DocFormerClassILDataset(Dataset):
    def __init__(self, data_dir: str, current_classes: List[str], tokenizer_name="bert-base-uncased", max_seq_length=512):
        self.data_dir = data_dir
        self.current_classes = current_classes
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.samples = []

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.current_classes)}
        self._load_samples()

    def _load_samples(self):
        for class_name in os.listdir(self.data_dir):
            if class_name not in self.current_classes:
                continue
            class_dir = os.path.join(self.data_dir, class_name)
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.samples.append({
                        'image_path': os.path.join(class_dir, file),
                        'class_name': class_name
                    })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        pixel_values = common_transform(image)
        input_ids = torch.tensor([self.tokenizer.cls_token_id] +
                                 [self.tokenizer.pad_token_id] * (self.max_seq_length - 2) +
                                 [self.tokenizer.sep_token_id])[:self.max_seq_length]
        attention_mask = torch.tensor([1] + [0] * (self.max_seq_length - 2) + [1])[:self.max_seq_length]
        bboxes = torch.tensor([[0, 0, image.width, 0, image.width, image.height, 0, image.height]] * self.max_seq_length)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bboxes": bboxes,
            "label": torch.tensor(self.class_to_idx[sample["class_name"]])
        }


def docformer_collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "bboxes": torch.stack([x["bboxes"] for x in batch]),
        "labels": torch.stack([x["label"] for x in batch])
    }

# ========================== Wrapper Loader ==========================
def get_class_il_loader(model_type: str, data_dir: str, current_classes: List[str], batch_size=32, num_workers=4, split="train"):
    path = os.path.join(data_dir, split)
    if model_type == "eaml":
        dataset = EAMLClassILDataset(path, current_classes, transform=common_transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=eaml_collate_fn)
    elif model_type == "docformer":
        dataset = DocFormerClassILDataset(path, current_classes)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=docformer_collate_fn)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

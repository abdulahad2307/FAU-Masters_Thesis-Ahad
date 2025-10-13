import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

def poly8_to_bbox4(poly):
    xs = poly[0::2]
    ys = poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]

class OCRTensorsDataset(Dataset):
    def __init__(self, image_dir, ocr_tensor_file, classes, max_length=512, bbox_style="rect"):
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.class2idx = {c: i for i, c in enumerate(classes)}
        self.max_length = max_length
        self.bbox_style = bbox_style
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        self.ocr_data = torch.load(ocr_tensor_file)
        self.ocr_map = {}
        for entry in self.ocr_data:
            im_path = Path(entry.get("image_path", ""))
            self.ocr_map[str(im_path).lower()] = entry

        self.samples = []
        for c in classes:
            class_dir = self.image_dir / c
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.*"):
                key = str(img_path).lower()
                if key in self.ocr_map:
                    self.samples.append((img_path, self.ocr_map[key], c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ocr_dict, cls = self.samples[idx]
        pil_image = Image.open(str(img_path)).convert("RGB")
        image = self.transform(pil_image)

        input_ids = ocr_dict["input_ids"]
        input_ids = input_ids.clamp(min=0, max=self.max_length - 1)
        attention_mask = ocr_dict["attention_mask"]

        if self.bbox_style == "rect":
            if "bbox" in ocr_dict:
                bbox = ocr_dict["bbox"]
            elif "bboxes" in ocr_dict:
                bbox = torch.stack([torch.tensor(poly8_to_bbox4(poly.tolist()), dtype=torch.float32)
                                   for poly in ocr_dict["bboxes"]])
            else:
                raise ValueError(f"No bbox or bboxes in OCR dict for {img_path}")
        elif self.bbox_style == "poly":
            if "bboxes" in ocr_dict:
                bbox = ocr_dict["bboxes"].float()
            elif "bbox" in ocr_dict:
                N = ocr_dict["bbox"].shape[0]
                bbox = torch.zeros((N, 8), dtype=torch.float32)
                rects = ocr_dict["bbox"]
                for i in range(N):
                    x0, y0, x1, y1 = rects[i]
                    bbox[i] = torch.tensor([x0, y0, x1, y0, x1, y1, x0, y1])
            else:
                raise ValueError(f"No bbox or bboxes in OCR dict for {img_path}")
        else:
            raise ValueError(f"Unsupported bbox_style: {self.bbox_style}")

        input_ids = self._pad_truncate(input_ids, self.max_length)
        attention_mask = self._pad_truncate(attention_mask, self.max_length)
        bbox = self._pad_truncate(bbox, self.max_length)

        label = torch.tensor(self.class2idx[cls], dtype=torch.long)

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "labels": label
        }

    def _pad_truncate(self, tensor, max_len):
        N = tensor.shape[0]
        if N > max_len:
            return tensor[:max_len]
        elif N < max_len:
            pad_shape = (max_len - N,) + tensor.shape[1:]
            pad = torch.zeros(pad_shape, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0)
        else:
            return tensor

def get_dataloaders(dataset_name, ocr_tensor_file, image_dir, batch_size=8, max_length=512, bbox_style="rect"):
    if dataset_name == "rvl_cdip":
        classes = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report',
                   'scientific_publication', 'specification', 'questionnaire', 'resume',
                   'memo', 'invoice', 'news_article', 'budget', 'presentation', 'advertisement']
    elif dataset_name == "tobacco3482":
        classes = ['ad', 'letter', 'form', 'memo', 'report', 'resume', 'scientific', 'specification', 'news']
    elif dataset_name == "docbank":
        classes = ['abstract', 'caption', 'equation', 'figure', 'list', 'paragraph', 'reference', 'table', 'title']
    elif dataset_name == "publaynet":
        classes = ['text', 'title', 'list', 'table', 'figure']
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    dataset = OCRTensorsDataset(image_dir, ocr_tensor_file, classes, max_length, bbox_style)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, len(classes)

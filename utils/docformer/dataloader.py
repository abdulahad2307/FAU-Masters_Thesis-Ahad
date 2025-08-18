import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class OCRDataset(Dataset):
    """
    OCRDataset loads images and pre-extracted OCR token tensors (.pt files)
    required for DocFormer text and bounding box inputs.
    """

    def __init__(self, root_dir, split, ocr_token_dir, config):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.ocr_token_dir = ocr_token_dir
        self.config = config

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_samples()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_samples(self):
        # Load image paths and labels; assumes directory structure root_dir/split/class_name/images
        classes = [d for d in sorted(os.listdir(self.root_dir)) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and process image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load OCR token file
        ocr_name = os.path.basename(img_path) + '.pt'
        ocr_path = os.path.join(self.ocr_token_dir, ocr_name)

        if os.path.isfile(ocr_path):
            ocr_data = torch.load(ocr_path)
            input_ids = ocr_data.get('input_ids', torch.zeros(self.config.max_position_embeddings, dtype=torch.long))
            attention_mask = ocr_data.get('attention_mask', torch.zeros(self.config.max_position_embeddings, dtype=torch.long))
            bboxes = ocr_data.get('bboxes', torch.zeros(self.config.max_position_embeddings, 8, dtype=torch.float32))
        else:
            # If no token file, fallback to zero tensors with CLS and SEP token ids set
            input_ids = torch.zeros(self.config.max_position_embeddings, dtype=torch.long)
            attention_mask = torch.zeros(self.config.max_position_embeddings, dtype=torch.long)
            bboxes = torch.zeros((self.config.max_position_embeddings, 8), dtype=torch.float32)
            input_ids[0] = 101  # [CLS]
            input_ids[1] = 102  # [SEP]
            attention_mask[0] = 1
            attention_mask[1] = 1

        # Prepare spatial features for bounding boxes (x_feats, y_feats) and absolute position ids
        x_feats = torch.zeros((self.config.max_position_embeddings, 8), dtype=torch.long)
        y_feats = torch.zeros((self.config.max_position_embeddings, 8), dtype=torch.long)
        abs_pos_ids = torch.arange(self.config.max_position_embeddings, dtype=torch.long)

        # Compute spatial features from bboxes where possible
        max_2d = self.config.max_position_embeddings - 1

        for i in range(min(bboxes.shape[0], self.config.max_position_embeddings)):
            bbox = bboxes[i]
            # Basic clamp and coordinate indexing for sample features
            x1 = int(bbox[0].item() * max_2d)
            x3 = int(bbox[4].item() * max_2d)  # 5th coord actually? Adjust if needed
            w = int(abs(bbox[2].item() - bbox[0].item()) * max_2d)
            y1 = int(bbox[1].item() * max_2d)
            y3 = int(bbox[5].item() * max_2d)
            h = int(abs(bbox[7].item() - bbox[1].item()) * max_2d)

            x_feats[i, 0] = x1
            x_feats[i, 1] = x3
            x_feats[i, 2] = w
            y_feats[i, 0] = y1
            y_feats[i, 1] = y3
            y_feats[i, 2] = h

            # TODO: Add relative positions if desired

        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bboxes': bboxes,
            'x_feats': x_feats,
            'y_feats': y_feats,
            'abs_pos_ids': abs_pos_ids,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'img_path': img_path,
        }


def collate_fn(batch):
    # Batch collate for dataloader: stacks and concatenates tensors appropriately
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'bboxes': torch.stack([item['bboxes'] for item in batch]),
        'x_feats': torch.stack([item['x_feats'] for item in batch]),
        'y_feats': torch.stack([item['y_feats'] for item in batch]),
        'abs_pos_ids': torch.stack([item['abs_pos_ids'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'img_path': [item['img_path'] for item in batch],
    }

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


class OCRData(Dataset):
    def __init__(self, config, data_dir: str, split: str = "train", ocr_token_dir: str = None, classes: list = None):
        self.config = config
        self.data_dir = os.path.join(data_dir, split)
        self.ocr_token_dir = ocr_token_dir
        self.classes = classes
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load samples
        self.samples = self._load_samples()
        self.class_names = [sample[1] for sample in self.samples]

        # Filter classes if specified
        if self.classes:
            classes_lower = [c.lower() for c in self.classes]
            filtered_samples = []
            filtered_class_names = []
            
            for sample in self.samples:
                if sample[1].lower() in classes_lower:
                    filtered_samples.append(sample)
                    filtered_class_names.append(sample[1])
            
            self.samples = filtered_samples
            self.class_names = filtered_class_names

            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
            self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        print(f"Loaded {len(self.samples)} samples for split '{split}'")

    def _load_samples(self):
        """Load samples from directory structure"""
        samples = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return samples
            
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        samples.append((img_path, class_dir.lower()))
        return samples

    def create_class_mappings(self):
        """Create class to index mappings"""
        unique_classes = sorted(set(self.class_names))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        print(f"Created mappings for {len(unique_classes)} classes: {list(unique_classes)}")

    def filter_classes(self, target_classes):
        """Filter samples to only include specified classes"""
        if not target_classes:
            return
            
        target_classes_set = set([cls.lower() for cls in target_classes])
        filtered_samples = []
        filtered_class_names = []
        
        for sample in self.samples:
            if sample[1] in target_classes_set:
                filtered_samples.append(sample)
                filtered_class_names.append(sample[1])
        
        self.samples = filtered_samples
        self.class_names = filtered_class_names
        print(f"Filtered to {len(self.samples)} samples from {len(target_classes)} classes")

    def __len__(self):
        return len(self.samples)

    def to_pyfloat(self,x):
        if hasattr(x, 'item'):
            return x.item()
        return x
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            pixel_values = self.transform(img)

            # Load pre-extracted OCR tokens
            ocr_filename = os.path.basename(img_path) + '.pt'
            ocr_token_path = os.path.join(self.ocr_token_dir, ocr_filename) if self.ocr_token_dir else None

            if ocr_token_path and os.path.exists(ocr_token_path):
                ocr_data = torch.load(ocr_token_path, map_location='cpu')
                input_ids = ocr_data.get('input_ids', torch.zeros(self.config.max_position_embeddings, dtype=torch.long))
                attention_mask = ocr_data.get('attention_mask', torch.zeros(self.config.max_position_embeddings, dtype=torch.long))
                bboxes = ocr_data.get('bboxes', None)
            else:
                # Fallback: empty tokens and boxes
                input_ids = torch.zeros(self.config.max_position_embeddings, dtype=torch.long)
                attention_mask = torch.zeros(self.config.max_position_embeddings, dtype=torch.long)
                input_ids[0] = self.tokenizer.cls_token_id
                input_ids[1] = self.tokenizer.sep_token_id
                attention_mask[0] = 1
                attention_mask[1] = 1
                bboxes = None

            max_seq_len = self.config.max_position_embeddings
            if bboxes is None:
                bboxes = torch.zeros((max_seq_len, 8), dtype=torch.float32)
                cls_bbox = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
                bboxes[0] = cls_bbox
                bboxes[1] = cls_bbox

            x_feats = torch.zeros((max_seq_len, 8), dtype=torch.long)
            y_feats = torch.zeros((max_seq_len, 8), dtype=torch.long)
            abs_pos_ids = torch.arange(max_seq_len, dtype=torch.long)

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                coords = [self.to_pyfloat(c) for c in bbox]
                x1, y1, x2, y2, x3, y3, x4, y4 = coords

                max_2d = self.config.max_position_embeddings
                x1_idx = int(round(x1 * (max_2d - 1)))
                x3_idx = int(round(x3 * (max_2d - 1)))
                w = int(round(abs(x2 - x1) * (max_2d - 1)))
                h = int(round(abs(y4 - y1) * (max_2d - 1)))

                # For relative positions, similarly:
                if i > 0:
                    prev_bbox = bboxes[i - 1]
                    prev_coords = [self.to_pyfloat(c) for c in prev_bbox]
                    Ax_rel = int(round(x1 - prev_coords[0]))
                    Ay_rel = int(round(y1 - prev_coords[1]))
                    x2_x1 = int(round((x2 - x1) * (max_2d - 1)))
                    x4_x1 = int(round((x4 - x1) * (max_2d - 1)))
                    y2_y1 = int(round((y2 - y1) * (max_2d - 1)))
                    y4_y1 = int(round((y4 - y1) * (max_2d - 1)))
                    x_c = (x1 + x2 + x3 + x4) / 4
                    y_c = (y1 + y2 + y3 + y4) / 4
                    prev_x_c = (prev_coords[0] + prev_coords[2] + prev_coords[4] + prev_coords[6]) / 4
                    prev_y_c = (prev_coords[1] + prev_coords[3] + prev_coords[5] + prev_coords[7]) / 4
                    xc_rel = int(round((x_c - prev_x_c) * (max_2d - 1)))
                    yc_rel = int(round((y_c - prev_y_c) * (max_2d - 1)))
                else:
                    Ax_rel = Ay_rel = x2_x1 = x4_x1 = y2_y1 = y4_y1 = xc_rel = yc_rel = 0

                Ax_rel = max(-max_2d, min(Ax_rel, max_2d))
                Ay_rel = max(-max_2d, min(Ay_rel, max_2d))
                xc_rel = max(-max_2d, min(xc_rel, max_2d))
                yc_rel = max(-max_2d, min(yc_rel, max_2d))

                x_feats[i, 0] = x1_idx
                x_feats[i, 1] = x3_idx
                x_feats[i, 2] = w
                x_feats[i, 3] = Ax_rel + max_2d
                x_feats[i, 4] = x2_x1 + max_2d
                x_feats[i, 5] = x4_x1 + max_2d
                x_feats[i, 6] = xc_rel + max_2d
                x_feats[i, 7] = xc_rel + max_2d

                y_feats[i, 0] = int(round(y1 * (max_2d - 1)))
                y_feats[i, 1] = int(round(y3 * (max_2d - 1)))
                y_feats[i, 2] = h
                y_feats[i, 3] = Ay_rel + max_2d
                y_feats[i, 4] = y2_y1 + max_2d
                y_feats[i, 5] = y4_y1 + max_2d
                y_feats[i, 6] = yc_rel + max_2d
                y_feats[i, 7] = yc_rel + max_2d

            sample =  {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'bboxes': bboxes,
                'x_feats': x_feats,
                'y_feats': y_feats,
                'abs_pos_ids': abs_pos_ids,
                'labels': torch.tensor([self.class_to_idx[label]], dtype=torch.long),
                'img_path': img_path
            }
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    assert v.numel() > 0, f"{k} is empty for {img_path}"
                    print(f"[DEBUG] {k} shape: {v.shape}")
            return sample

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        """Return empty sample for error cases"""
        max_seq_len = self.config.max_position_embeddings
        cls_bbox = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        
        # Create bboxes tensor
        bboxes = torch.zeros((max_seq_len, 8), dtype=torch.float32)
        bboxes[0] = cls_bbox  # [CLS] token bbox
        bboxes[1] = cls_bbox  # [SEP] token bbox
        
        # Create attention mask for [CLS] and [SEP] only
        attention_mask = torch.zeros(max_seq_len, dtype=torch.long)
        attention_mask[0] = 1  # [CLS]
        attention_mask[1] = 1  # [SEP]
        
        # Create input_ids for [CLS] and [SEP]
        input_ids = torch.zeros(max_seq_len, dtype=torch.long)
        input_ids[0] = self.tokenizer.cls_token_id
        input_ids[1] = self.tokenizer.sep_token_id

        # Create spatial features
        x_feats = torch.zeros((max_seq_len, 8), dtype=torch.long)
        y_feats = torch.zeros((max_seq_len, 8), dtype=torch.long)
        abs_pos_ids = torch.arange(max_seq_len, dtype=torch.long)
        
        return {
            'pixel_values': torch.zeros((3, 224, 224), dtype=torch.float32),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bboxes': bboxes,
            'x_feats': x_feats,
            'y_feats': y_feats,
            'abs_pos_ids': abs_pos_ids,
            'labels': torch.tensor(0, dtype=torch.long),
            'img_path': ''
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bboxes': torch.stack([x['bboxes'] for x in batch]),
        'x_feats': torch.stack([x['x_feats'] for x in batch]),
        'y_feats': torch.stack([x['y_feats'] for x in batch]),
        'abs_pos_ids': torch.stack([x['abs_pos_ids'] for x in batch]),
        'labels': torch.cat([x['labels'] for x in batch]),
    }

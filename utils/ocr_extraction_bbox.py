import os
import time
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import BertTokenizer

def normalize_bbox(bbox, img_width, img_height):
    """Normalize bounding box coordinates to relative [0, 1] range"""
    normalized = []
    for i in range(0, len(bbox), 2):
        x = bbox[i] / img_width
        y = bbox[i+1] / img_height
        normalized.extend([x, y])
    return normalized

def get_ocr_results_tesseract(image):
    import pytesseract
    from pytesseract import Output

    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    texts = []
    boxes = []
    img_width, img_height = image.size
    
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        text = d['text'][i].strip()
        if text == '':
            continue
        try:
            conf = int(d['conf'][i])
        except:
            conf = 0
        if conf < 50:
            continue
        
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        # Convert to 8-point polygon (x1,y1,x2,y1,x2,y2,x1,y2)
        bbox_8 = [x, y, x + w, y, x + w, y + h, x, y + h]
        bbox_norm = normalize_bbox(bbox_8, img_width, img_height)
        
        texts.append(text)
        boxes.append(bbox_norm)
    
    return texts, boxes

def get_ocr_results_easyocr(image):
    import easyocr
    if easyocr is None:
        raise ImportError("easyocr is not installed")
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(np.array(image))
    
    texts = []
    boxes = []
    img_width, img_height = image.size
    
    for bbox, text, conf in results:
        if conf < 0.5:
            continue
        # bbox is list of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        bbox_8 = [coord for point in bbox for coord in point]
        bbox_norm = normalize_bbox(bbox_8, img_width, img_height)
        texts.append(text)
        boxes.append(bbox_norm)
    return texts, boxes

def get_ocr_results_pero(image):
    try:
        from pero_ocr.document_ocr import DocumentOCR
        doc_ocr = DocumentOCR()
        result = doc_ocr.run_ocr(np.array(image)[:, :, ::-1])  # RGB to BGR
        texts = []
        boxes = []
        for page in result.pages:
            for region in page.regions:
                for line in region.lines:
                    texts.append(line.transcription)
                    polygon = line.polygon  # list of 4 points (x, y)
                    bbox_8 = []
                    for point in polygon:
                        bbox_8.extend([point[0], point[1]])
                    bbox_norm = normalize_bbox(bbox_8, image.width, image.height)
                    boxes.append(bbox_norm)
        return texts, boxes
    except Exception as e:
        print(f"Pero OCR extraction failed: {e}")
        return [], []

def tokenize_and_save(texts, boxes, tokenizer, max_len, output_path):
    # Join texts into a single string for tokenization
    full_text = " ".join(texts) if texts else ""
    if full_text:
        full_text = "[CLS] " + full_text + " [SEP]"
    else:
        full_text = "[CLS] [SEP]"
    
    encoding = tokenizer(full_text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    
    # Prepare bounding boxes tensor [max_len, 8]
    bboxes = torch.zeros((max_len, 8), dtype=torch.float32)
    cls_bbox = torch.tensor([0, 0, 1, 0, 1, 1, 0, 1], dtype=torch.float32)  # Full page box for CLS and SEP
    
    bboxes[0] = cls_bbox
    # Fill bounding boxes for tokens after CLS (reserve space for SEP)
    num_boxes = min(len(boxes), max_len - 2)
    if num_boxes > 0:
        bboxes[1:1+num_boxes] = torch.tensor(boxes[:num_boxes], dtype=torch.float32)
    
    # SEP box same as CLS
    if (1+num_boxes) < max_len:
        bboxes[1+num_boxes] = cls_bbox
    
    # Save as a dict for loading during training
    torch.save({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bboxes": bboxes
    }, output_path)

def precompute_ocr(data_dir, output_dir, ocr_engine="tesseract", max_size=640, max_len=512, offset=0, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect image paths
    all_image_paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root, fname))
    
    # Apply offset and max_images filters
    if offset > 0:
        all_image_paths = all_image_paths[offset:]
    if max_images is not None:
        all_image_paths = all_image_paths[:max_images]
    
    print(f"Processing {len(all_image_paths)} images with OCR engine {ocr_engine}")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processed = 0
    
    for img_path in tqdm(all_image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            
            if ocr_engine == "tesseract":
                texts, boxes = get_ocr_results_tesseract(image)
            elif ocr_engine == "easyocr":
                texts, boxes = get_ocr_results_easyocr(image)
            elif ocr_engine == "pero":
                texts, boxes = get_ocr_results_pero(image)
            else:
                texts, boxes = [], []
            
            save_path = os.path.join(output_dir, os.path.basename(img_path) + ".pt")
            tokenize_and_save(texts, boxes, tokenizer, max_len, save_path)
            processed += 1
            
            # Cleanup
            del image
            gc.collect()
        
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    
    print(f"OCR extraction completed for {processed} images out of {len(all_image_paths)}")

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR Extraction with bounding boxes for DocFormer")
    parser.add_argument('--data_dir', required=True, help='Path to the image dataset root')
    parser.add_argument('--output_dir', required=True, help='Directory to save extracted OCR token files')
    parser.add_argument('--ocr_engine', default='tesseract', choices=['tesseract', 'easyocr', 'pero'], help='OCR engine to use')
    parser.add_argument('--max_size', type=int, default=1024, help='Maximum image dimension size')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum token length for OCR tokens')
    parser.add_argument('--offset', type=int, default=0, help='Starting index in image list')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    args = parser.parse_args()
    
    precompute_ocr(args.data_dir, args.output_dir, args.ocr_engine, args.max_size, args.max_len, args.offset, args.max_images)

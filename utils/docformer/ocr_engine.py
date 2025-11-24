import torch
import logging
from PIL import Image
from typing import List, Dict, Optional
import os
import numpy as np

class UnifiedOCREngine:
    """Unified OCR interface supporting multiple engines"""
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.processor = None
        self.model = None
        
        # Initialize selected OCR engine
        try:
            if config.ocr_engine == "tesseract":
                self._init_tesseract()
            elif config.ocr_engine == "trocr":
                self._init_trocr()
            elif config.ocr_engine == "pero":
                self._init_pero()
            elif config.ocr_engine == "easyocr":
                self._init_easyocr()
            elif config.ocr_engine == "paddleocr":
                self._init_paddleocr()
            else:
                raise ValueError(f"Unsupported OCR engine: {config.ocr_engine}")
        except Exception as e:
            print(f"Failed to initialize {config.ocr_engine}: {str(e)}")
            print("Falling back to dummy OCR")
            self._init_dummy()
    """
    def _init_tesseract(self):
        #Initialize Tesseract OCR
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
            pytesseract.get_tesseract_version()
            print("Tesseract OCR initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Tesseract: {str(e)}")
            raise
    """
    def _init_trocr(self):
        """Initialize TrOCR"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-printed",
                use_fast=True
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed"
            )
            self.model.to(self.config.ocr_device)
            self.model.eval()
            print("TrOCR initialized successfully")
        except Exception as e:
            print(f"Failed to initialize TrOCR: {str(e)}")
            raise
    """
    def _init_pero(self):
        #Initialize Pero OCR
        try:
            from pero_ocr.ocr_engine import OCREngine
            self.engine = OCREngine(config_path="pero_ocr/config.yaml")
            if torch.cuda.is_available():
                self.engine.use_gpu()
            print("Pero OCR initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Pero OCR: {str(e)}")
            raise

    def _init_easyocr(self):
        #Initialize EasyOCR
        try:
            import easyocr
            self.engine = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {str(e)}")
            raise

    def _init_paddleocr(self):
        #Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.engine = PaddleOCR(use_angle_cls=True, lang='en', 
                                  use_gpu=torch.cuda.is_available())
            print("PaddleOCR initialized successfully")
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {str(e)}")
            raise

    def _init_dummy(self):
        Initialize dummy OCR for fallback
        self.engine = "dummy"
        print("Dummy OCR initialized")
    """
    @torch.no_grad()
    def process_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Process batch of images with selected OCR engine"""
        if self.config.ocr_engine == "tesseract":
            return self._process_tesseract(images)
        elif self.config.ocr_engine == "trocr":
            return self._process_trocr(images)
        elif self.config.ocr_engine == "pero":
            return self._process_pero(images)
        elif self.config.ocr_engine == "easyocr":
            return self._process_easyocr(images)
        elif self.config.ocr_engine == "paddleocr":
            return self._process_paddleocr(images)
        else:
            return self._process_dummy(images)

    def _process_tesseract(self, images):
        """Process with Tesseract"""
        import pytesseract
        results = []
        for img in images:
            try:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                texts = []
                bboxes = []
                
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > self.config.ocr_confidence_threshold * 100:
                        text = data['text'][i].strip()
                        if text:
                            texts.append(text)
                            # Normalize coordinates
                            x = data['left'][i] / img.width
                            y = data['top'][i] / img.height
                            w = data['width'][i] / img.width
                            h = data['height'][i] / img.height
                            
                            # 8-coordinate format
                            bbox = [x, y, x+w, y, x+w, y+h, x, y+h]
                            bboxes.append(bbox)
                
                results.append({
                    "text": " ".join(texts) if texts else "[UNK]",
                    "bboxes": bboxes
                })
            except Exception as e:
                print(f"Tesseract processing error: {str(e)}")
                results.append({"text": "[UNK]", "bboxes": []})
        
        return results

    def _process_trocr(self, images):
        """Process with TrOCR"""
        try:
            # Process in batches
            all_texts = []
            for i in range(0, len(images), self.config.ocr_batch_size):
                batch = images[i:i + self.config.ocr_batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.config.ocr_device)
                generated_ids = self.model.generate(**inputs, max_length=50)
                texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                all_texts.extend(texts)
            
            # TrOCR doesn't provide bboxes, create dummy ones
            return [{"text": text if text else "[UNK]", "bboxes": []} for text in all_texts]
        except Exception as e:
            print(f"TrOCR processing error: {str(e)}")
            return [{"text": "[UNK]", "bboxes": []} for _ in images]

    def _process_easyocr(self, images):
        """Process with EasyOCR"""
        results = []
        for img in images:
            try:
                img_array = np.array(img)
                ocr_results = self.engine.readtext(img_array)
                
                texts = []
                bboxes = []
                
                for (bbox_coords, text, confidence) in ocr_results:
                    if confidence > self.config.ocr_confidence_threshold:
                        texts.append(text)
                        # Normalize coordinates
                        bbox_norm = []
                        for point in bbox_coords:
                            bbox_norm.extend([point[0]/img.width, point[1]/img.height])
                        bboxes.append(bbox_norm)
                
                results.append({
                    "text": " ".join(texts) if texts else "[UNK]",
                    "bboxes": bboxes
                })
            except Exception as e:
                print(f"EasyOCR processing error: {str(e)}")
                results.append({"text": "[UNK]", "bboxes": []})
        
        return results

    def _process_paddleocr(self, images):
        """Process with PaddleOCR"""
        results = []
        for img in images:
            try:
                img_array = np.array(img)
                ocr_results = self.engine.ocr(img_array, cls=True)
                
                texts = []
                bboxes = []
                
                if ocr_results[0]:
                    for line in ocr_results[0]:
                        bbox_coords, (text, confidence) = line
                        if confidence > self.config.ocr_confidence_threshold:
                            texts.append(text)
                            # Normalize coordinates
                            bbox_norm = []
                            for point in bbox_coords:
                                bbox_norm.extend([point[0]/img.width, point[1]/img.height])
                            bboxes.append(bbox_norm)
                
                results.append({
                    "text": " ".join(texts) if texts else "[UNK]",
                    "bboxes": bboxes
                })
            except Exception as e:
                print(f"PaddleOCR processing error: {str(e)}")
                results.append({"text": "[UNK]", "bboxes": []})
        
        return results

    def _process_pero(self, images):
        """Process with Pero OCR"""
        results = []
        for img in images:
            try:
                output = self.engine.process_page(img)
                texts = [line.transcription for line in output.lines if line.transcription]
                bboxes = [self._parse_pero_bbox(line, img.width, img.height) for line in output.lines]
                
                results.append({
                    "text": " ".join(texts) if texts else "[UNK]",
                    "bboxes": bboxes
                })
            except Exception as e:
                print(f"Pero OCR processing error: {str(e)}")
                results.append({"text": "[UNK]", "bboxes": []})
        return results
    
    def _parse_pero_bbox(self, line, img_width, img_height):
        """Parse Pero bounding box and normalize"""
        coords = line.geometry
        normalized = []
        for point in coords:
            normalized.extend([point[0]/img_width, point[1]/img_height])
        return normalized

    def _process_dummy(self, images):
        """Dummy OCR for fallback"""
        return [{"text": "[UNK]", "bboxes": []} for _ in images]

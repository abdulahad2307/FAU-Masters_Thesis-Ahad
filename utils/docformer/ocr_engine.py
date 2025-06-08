import torch
import logging
from PIL import Image
from typing import List, Dict, Optional
import pytesseract
from pero_ocr.ocr_engine import OCREngine
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

class OCRWrapper:
    """Unified OCR interface with GPU acceleration support"""
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.processor = None
        self.model = None
        
        if config.ocr_engine == "trocr":
            self._init_trocr()
        elif config.ocr_engine == "pero":
            self._init_pero()
        elif config.ocr_engine == "tesseract":
            self._init_tesseract()
        else:
            raise ValueError(f"Unsupported OCR engine: {config.ocr_engine}")

    def _init_trocr(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.model.to(self.config.ocr_device)
        self.model.eval()
        
    def _init_pero(self):
        self.engine = OCREngine(config_path="pero_ocr/config.yaml")
        if torch.cuda.is_available():
            self.engine.use_gpu()
            
    def _init_tesseract(self):
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
    @torch.no_grad()
    def process_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Process batch of images with selected OCR engine"""
        if self.config.ocr_engine == "trocr":
            return self._process_trocr(images)
        elif self.config.ocr_engine == "pero":
            return self._process_pero(images)
        return self._process_tesseract(images)
    
    def _process_trocr(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.config.ocr_device)
        generated_ids = self.model.generate(**inputs)
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return [{"text": text, "bboxes": []} for text in texts]  # BBox not available
    
    def _process_pero(self, images):
        results = []
        for img in images:
            output = self.engine.process_page(img)
            results.append({
                "text": " ".join([l.transcription for l in output.lines]),
                "bboxes": [self._parse_pero_bbox(line) for line in output.lines]
            })
        return results
    
    def _process_tesseract(self, images):
        results = []
        for img in images:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            texts = []
            bboxes = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 60:  # Confidence threshold
                    texts.append(data['text'][i])
                    bboxes.append([
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    ])
            results.append({
                "text": " ".join(texts),
                "bboxes": bboxes
            })
        return results
    
    def _parse_pero_bbox(self, line):
        return [line.geometry[0][0], line.geometry[0][1],
                line.geometry[1][0], line.geometry[1][1]]

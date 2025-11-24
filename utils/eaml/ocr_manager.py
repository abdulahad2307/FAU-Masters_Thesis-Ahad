import time
import torch
import torch.nn as nn
from functools import lru_cache

class OCRManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRManager, cls).__new__(cls)
            cls._instance._ocr_engines = {}
        return cls._instance
    
    def get_ocr_engine(self, engine_type, **kwargs):
        """Get or initialize an OCR engine of the specified type"""
        if engine_type not in self._ocr_engines:
            start_time = time.time()
            print(f"Initializing {engine_type} OCR engine...", end=" ")
            
            try:
                if engine_type == "trocr":
                    self._ocr_engines[engine_type] = self._initialize_trocr(**kwargs)
                elif engine_type == "tesseract":
                    self._ocr_engines[engine_type] = self._initialize_tesseract(**kwargs)
                elif engine_type == "easyocr":
                    self._ocr_engines[engine_type] = self._initialize_easyocr(**kwargs)
                else:
                    raise ValueError(f"Unsupported OCR engine: {engine_type}")
                
                print(f"done in {time.time()-start_time:.2f}s")
            except Exception as e:
                print(f"\nFailed to initialize {engine_type} OCR engine: {str(e)}")
                raise
                
        return self._ocr_engines[engine_type]
    
    def _initialize_trocr(self, model_name="microsoft/trocr-base-handwritten", **kwargs):
        """Initialize TrOCR model"""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Initialize missing pooler weights to silence the warning
        if not hasattr(model.encoder, 'pooler'):
            class DummyPooler(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                    
            model.encoder.pooler = DummyPooler(model.encoder.config)
            # Initialize weights
            nn.init.xavier_uniform_(model.encoder.pooler.dense.weight)
            nn.init.zeros_(model.encoder.pooler.dense.bias)
            
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        return {"processor": processor, "model": model}
    
    def _initialize_tesseract(self, **kwargs):
        """Initialize Tesseract OCR"""
        import pytesseract
        return {"engine": pytesseract}
    
    def _initialize_easyocr(self, languages=["en"], **kwargs):
        """Initialize EasyOCR"""
        import easyocr
        reader = easyocr.Reader(languages)
        return {"reader": reader}

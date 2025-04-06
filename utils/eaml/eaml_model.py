import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion_module import FusionModule

class EAMLModel(nn.Module):
    def __init__(self, num_classes=16, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.fusion_module = FusionModule(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize OCR model
        self.ocr_processor = None
        self.ocr_model = None

    def init_ocr(self):
        if self.ocr_processor is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.ocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-handwritten", 
                use_fast=True
            )
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten"
            )
            # Freeze OCR model
            for param in self.ocr_model.parameters():
                param.requires_grad = False

    def forward(self, images, texts):
        image_feat = self.image_encoder(images)
        text_feat = self.text_encoder(texts)
        fused_feat = self.fusion_module(image_feat, text_feat)
        return self.classifier(fused_feat)
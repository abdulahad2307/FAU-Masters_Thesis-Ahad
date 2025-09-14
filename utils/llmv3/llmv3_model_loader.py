import torch
from transformers import LayoutLMv3ForSequenceClassification

def load_model(num_labels):
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=num_labels
    )
    return model

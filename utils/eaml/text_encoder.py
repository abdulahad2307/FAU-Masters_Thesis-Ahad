import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", embed_dim=512):
        """
        Implementation fo Text feature extractor using a pretrained BERT model.
        
        Parameters:
            model_name (str): Name of the BERT model.
            embed_dim (int): Output feature dimension.
        """
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)
        
        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = True #False  ##  # True for trainable, False for frozen

    def forward(self, text):

        if text is None:
            raise ValueError("Text input cannot be None")
        if isinstance(text, str):
            text = self.tokenizer(text, return_tensors="pt")
        
        text = {key: val.to(next(self.bert.parameters()).device) 
               for key, val in text.items()}
        
        outputs = self.bert(**text)
        return self.fc(outputs.last_hidden_state[:, 0, :])
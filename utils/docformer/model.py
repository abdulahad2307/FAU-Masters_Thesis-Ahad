import torch
import torch.nn as nn
import math
import time
from transformers import LayoutLMModel, LayoutLMConfig
from torchvision.models import resnet50
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SpatialEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.y_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        max_rel_pos = 2 * config.max_position_embeddings - 1
        self.rel_x_embeddings = nn.Embedding(max_rel_pos, config.hidden_size)
        self.rel_y_embeddings = nn.Embedding(max_rel_pos, config.hidden_size)
        self.width_embeddings = nn.Linear(1, config.hidden_size)
        self.height_embeddings = nn.Linear(1, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, bboxes):
        bboxes = bboxes.float()
        x1, y1, x2, y2, x3, y3, x4, y4 = bboxes.unbind(-1)
        x1_norm = (x1 * (self.config.max_position_embeddings - 1)).long()
        y1_norm = (y1 * (self.config.max_position_embeddings - 1)).long()
        x2_norm = (x2 * (self.config.max_position_embeddings - 1)).long()
        y2_norm = (y2 * (self.config.max_position_embeddings - 1)).long()
        width = (x2 - x1).unsqueeze(-1).float()
        height = (y3 - y1).unsqueeze(-1).float()
        x1_emb = self.x_embeddings(x1_norm)
        y1_emb = self.y_embeddings(y1_norm)
        x2_emb = self.x_embeddings(x2_norm)
        y2_emb = self.y_embeddings(y2_norm)
        width_emb = self.width_embeddings(width)
        height_emb = self.height_embeddings(height)
        rel_x = (x2_norm - x1_norm) + self.config.max_position_embeddings - 1
        rel_y = (y2_norm - y1_norm) + self.config.max_position_embeddings - 1
        rel_x_emb = self.rel_x_embeddings(rel_x.clamp(0, 2*self.config.max_position_embeddings-2))
        rel_y_emb = self.rel_y_embeddings(rel_y.clamp(0, 2*self.config.max_position_embeddings-2))
        embeddings = x1_emb + y1_emb + x2_emb + y2_emb + width_emb + height_emb + rel_x_emb + rel_y_emb
        return self.LayerNorm(embeddings)

class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, config.hidden_size, 1)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        features = self.conv(features)
        features = self.pool(features)
        return features.flatten(2).transpose(1, 2)

class MultiModalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.spatial_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.spatial_key = nn.Linear(config.hidden_size, self.all_head_size)
        max_rel_pos = 2 * config.max_position_embeddings - 1
        self.rel_pos_bias = nn.Embedding(max_rel_pos, self.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, spatial_embeddings, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        spatial_query_layer = self.spatial_query(spatial_embeddings)
        spatial_key_layer = self.spatial_key(spatial_embeddings)
        query_layer = mixed_query_layer.view(*mixed_query_layer.size()[:-1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        key_layer = mixed_key_layer.view(*mixed_key_layer.size()[:-1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        value_layer = mixed_value_layer.view(*mixed_value_layer.size()[:-1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        spatial_query_layer = spatial_query_layer.view(*spatial_query_layer.size()[:-1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        spatial_key_layer = spatial_key_layer.view(*spatial_key_layer.size()[:-1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        spatial_attention_scores = torch.matmul(spatial_query_layer, spatial_key_layer.transpose(-1, -2))
        seq_length = hidden_states.size(1)
        position_ids = torch.arange(seq_length, device=hidden_states.device)
        rel_pos = position_ids.unsqueeze(1) - position_ids.unsqueeze(0) + self.config.max_position_embeddings - 1
        rel_pos_bias = self.rel_pos_bias(rel_pos.clamp(0, 2*self.config.max_position_embeddings-2)).permute(2, 0, 1)
        attention_scores = (attention_scores + spatial_attention_scores + rel_pos_bias.unsqueeze(0)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        return context_layer.view(*context_layer.size()[:-2], self.all_head_size)

class DocFormer(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config
        self.text_embeddings = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
        self.visual_backbone = VisualBackbone(config)
        self.spatial_embeddings = SpatialEmbeddings(config)
        self.encoder = nn.ModuleList([nn.ModuleDict({
            'attention': MultiModalSelfAttention(config),
            'output': nn.Linear(config.hidden_size, config.hidden_size)
        }) for _ in range(config.num_hidden_layers)])
        self.mm_mlm_head = nn.Linear(config.hidden_size, self.text_embeddings.config.vocab_size)
        self.ltr_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3*224*224)
        )
        self.tdi_head = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, num_classes) if num_classes else None
        self.init_weights()

    def forward(self, input_ids, bboxes, attention_mask, pixel_values, labels=None, task="pretrain"):
        text_features = self.text_embeddings(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        visual_features = self.visual_backbone(pixel_values)
        spatial_features = self.spatial_embeddings(bboxes)
        combined_features = torch.cat([text_features, visual_features], dim=1)
        combined_spatial = torch.cat([spatial_features, spatial_features[:, :visual_features.size(1)]], dim=1)
        extended_attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), visual_features.size(1), 
                                      device=attention_mask.device)], dim=1).unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer in self.encoder:
            attention_output = layer['attention'](combined_features, combined_spatial, extended_attention_mask)
            layer_output = layer['output'](attention_output)
            combined_features = layer_output + combined_features
        if task == "pretrain":
            return {
                "mm_mlm_logits": self.mm_mlm_head(combined_features[:, :text_features.size(1)]),
                "ltr_output": self.ltr_head(combined_features.mean(dim=1)).view(-1,3,224,224),
                "tdi_logits": self.tdi_head(combined_features.mean(dim=1))
            }
        elif task == "classification":
            logits = self.classifier(combined_features[:, 0])
            return {"logits": logits, "loss": torch.nn.functional.cross_entropy(logits, labels)} if labels else {"logits": logits}
        return {"last_hidden_state": combined_features}

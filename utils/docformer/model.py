import torch
import torch.nn as nn
import math
import logging
from transformers import BertModel, BertConfig
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Optional

class SpatialEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Coordinate embeddings
        self.x_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.y_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Relative position embeddings
        max_rel_pos = 2 * config.max_position_embeddings - 1
        self.rel_x_embeddings = nn.Embedding(max_rel_pos, config.hidden_size)
        self.rel_y_embeddings = nn.Embedding(max_rel_pos, config.hidden_size)
        
        # Size embeddings
        self.width_embeddings = nn.Linear(1, config.hidden_size)
        self.height_embeddings = nn.Linear(1, config.hidden_size)
        
        # Normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, bboxes):
        batch_size, seq_len = bboxes.shape[:2]
        
        # Handle empty or invalid bboxes
        bboxes = bboxes.float()
        bboxes = torch.clamp(bboxes, 0.0, 1.0)
        
        # Extract coordinates (handle both 4 and 8 coordinate formats)
        if bboxes.size(-1) >= 4:
            x1, y1, x2, y2 = bboxes[:, :, 0], bboxes[:, :, 1], bboxes[:, :, 2], bboxes[:, :, 3]
        else:
            # Handle degenerate case
            x1 = y1 = x2 = y2 = torch.zeros_like(bboxes[:, :, 0])
        
        # Normalize to embedding indices
        x1_norm = (x1 * (self.config.max_position_embeddings - 1)).long().clamp(0, self.config.max_position_embeddings - 1)
        y1_norm = (y1 * (self.config.max_position_embeddings - 1)).long().clamp(0, self.config.max_position_embeddings - 1)
        x2_norm = (x2 * (self.config.max_position_embeddings - 1)).long().clamp(0, self.config.max_position_embeddings - 1)
        y2_norm = (y2 * (self.config.max_position_embeddings - 1)).long().clamp(0, self.config.max_position_embeddings - 1)
        
        # Calculate dimensions
        width = torch.clamp(x2 - x1, 0.0, 1.0).unsqueeze(-1)
        height = torch.clamp(y2 - y1, 0.0, 1.0).unsqueeze(-1)
        
        # Get embeddings
        x1_emb = self.x_embeddings(x1_norm)
        y1_emb = self.y_embeddings(y1_norm)
        x2_emb = self.x_embeddings(x2_norm)
        y2_emb = self.y_embeddings(y2_norm)
        width_emb = self.width_embeddings(width)
        height_emb = self.height_embeddings(height)
        
        # Relative positions
        rel_x = (x2_norm - x1_norm) + self.config.max_position_embeddings - 1
        rel_y = (y2_norm - y1_norm) + self.config.max_position_embeddings - 1
        rel_x = rel_x.clamp(0, 2 * self.config.max_position_embeddings - 2)
        rel_y = rel_y.clamp(0, 2 * self.config.max_position_embeddings - 2)
        
        rel_x_emb = self.rel_x_embeddings(rel_x)
        rel_y_emb = self.rel_y_embeddings(rel_y)
        
        # Combine all embeddings
        embeddings = x1_emb + y1_emb + x2_emb + y2_emb + width_emb + height_emb + rel_x_emb + rel_y_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load ResNet50 with proper weights
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature projection
        self.conv = nn.Conv2d(2048, config.hidden_size, 1)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))  # Fixed 16x16 grid
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, pixel_values):
        # Extract features
        features = self.backbone(pixel_values)  # [B, 2048, H, W]
        features = self.conv(features)          # [B, 768, H, W]
        features = self.pool(features)          # [B, 768, 16, 16]
        
        # Flatten and transpose
        features = features.flatten(2).transpose(1, 2)  # [B, 256, 768]
        features = self.layer_norm(features)
        
        return features

class MultiModalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Text attention
        self.text_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.text_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.text_value = nn.Linear(config.hidden_size, self.all_head_size)

        # Visual attention
        self.visual_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.visual_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.visual_value = nn.Linear(config.hidden_size, self.all_head_size)

        # Shared spatial attention
        self.spatial_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.spatial_key = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_features, visual_features, spatial_features, attention_mask=None):
        batch_size = text_features.size(0)
        text_len = text_features.size(1)
        visual_len = visual_features.size(1)

        # Text attention
        text_q = self.transpose_for_scores(self.text_query(text_features))
        text_k = self.transpose_for_scores(self.text_key(text_features))
        text_v = self.transpose_for_scores(self.text_value(text_features))

        # Visual attention
        visual_q = self.transpose_for_scores(self.visual_query(visual_features))
        visual_k = self.transpose_for_scores(self.visual_key(visual_features))
        visual_v = self.transpose_for_scores(self.visual_value(visual_features))

        # Spatial attention (shared)
        spatial_q = self.transpose_for_scores(self.spatial_query(spatial_features))
        spatial_k = self.transpose_for_scores(self.spatial_key(spatial_features))

        # Compute attention scores
        text_scores = torch.matmul(text_q, text_k.transpose(-1, -2))
        visual_scores = torch.matmul(visual_q, visual_k.transpose(-1, -2))

        # Add spatial attention
        text_spatial_scores = torch.matmul(spatial_q[:, :, :text_len], spatial_k[:, :, :text_len].transpose(-1, -2))
        visual_spatial_scores = torch.matmul(spatial_q[:, :, text_len:], spatial_k[:, :, text_len:].transpose(-1, -2))

        # Combine scores
        text_scores = (text_scores + text_spatial_scores) / math.sqrt(self.attention_head_size)
        visual_scores = (visual_scores + visual_spatial_scores) / math.sqrt(self.attention_head_size)

        # Apply attention masks
        if attention_mask is not None:
            text_mask = attention_mask[:, :text_len].unsqueeze(1).unsqueeze(2)
            visual_mask = attention_mask[:, text_len:].unsqueeze(1).unsqueeze(2)

            text_scores = text_scores + (1.0 - text_mask) * -10000.0
            visual_scores = visual_scores + (1.0 - visual_mask) * -10000.0

        # Softmax and apply to values
        text_probs = torch.softmax(text_scores, dim=-1)
        visual_probs = torch.softmax(visual_scores, dim=-1)

        text_probs = self.dropout(text_probs)
        visual_probs = self.dropout(visual_probs)

        text_context = torch.matmul(text_probs, text_v)
        visual_context = torch.matmul(visual_probs, visual_v)

        # Reshape and combine
        text_context = text_context.permute(0, 2, 1, 3).contiguous()
        visual_context = visual_context.permute(0, 2, 1, 3).contiguous()

        text_context = text_context.view(batch_size, text_len, self.all_head_size)
        visual_context = visual_context.view(batch_size, visual_len, self.all_head_size)

        # Residual connections
        text_output = self.layer_norm(text_context + text_features)
        visual_output = self.layer_norm(visual_context + visual_features)

        return text_output, visual_output

class DocFormer(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config
        
        # Initialize BERT for text processing
        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings
        )
        self.text_embeddings = BertModel(bert_config)
        
        # Visual and spatial components
        self.visual_backbone = VisualBackbone(config)
        self.spatial_embeddings = SpatialEmbeddings(config)
        
        # Multi-modal attention layers
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiModalSelfAttention(config),
                'feed_forward': nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    nn.GELU(),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout_prob)
                )
            }) for _ in range(config.num_hidden_layers)
        ])
        
        # ALWAYS initialize pretraining heads (needed for progressive training)
        self.mm_mlm_head = nn.Linear(config.hidden_size, bert_config.vocab_size)
        self.ltr_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3 * 224 * 224)
        )
        self.tdi_head = nn.Linear(config.hidden_size, 1)
        
        # Classification head for fine-tuning
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, num_classes)
            )
        else:
            self.classifier = None
            
        # Initialize weights
        self.apply(self._init_weights)
        print(f"DocFormer initialized with {num_classes} classes")


    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    
    def forward(
    self, 
    input_ids, 
    bboxes, 
    attention_mask, 
    pixel_values, 
    labels=None, 
    task="finetune", 
    **kwargs  # Accept extra keys to avoid forward errors
):
        batch_size = input_ids.size(0)
        if self.config.training_stage == "text_pretrain":
            text_outputs = self.text_embeddings(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state

            # ALWAYS generate logits during text pretraining
            if self.classifier is not None:
                logits = self.classifier(text_features[:, 0])
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                return {"logits": logits, "loss": loss}

        # Normal multimodal processing for other stages
        # Initialize text_features FIRST to avoid UnboundLocalError
        text_outputs = self.text_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # This was missing in error case

        visual_features = self.visual_backbone(pixel_values)
        spatial_features = self.spatial_embeddings(bboxes)  # USES bboxes as required

        # Ensure spatial features match sequence length
        seq_len = text_features.size(1)  # Now properly initialized
        visual_len = visual_features.size(1)
        
        # Pad or truncate spatial features
        if spatial_features.size(1) < seq_len + visual_len:
            padding = torch.zeros(
                batch_size, 
                seq_len + visual_len - spatial_features.size(1), 
                self.config.hidden_size,
                device=spatial_features.device
            )
            spatial_features = torch.cat([spatial_features, padding], dim=1)
        else:
            spatial_features = spatial_features[:, :seq_len + visual_len]
        
        # Create extended attention mask
        visual_mask = torch.ones(batch_size, visual_len, device=attention_mask.device)
        extended_attention_mask = torch.cat([attention_mask, visual_mask], dim=1)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            text_features, visual_features = layer['attention'](
                text_features, visual_features, spatial_features, extended_attention_mask
            )
            
            # Feed forward
            text_features = layer['feed_forward'](text_features) + text_features
            visual_features = layer['feed_forward'](visual_features) + visual_features
        
        # Combine features
        combined_features = torch.cat([text_features, visual_features], dim=1)
        
        # Task-specific outputs
        if task == "pretrain":
            mm_mlm_logits = self.mm_mlm_head(text_features) if hasattr(self, 'mm_mlm_head') else None
            ltr_output = self.ltr_head(combined_features.mean(dim=1)).view(batch_size, 3, 224, 224) if hasattr(self, 'ltr_head') else None
            tdi_logits = self.tdi_head(combined_features.mean(dim=1)) if hasattr(self, 'tdi_head') else None
            
            return {
                "mm_mlm_logits": mm_mlm_logits,
                "ltr_output": ltr_output,
                "tdi_logits": tdi_logits
            }

        elif task == "finetune" and self.classifier is not None:
            # Use [CLS] token for classification
            logits = self.classifier(text_features[:, 0])
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return {"logits": logits, "loss": loss}
        
        return {"last_hidden_state": combined_features}

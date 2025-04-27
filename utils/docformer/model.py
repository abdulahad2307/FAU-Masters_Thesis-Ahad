import torch
import torch.nn as nn
import math
import time
from transformers import LayoutLMModel, LayoutLMConfig
from torchvision.models import resnet50
import logging
from typing import Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timer:
    """Enhanced timing helper with logging"""
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        logger.debug(f"{self.name} took {self.duration:.4f}s")

class SpatialEmbeddings(nn.Module):
    """Enhanced spatial embeddings with logging and error handling"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Absolute position embeddings
        self.x_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.y_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Relative position embeddings
        max_rel_pos = 2 * config.max_position_embeddings - 1
        self.rel_x_embeddings = nn.Embedding(max_rel_pos, config.hidden_size)
        self.rel_y_embeddings = nn.Embedding(max_rel_pos, config.hidden_size)
        
        # Size embeddings
        self.width_embeddings = nn.Linear(1, config.hidden_size)
        self.height_embeddings = nn.Linear(1, config.hidden_size)
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        logger.info("Initialized SpatialEmbeddings")
    
    def forward(self, bboxes):
        with Timer("SpatialEmbeddings"):
            try:
                bboxes = bboxes.float()
                batch_size, seq_len = bboxes.shape[:2]
                
                # Extract coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
                x1, y1, x2, y2, x3, y3, x4, y4 = bboxes.unbind(-1)
                
                # Normalize coordinates to be within embedding range [0, max_position_embeddings-1]
                def safe_normalize(coords):
                    coords = torch.clamp(coords, 0, 1)
                    normalized = (coords * (self.config.max_position_embeddings - 1)).long()
                    return torch.clamp(normalized, 0, self.config.max_position_embeddings - 1)
                
                x1_norm = safe_normalize(x1)
                y1_norm = safe_normalize(y1)
                x2_norm = safe_normalize(x2)
                y2_norm = safe_normalize(y2)
                
                # Calculate normalized width and height
                width = (x2 - x1).unsqueeze(-1).float()
                height = (y3 - y1).unsqueeze(-1).float()
                
                # Absolute position embeddings
                x1_emb = self.x_embeddings(x1_norm)
                y1_emb = self.y_embeddings(y1_norm)
                x2_emb = self.x_embeddings(x2_norm)
                y2_emb = self.y_embeddings(y2_norm)
                
                # Size embeddings
                width_emb = self.width_embeddings(width)
                height_emb = self.height_embeddings(height)
                
                # Relative position embeddings with safe bounds
                rel_x = (x2_norm - x1_norm) + self.config.max_position_embeddings - 1
                rel_y = (y2_norm - y1_norm) + self.config.max_position_embeddings - 1
                rel_x = torch.clamp(rel_x, 0, 2 * self.config.max_position_embeddings - 2)
                rel_y = torch.clamp(rel_y, 0, 2 * self.config.max_position_embeddings - 2)
                
                rel_x_emb = self.rel_x_embeddings(rel_x)
                rel_y_emb = self.rel_y_embeddings(rel_y)
                
                # Combine all embeddings
                embeddings = (x1_emb + y1_emb + x2_emb + y2_emb + 
                             width_emb + height_emb + 
                             rel_x_emb + rel_y_emb)
                
                embeddings = self.LayerNorm(embeddings)
                embeddings = self.dropout(embeddings)
                
                return embeddings
            except Exception as e:
                logger.error(f"Error in SpatialEmbeddings: {str(e)}")
                raise

class VisualBackbone(nn.Module):
    """Enhanced visual backbone with pretrained ResNet50"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        with Timer("Loading ResNet50"):
            try:
                resnet = resnet50(pretrained=True)
                
                # Remove the last two layers (avgpool and fc)
                self.backbone = nn.Sequential(*list(resnet.children())[:-2])
                
                # 1x1 conv to reduce channel dimension
                self.conv = nn.Conv2d(
                    in_channels=2048,
                    out_channels=config.hidden_size,
                    kernel_size=1
                )
                
                # Adaptive pooling to fixed size
                self.pool = nn.AdaptiveAvgPool2d((16, 16))
                
                logger.info("Initialized VisualBackbone with pretrained ResNet50")
            except Exception as e:
                logger.error(f"Failed to initialize VisualBackbone: {str(e)}")
                raise
    
    def forward(self, pixel_values):
        with Timer("VisualBackbone"):
            try:
                # pixel_values shape: (batch_size, 3, height, width)
                features = self.backbone(pixel_values)  # (batch_size, 2048, h/32, w/32)
                features = self.conv(features)         # (batch_size, hidden_size, h/32, w/32)
                features = self.pool(features)         # (batch_size, hidden_size, 16, 16)
                
                # Flatten spatial dimensions
                batch_size = features.shape[0]
                features = features.flatten(2).transpose(1, 2)  # (batch_size, 256, hidden_size)
                
                return features
            except Exception as e:
                logger.error(f"Error in VisualBackbone forward: {str(e)}")
                raise

class MultiModalSelfAttention(nn.Module):
    """Enhanced multi-modal self-attention with shared spatial embeddings"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Spatial attention projections (shared between modalities)
        self.spatial_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.spatial_key = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Relative position bias
        max_rel_pos = 2 * config.max_position_embeddings - 1
        self.rel_pos_bias = nn.Embedding(max_rel_pos, self.num_attention_heads)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        logger.info(f"Initialized MultiModalSelfAttention with {self.num_attention_heads} heads")
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, spatial_embeddings, attention_mask=None):
        with Timer("MultiModalSelfAttention"):
            try:
                # hidden_states shape: (batch_size, seq_len, hidden_size)
                # spatial_embeddings shape: (batch_size, seq_len, hidden_size)
                
                # Project queries, keys, values
                mixed_query_layer = self.query(hidden_states)
                mixed_key_layer = self.key(hidden_states)
                mixed_value_layer = self.value(hidden_states)
                
                # Project spatial queries and keys
                spatial_query_layer = self.spatial_query(spatial_embeddings)
                spatial_key_layer = self.spatial_key(spatial_embeddings)
                
                # Transpose for attention scores
                query_layer = self.transpose_for_scores(mixed_query_layer)
                key_layer = self.transpose_for_scores(mixed_key_layer)
                value_layer = self.transpose_for_scores(mixed_value_layer)
                
                spatial_query_layer = self.transpose_for_scores(spatial_query_layer)
                spatial_key_layer = self.transpose_for_scores(spatial_key_layer)
                
                # Calculate attention scores
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                spatial_attention_scores = torch.matmul(spatial_query_layer, spatial_key_layer.transpose(-1, -2))
                
                # Add relative position bias
                seq_length = hidden_states.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                rel_pos = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)
                rel_pos += self.config.max_position_embeddings - 1  # Center around 0
                rel_pos = torch.clamp(rel_pos, 0, 2 * self.config.max_position_embeddings - 2)
                rel_pos_bias = self.rel_pos_bias(rel_pos).permute(2, 0, 1)
                
                # Combine attention scores
                attention_scores = (attention_scores + spatial_attention_scores + rel_pos_bias.unsqueeze(0)) / 3
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)
                
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                
                # Normalize attention scores
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                attention_probs = self.dropout(attention_probs)
                
                # Context layer
                context_layer = torch.matmul(attention_probs, value_layer)
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
                context_layer = context_layer.view(*new_context_layer_shape)
                
                return context_layer
            except Exception as e:
                logger.error(f"Error in MultiModalSelfAttention forward: {str(e)}")
                raise

class DocFormerLayer(nn.Module):
    """Enhanced DocFormer transformer layer with residual connections"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiModalSelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()
        
        logger.info("Initialized DocFormerLayer")
    
    def forward(self, hidden_states, spatial_embeddings, attention_mask=None):
        with Timer("DocFormerLayer"):
            try:
                # Self-attention
                attention_output = self.attention(
                    hidden_states=hidden_states,
                    spatial_embeddings=spatial_embeddings,
                    attention_mask=attention_mask
                )
                
                # Intermediate and output
                intermediate_output = self.intermediate(attention_output)
                intermediate_output = self.activation(intermediate_output)
                layer_output = self.output(intermediate_output)
                layer_output = self.dropout(layer_output)
                layer_output = self.LayerNorm(layer_output + attention_output)
                
                return layer_output
            except Exception as e:
                logger.error(f"Error in DocFormerLayer forward: {str(e)}")
                raise

class DocFormerEncoder(nn.Module):
    """Enhanced DocFormer encoder with layer-wise logging"""
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DocFormerLayer(config) for _ in range(config.num_hidden_layers)])
        logger.info(f"Initialized DocFormerEncoder with {config.num_hidden_layers} layers")
    
    def forward(self, hidden_states, spatial_embeddings, attention_mask=None):
        for layer_idx, layer_module in enumerate(self.layer):
            with Timer(f"EncoderLayer_{layer_idx}"):
                hidden_states = layer_module(hidden_states, spatial_embeddings, attention_mask)
        return hidden_states

class DocFormer(nn.Module):
    """Complete enhanced DocFormer model with pre-training and classification support"""
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config
        
        with Timer("Initializing LayoutLM"):
            try:
                layoutlm_config = LayoutLMConfig.from_pretrained("microsoft/layoutlm-base-uncased")
                self.text_embeddings = LayoutLMModel.from_pretrained(
                    "microsoft/layoutlm-base-uncased",
                    config=layoutlm_config
                )
                logger.info("Loaded pretrained LayoutLM weights")
            except Exception as e:
                logger.error(f"Failed to initialize LayoutLM: {str(e)}")
                raise
        
        # Visual backbone
        self.visual_backbone = VisualBackbone(config)
        
        # Spatial embeddings
        self.spatial_embeddings = SpatialEmbeddings(config)
        
        # Encoder
        self.encoder = DocFormerEncoder(config)
        
        # Pre-training heads
        self.mm_mlm_head = nn.Linear(config.hidden_size, layoutlm_config.vocab_size)
        self.ltr_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3 * 224 * 224)  # For reconstructing 224x224 RGB images
        )
        self.tdi_head = nn.Linear(config.hidden_size, 1)
        
        # Classification head
        if num_classes is not None:
            self.classifier = nn.Linear(config.hidden_size, num_classes)
            logger.info(f"Initialized classifier with {num_classes} classes")
        else:
            self.classifier = None
        
        self.init_weights()
        logger.info("DocFormer model initialized successfully")

    def init_weights(self):
        """Enhanced weight initialization with logging"""
        try:
            # Initialize visual backbone
            for name, module in self.visual_backbone.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            
            # Initialize pre-training heads
            self.mm_mlm_head.weight.data = self.text_embeddings.embeddings.word_embeddings.weight.data
            self.mm_mlm_head.bias.data = torch.zeros_like(self.mm_mlm_head.bias.data)
            
            for module in self.ltr_head.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
            
            self.tdi_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.tdi_head.bias.data.zero_()
            
            # Initialize classifier if it exists
            if self.classifier is not None:
                nn.init.xavier_uniform_(self.classifier.weight)
                if self.classifier.bias is not None:
                    self.classifier.bias.data.zero_()
            
            logger.info("Weights initialized successfully")
        except Exception as e:
            logger.error(f"Error in weight initialization: {str(e)}")
            raise

    def forward(
        self,
        input_ids=None,
        bboxes=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        task="pretrain"
    ):
        with Timer("DocFormerForward"):
            try:
                # Extract features from all modalities
                text_features = self.text_embeddings(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                ).last_hidden_state
                
                visual_features = self.visual_backbone(pixel_values)
                spatial_features = self.spatial_embeddings(bboxes)
                
                # Combine text and visual features
                combined_features = torch.cat([text_features, visual_features], dim=1)
                combined_spatial = torch.cat([spatial_features, spatial_features[:, :visual_features.size(1)]], dim=1)
                
                # Extend attention mask for visual features
                if attention_mask is not None:
                    visual_attention_mask = torch.ones(
                        (attention_mask.size(0), visual_features.size(1)),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    extended_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
                    extended_attention_mask = extended_attention_mask[:, None, None, :]
                    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                else:
                    extended_attention_mask = None
                
                # Pass through encoder
                encoder_outputs = self.encoder(
                    hidden_states=combined_features,
                    spatial_embeddings=combined_spatial,
                    attention_mask=extended_attention_mask
                )
                
                # Task-specific outputs
                if task == "pretrain":
                    # Multi-modal masked language modeling
                    mm_mlm_logits = self.mm_mlm_head(encoder_outputs[:, :text_features.size(1)])
                    
                    # Learn to reconstruct
                    ltr_output = self.ltr_head(encoder_outputs.mean(dim=1))
                    ltr_output = ltr_output.view(-1, 3, 224, 224)
                    
                    # Text describes image
                    tdi_logits = self.tdi_head(encoder_outputs.mean(dim=1))
                    
                    return {
                        "mm_mlm_logits": mm_mlm_logits,
                        "ltr_output": ltr_output,
                        "tdi_logits": tdi_logits,
                        "encoder_outputs": encoder_outputs
                    }
                elif task == "classification" and self.classifier is not None:
                    # Classification task - use [CLS] token representation
                    cls_output = encoder_outputs[:, 0]  # First token is [CLS]
                    logits = self.classifier(cls_output)
                    
                    # Calculate loss if labels provided
                    loss = None
                    if labels is not None:
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                    
                    return {
                        "logits": logits,
                        "loss": loss,
                        "last_hidden_state": encoder_outputs
                    }
                else:
                    # For other downstream tasks, return the encoder outputs
                    return {
                        "last_hidden_state": encoder_outputs,
                        "text_features": encoder_outputs[:, :text_features.size(1)],
                        "visual_features": encoder_outputs[:, text_features.size(1):]
                    }
            except Exception as e:
                logger.error(f"Error in DocFormer forward: {str(e)}")
                raise
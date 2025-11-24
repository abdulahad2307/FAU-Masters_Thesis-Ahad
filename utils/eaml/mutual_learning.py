import torch
import torch.nn.functional as F

class TruncatedKLDLoss(torch.nn.Module):
    def __init__(self, threshold=0.1):
        """
        Implementation of truncated KL Divergence loss for mutual learning.
        
        Parameters:
            threshold (float): Minimum value to truncate KL loss.
        """
        super(TruncatedKLDLoss, self).__init__()
        self.threshold = threshold
        
    def forward(self, p_logits, q_logits):
        """
        Compute truncated KL divergence between two distributions.
        
        Args:
            p_logits: Predicted logits from first model
            q_logits: Predicted logits from second model
        """
        p = F.softmax(p_logits, dim=1)
        q = F.softmax(q_logits, dim=1)
        
        # KL(p||q)
        kl_loss = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=1)
        
        # Truncate values below threshold
        truncated_loss = torch.mean(torch.clamp(kl_loss, min=self.threshold))
        
        return truncated_loss

class MutualLearningLoss(torch.nn.Module):
    def __init__(self, cls_weight=1.0, kld_weight=0.3, threshold=0.1):
        """
        Combined loss for mutual learning with classification and KL divergence.
        
        Parameters:
            cls_weight (float): Weight for classification loss
            kld_weight (float): Weight for KL divergence loss
            threshold (float): Threshold for truncated KL divergence
        """
        super(MutualLearningLoss, self).__init__()
        self.cls_weight = cls_weight
        self.kld_weight = kld_weight
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.kld_criterion = TruncatedKLDLoss(threshold=threshold)
        
    def forward(self, outputs, labels):
        """
        Compute the combined mutual learning loss.
        
        Args:
            outputs: Dictionary containing logits from different branches
            labels: Ground truth labels
        """
        # Extract logits
        image_logits = outputs['image_logits']
        text_logits = outputs['text_logits']
        fusion_logits = outputs['fusion_logits']
        
        # Classification losses
        img_cls_loss = self.cls_criterion(image_logits, labels)
        txt_cls_loss = self.cls_criterion(text_logits, labels)
        fusion_cls_loss = self.cls_criterion(fusion_logits, labels)
        
        # Mutual learning losses (KL divergence)
        img_txt_loss = self.kld_criterion(image_logits, text_logits)
        img_fusion_loss = self.kld_criterion(image_logits, fusion_logits)
        txt_fusion_loss = self.kld_criterion(text_logits, fusion_logits)
        
        # Total loss
        cls_loss = fusion_cls_loss + 0.5 * (img_cls_loss + txt_cls_loss)
        kld_loss = img_txt_loss + img_fusion_loss + txt_fusion_loss
        
        total_loss = self.cls_weight * cls_loss + self.kld_weight * kld_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'kld_loss': kld_loss,
            'img_cls_loss': img_cls_loss,
            'txt_cls_loss': txt_cls_loss,
            'fusion_cls_loss': fusion_cls_loss
        }

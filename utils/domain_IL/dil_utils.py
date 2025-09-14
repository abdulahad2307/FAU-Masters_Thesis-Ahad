import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
#from typing import Dict, List, Optional, Tuple, Union

# ----------- Domain Incremental Strategies -----------

class DomainIncrementalStrategy:
    """Base class for domain incremental learning strategies."""
    def __init__(self, device):
        self.device = device

    def adapt(self, model, old_domains, new_domain):
        return model

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None, bias_reg=None):
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)

        if "texts" in batch and batch["texts"] is not None:
            texts = batch["texts"]
            if isinstance(texts, dict):
                texts = {k: v.to(self.device) for k, v in texts.items()}
                logits = model(images, texts)
            else:
                logits = model(images, {'input_ids': texts.to(self.device)})
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        if ewc:
            loss += ewc.penalty(model)
        if bias_reg:
            loss += bias_reg(model, batch)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels


class StandardDomainIL(DomainIncrementalStrategy):
    """
    Standard domain-incremental learning:
    - Pure cross-entropy classification loss
    - Optional EWC penalty to mitigate forgetting
    - Optional bias/correlation regularizer
    - Supports both dict-style and tuple-style batches
    """
    def adapt_model(self, model, old_domains, new_domain):
        if hasattr(model, "add_domain_head") and callable(getattr(model, "add_domain_head")):
            model.add_domain_head(new_domain)
        return model

    def _move_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: self._move_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._move_device(v) for v in x)
        return x

    def _forward(self, model, inputs):
        if isinstance(inputs, dict):
            if "images" in inputs and "texts" in inputs and inputs["texts"] is not None:
                return model(inputs["images"], inputs["texts"])
            elif "images" in inputs:
                return model(inputs["images"])
            else:
                return model(**inputs)
        return model(inputs)

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None, bias_reg=None):
        if isinstance(batch, tuple) and len(batch) == 3:
            inputs, domain, labels = batch
            labels = self._move_device(labels)
            inputs = self._move_device(inputs)
        elif isinstance(batch, dict):
            labels = self._move_device(batch["labels"])
            if "texts" in batch:
                inputs = {"images": batch["images"], "texts": batch["texts"]}
            else:
                inputs = {"images": batch["images"]}
            inputs = self._move_device(inputs)
            domain = batch.get("domain", None)
        else:
            raise ValueError("Batch format unsupported")

        logits = self._forward(model, inputs)

        loss = criterion(logits, labels)
        if ewc:
            loss += ewc.penalty(model)
        if bias_reg:
            loss += bias_reg(model, {"inputs": inputs, "labels": labels, "domain": domain})

        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels


class DistillationDomainIL(DomainIncrementalStrategy):
    """Domain-IL with knowledge distillation regularizer."""
    def __init__(self, device, temperature=2.0, lambda_distill=1.0):
        super().__init__(device)
        self.temperature = temperature
        self.lambda_distill = lambda_distill

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None, bias_reg=None):
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)

        if "texts" in batch and batch["texts"] is not None:
            texts = {k: v.to(self.device) for k, v in batch["texts"].items()}
            logits = model(images, texts)
            old_logits = old_model(images, texts) if old_model else None
        else:
            logits = model(images)
            old_logits = old_model(images) if old_model else None

        cls_loss = criterion(logits, labels)
        dist_loss = 0.0
        if old_logits is not None:
            old_num_classes = old_logits.size(1)
            with torch.no_grad():
                soft_targets = torch.softmax(old_logits / self.temperature, dim=1)
            soft_preds = F.log_softmax(logits[:, :old_num_classes] / self.temperature, dim=1)
            dist_loss = -torch.sum(soft_targets * soft_preds) / soft_preds.size(0)

        total_loss = cls_loss + self.lambda_distill * dist_loss

        if ewc:
            total_loss += ewc.penalty(model)
        if bias_reg:
            total_loss += bias_reg(model, batch)

        preds = torch.argmax(logits, dim=1)
        return total_loss, preds, labels


# ----------- EWC for Domain-IL -----------

class EWC:
    """Elastic Weight Consolidation for domain incremental learning."""
    def __init__(self, model: nn.Module, dataloader, device: torch.device, lambda_ewc: float = 5000.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}
        self._compute_fisher(dataloader)
        for n, p in self.params.items():
            self._means[n] = p.data.clone()

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.train()
        samples_count = 0
        for batch in dataloader:
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)
            if 'texts' in batch and batch['texts'] is not None:
                text_inputs = {k: v.to(self.device) for k, v in batch['texts'].items()}
                logits = self.model(images, text_inputs)
            else:
                logits = self.model(images)
            log_probs = F.log_softmax(logits, dim=1)
            for i in range(len(labels)):
                self.model.zero_grad()
                log_prob = log_probs[i, labels[i]]
                log_prob.backward(retain_graph=(i < len(labels)-1))
                for n, p in self.params.items():
                    if p.grad is not None:
                        fisher[n] += p.grad.data ** 2
                samples_count += 1
        for n in fisher.keys():
            fisher[n] /= max(samples_count, 1)
        self._fisher = fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if any(classifier_name in n for classifier_name in ['classifier', 'image_classifier', 'text_classifier', 'fusion_classifier']):
                continue
            if n in self._means and p.shape == self._means[n].shape:
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        return self.lambda_ewc * loss
    

# ----------- Exemplar Management -----------

class ExemplarManager:
    def __init__(self, max_exemplars=200, max_per_class=20, selection_strategy="random"):
        self.exemplars = {}  # class_idx -> list of exemplar samples (dict with tensors)
        self.max_exemplars = max_exemplars
        self.max_per_class = max_per_class
        self.selection_strategy = selection_strategy.lower()
        assert self.selection_strategy in ["random", "herding"], "Invalid selection_strategy"

    def add_exemplars(self, class_idx, samples):
        # Replace existing exemplars for that class with new selection
        self.exemplars[class_idx] = samples
        self._balance()

    def _balance(self):
        classes = list(self.exemplars.keys())
        if not classes:
            return
        allowed_per = min(self.max_per_class, self.max_exemplars // len(classes))
        for cl in classes:
            if len(self.exemplars[cl]) > allowed_per:
                self.exemplars[cl] = self.exemplars[cl][:allowed_per]

    def get_replay_batch(self, device, batch_size=32):
        all_samples = sum(self.exemplars.values(), [])
        if len(all_samples) == 0:
            return None
        samples = random.sample(all_samples, min(batch_size, len(all_samples)))

        images = torch.stack([s["images"] for s in samples]).to(device)
        labels = torch.tensor([s["labels"] for s in samples], device=device)

        if "texts" in samples[0]:
            texts_keys = samples[0]["texts"].keys()
            texts = {k: torch.stack([s["texts"][k] for s in samples]).to(device) for k in texts_keys}
        else:
            texts = None

        batch = {"images": images, "labels": labels}
        if texts is not None:
            batch["texts"] = texts
        return batch

    def update_exemplars(self, model, dataset, device):
        model.eval()
        features_per_class = {}
        samples_per_class = {}

        # Extract features per sample
        with torch.no_grad():
            for idx in range(len(dataset)):
                sample = dataset[idx]
                label = sample["label"] if isinstance(sample["label"], int) else sample["label"].item()
                img = sample["image"].unsqueeze(0).to(device)
                txt = sample.get("text")
                if txt is not None:
                    txt = {k: v.unsqueeze(0).to(device) for k, v in txt.items()}
                    feat = model.extract_features(img, txt)
                else:
                    feat = model.extract_features(img)
                feat = feat.cpu().squeeze(0)

                features_per_class.setdefault(label, []).append(feat)
                samples_per_class.setdefault(label, []).append(sample)

        # Select exemplars per class according to strategy
        for cls in features_per_class:
            feats = torch.stack(features_per_class[cls])  # (N_samples, feature_dim)
            samples = samples_per_class[cls]
            K = min(self.max_per_class, len(samples))

            if self.selection_strategy == "random":
                selected_indices = random.sample(range(len(samples)), K)

            elif self.selection_strategy == "herding":
                # Herding: Select exemplars that best approximate class mean in feature space
                class_mean = feats.mean(dim=0)
                selected_indices = []
                selected_features = torch.zeros_like(class_mean).unsqueeze(0)  # cumulative sum

                # Greedy selection to minimize distance to class_mean
                candidates = set(range(len(samples)))
                for _ in range(K):
                    if not candidates:
                        break
                    distances = []
                    for idx in candidates:
                        temp_sum = selected_features.sum(dim=0) + feats[idx]
                        mean_feat = temp_sum / (len(selected_indices) + 1)
                        dist = torch.norm(class_mean - mean_feat).item()
                        distances.append((dist, idx))
                    distances.sort()
                    best_idx = distances[0][1]
                    selected_indices.append(best_idx)
                    selected_features = torch.cat([selected_features, feats[best_idx].unsqueeze(0)], dim=0)
                    candidates.remove(best_idx)

            else:
                raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

            # Add selected exemplars to manager for this class
            selected_samples = [samples[i] for i in selected_indices]
            self.exemplars[cls] = selected_samples

        model.train()



# ----------- Adaptive Learning Rate -----------

class AdaptiveLR:
    """Adaptive LR scheduler for domain incremental learning."""
    def __init__(self, optimizer, base_lr=1e-3, decay_factor=0.7, patience=3, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.patience = patience
        self.factor = decay_factor
        self.min_lr = min_lr
        self.counter = 0
        self.best = None

    def step(self, metrics):
        acc = metrics['accuracy']
        if self.best is None or acc > self.best:
            self.best = acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.base_lr = max(self.base_lr * self.factor, self.min_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr
                self.counter = 0
                return True
        return False

    def get_lr(self):
        return self.base_lr


# ----------- Feature Extraction -----------

def extract_features(model, dataloader, device):
    model.eval()
    features = {}
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["images"].to(device)
            if "texts" in batch and batch["texts"] is not None:
                text_inputs = {k: v.to(device) for k, v in batch["texts"].items()}
                feats = model.extract_features(imgs, text_inputs)
            else:
                feats = model.extract_features(imgs)
            labels = batch["labels"].cpu().numpy()
            for f, l in zip(feats.cpu().numpy(), labels):
                if l not in features:
                    features[l] = []
                features[l].append(f)
    for l in features:
        features[l] = np.stack(features[l])
    return features
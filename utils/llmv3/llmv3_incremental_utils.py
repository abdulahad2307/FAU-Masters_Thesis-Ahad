import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def expand_classifier(model, old_num_classes, new_num_classes, device):
    old_classifier = model.classifier
    old_weight = old_classifier.weight.data
    old_bias = old_classifier.bias.data if old_classifier.bias is not None else None

    in_features = old_weight.shape[1]

    new_classifier = nn.Linear(in_features, new_num_classes).to(device)
    nn.init.normal_(new_classifier.weight, mean=0.0, std=0.02)
    if new_classifier.bias is not None:
        nn.init.zeros_(new_classifier.bias)

    new_classifier.weight.data[:old_num_classes] = old_weight
    if old_bias is not None:
        new_classifier.bias.data[:old_num_classes] = old_bias

    model.classifier = new_classifier

class BiasCorrectionLayer(nn.Module):
    def __init__(self, num_old_classes, num_new_classes):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_old_classes + num_new_classes))

    def forward(self, logits):
        return logits + self.bias

class EWC:
    def __init__(self, model, dataloader, device, fisher_n=500, lambda_ewc=5000):
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.fisher_n = fisher_n
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}
        self.compute_fisher()

    def compute_fisher(self):
        for n, p in self.params.items():
            self._fisher[n] = torch.zeros_like(p, device=self.device)
            self._means[n] = p.clone().detach()

        self.model.eval()
        count = 0
        for batch in self.dataloader:
            if count >= self.fisher_n:
                break
            for k in batch:
                batch[k] = batch[k].to(self.device)
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self._fisher[n] += p.grad.data.clone().pow(2)
            count += 1

        for n in self._fisher:
            self._fisher[n] /= count

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n]).pow(2)).sum()
        return self.lambda_ewc * loss

def distillation_loss(new_logits, old_logits, temperature=2.0, alpha=0.5):
    new_logits_scaled = new_logits / temperature
    old_logits_scaled = old_logits / temperature
    loss = F.kl_div(
        F.log_softmax(new_logits_scaled, dim=1),
        F.softmax(old_logits_scaled, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    return loss * alpha

class ExemplarHandler:
    def __init__(self, max_exemplars_per_class=20, selection_method="random"):
        self.max_exemplars_per_class = max_exemplars_per_class
        self.selection_method = selection_method
        self.exemplars = {}

    def update_exemplars(self, dataset, class_labels, model, device):
        model.eval()
        self.exemplars = {}
        for cls in class_labels:
            features = []
            indices = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    sample = dataset[i]
                    if sample['labels'].item() != cls:
                        continue
                    inputs = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k != 'labels' and k != 'image_id'}
                    feat = model.extract_features(**inputs)
                    features.append(feat.squeeze(0).cpu())
                    indices.append(i)

            features = torch.stack(features)
            class_mean = features.mean(dim=0)
            selected_idxs = []
            exemplar_feats = []

            features_accum = features.clone()

            for _ in range(min(self.max_exemplars_per_class, len(indices))):
                distances = torch.norm(class_mean - features_accum, dim=1)
                min_idx = torch.argmin(distances).item()
                selected_idxs.append(indices[min_idx])
                exemplar_feats.append(features_accum[min_idx])
                class_mean = (class_mean * len(selected_idxs) + features_accum[min_idx]) / (len(selected_idxs) + 1)
                features_accum = torch.cat([features_accum[:min_idx], features_accum[min_idx+1:]], dim=0)
                indices.pop(min_idx)

            self.exemplars[cls] = [dataset[idx] for idx in selected_idxs]

    def get_exemplar_dataset(self):
        all_exemplars = []
        for samples in self.exemplars.values():
            all_exemplars.extend(samples)
        return all_exemplars

def evaluate(model, dataloader, device, all_classes, full_model_acc=None, split_name='Test'):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            total_loss += loss.item() * batch['labels'].size(0)
            pred = outputs.logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(batch['labels'].cpu().tolist())

    total_samples = len(trues)
    avg_loss = total_loss / total_samples
    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='macro', zero_division=0)
    gil = (acc - full_model_acc) / (1 - full_model_acc) if full_model_acc is not None else None

    print(f"{split_name} - Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    if gil is not None:
        print(f"{split_name} - Incremental Learning Gap (G_IL): {gil:.4f}")

    class_acc = []
    for cls in all_classes:
        idxs = [i for i,t in enumerate(trues) if t == cls]
        if not idxs:
            class_acc.append(0)
            continue
        class_correct = sum([preds[i]==trues[i] for i in idxs])
        class_acc.append(class_correct / len(idxs))

    if split_name == "Test":
        print("\nClass-wise Test Accuracy:")
        for clss, cacc in zip(all_classes, class_acc):
            print(f"  {clss}: {cacc:.4f}")

    torch.cuda.empty_cache()
    return avg_loss, acc, p, r, f1, gil, class_acc

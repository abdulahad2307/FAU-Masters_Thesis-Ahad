import torch
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os

class DocFormerEvaluator:
    def __init__(self, model, config, test_loader, class_names):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = torch.device(config.device)
        self.best_model_path = os.path.join(config.output_dir, "best_model.pt")
        
    def evaluate_model(self, output_dir):
        """Main evaluation method - Fixed method name"""
        return self.evaluate()
        
    def evaluate(self):
        """Comprehensive evaluation as per paper metrics"""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {self.best_model_path}")
        else:
            print(f"No checkpoint found at {self.best_model_path}, using current model state")
            
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch, task="finetune")
                
                if 'logits' in outputs:
                    preds = torch.argmax(outputs['logits'], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
                
        if len(all_preds) == 0:
            print("No predictions generated during evaluation")
            return {'accuracy': 0.0, 'f1_macro': 0.0}
            
        return self._compute_metrics(np.array(all_preds), np.array(all_labels))
        
    def _compute_metrics(self, preds, labels):
        """Compute paper-specified metrics"""
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "precision_macro": precision_score(labels, preds, average="macro"),
            "recall_macro": recall_score(labels, preds, average="macro")
        }
        
        # Class-wise metrics
        try:
            cls_report = classification_report(labels, preds, target_names=self.class_names, output_dict=True)
            metrics["class_wise"] = cls_report
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            metrics["class_wise"] = {}
        
        # Confusion matrix
        try:
            self._plot_confusion_matrix(labels, preds)
        except Exception as e:
            print(f"Could not generate confusion matrix: {e}")
        
        # Compare with paper benchmarks
        paper_metrics = self._get_paper_benchmarks()
        metrics["paper_comparison"] = paper_metrics
        
        return metrics
        
    def _plot_confusion_matrix(self, labels, preds):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(12,10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                      xticklabels=self.class_names,
                      yticklabels=self.class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, "confusion_matrix.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
        
    def _get_paper_benchmarks(self):
        """Return paper's reported metrics for comparison"""
        return {
            "rvl_cdip": {"accuracy": 0.9617},
            "funsd": {"f1": 0.8334},
            "cord": {"f1": 0.9633},
            "kleister_nda": {"f1": 0.858}
        }
        
    def generate_report(self, metrics):
        """Generate detailed evaluation report"""
        report = {
            "dataset_stats": {
                "num_samples": len(self.test_loader.dataset),
                "num_classes": len(self.class_names)
            },
            "metrics": metrics,
            "hardware": {
                "device": str(self.device),
                "mixed_precision": self.config.use_amp
            }
        }
        
        try:
            with open(os.path.join(self.config.output_dir, "evaluation_report.json"), "w") as f:
                json.dump(report, f, indent=2)
                
            print(f"Evaluation report saved to {self.config.output_dir}")
        except Exception as e:
            print(f"Could not save evaluation report: {e}")
        
        # Print key comparisons
        print(f"=== Evaluation Results ===")
        print(f"Accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
        print(f"F1 Macro: {metrics.get('f1_macro', 0)*100:.2f}%")
        print(f"Precision: {metrics.get('precision_macro', 0)*100:.2f}%")
        print(f"Recall: {metrics.get('recall_macro', 0)*100:.2f}%")
        
        paper_result = metrics.get("paper_comparison", {}).get("rvl_cdip", {})
        if paper_result:
            print(f"Paper Benchmark: {paper_result.get('accuracy', 0)*100:.2f}%")
            gap = (metrics.get('accuracy', 0) - paper_result.get('accuracy', 0)) * 100
            print(f"Performance Gap: {gap:+.2f}%")

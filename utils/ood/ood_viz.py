import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional

def plot_ood_histograms(y_true, y_pred, scores, savepath: Optional[str] = None, method_name: str = "OOD"):
    """
    Plot histograms for OOD detection results
    
    Args:
        y_true: True labels (1 for known, 0 for unknown)
        y_pred: Predicted labels (1 for known, 0 for unknown)  
        scores: OOD scores
        savepath: Path to save the plot
        method_name: Name of the OOD method for title
    """
    plt.figure(figsize=(15, 5))
    
    # Extract scores for known and unknown classes
    known_mask = y_true == 1
    unknown_mask = y_true == 0
    
    known_scores = scores[known_mask] if np.any(known_mask) else np.array([])
    unknown_scores = scores[unknown_mask] if np.any(unknown_mask) else np.array([])
    
    # Plot 1: Score distributions
    plt.subplot(1, 3, 1)
    if len(known_scores) > 0:
        plt.hist(known_scores, bins=30, alpha=0.7, label=f'Known Classes (n={len(known_scores)})', 
                 color='blue', density=True)
    if len(unknown_scores) > 0:
        plt.hist(unknown_scores, bins=30, alpha=0.7, label=f'Unknown Classes (n={len(unknown_scores)})', 
                 color='red', density=True)
    
    plt.xlabel(f'{method_name} Score')
    plt.ylabel('Density')
    plt.title(f'{method_name} Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: ROC-style plot (if we have both classes)
    plt.subplot(1, 3, 2)
    if len(known_scores) > 0 and len(unknown_scores) > 0:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{method_name} ROC Curve')
        plt.legend(loc="lower right")
    else:
        plt.text(0.5, 0.5, 'ROC curve requires\nboth classes', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('ROC Curve (Insufficient Data)')
    
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    plt.subplot(1, 3, 3)
    from sklearn.metrics import confusion_matrix
    
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Unknown', 'Predicted Known'],
                   yticklabels=['True Unknown', 'True Known'])
        plt.title(f'{method_name} Confusion Matrix')
    else:
        plt.text(0.5, 0.5, 'Confusion matrix\nrequires predictions', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Confusion Matrix (Insufficient Data)')
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"OOD visualization saved to: {savepath}")
    
    plt.show()

def plot_ood_threshold_analysis(y_true, scores, method_name: str = "OOD", savepath: Optional[str] = None):
    """
    Plot threshold analysis for OOD detection
    
    Args:
        y_true: True labels (1 for known, 0 for unknown)
        scores: OOD scores
        method_name: Name of the OOD method
        savepath: Path to save the plot
    """
    if len(np.unique(y_true)) < 2:
        print("Threshold analysis requires both known and unknown samples")
        return
        
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Precision-Recall curve
    plt.subplot(1, 2, 1)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    avg_precision = average_precision_score(y_true, scores)
    
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{method_name} Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Threshold vs Metrics
    plt.subplot(1, 2, 2)
    thresholds_range = np.linspace(scores.min(), scores.max(), 100)
    accuracies = []
    
    for thresh in thresholds_range:
        pred = (scores >= thresh).astype(int)
        acc = np.mean(pred == y_true)
        accuracies.append(acc)
    
    plt.plot(thresholds_range, accuracies, label='Accuracy', color='green', lw=2)
    plt.xlabel(f'{method_name} Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'{method_name} Threshold vs Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Threshold analysis saved to: {savepath}")
    
    plt.show()

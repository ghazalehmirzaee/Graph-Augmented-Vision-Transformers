# src/utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculate and track metrics for multi-label classification"""

    def __init__(self, disease_names):
        self.disease_names = disease_names

    def calculate_metrics(self, targets, predictions, threshold=0.5):
        """Calculate comprehensive metrics for multi-label classification"""
        metrics = {}

        # Convert predictions to binary using threshold
        binary_preds = (predictions > threshold).astype(float)

        # Calculate per-class metrics
        for i, disease in enumerate(self.disease_names):
            if len(np.unique(targets[:, i])) > 1:
                metrics[f'{disease}_auc'] = roc_auc_score(
                    targets[:, i], predictions[:, i]
                )
                metrics[f'{disease}_ap'] = average_precision_score(
                    targets[:, i], predictions[:, i]
                )
                metrics[f'{disease}_f1'] = f1_score(
                    targets[:, i], binary_preds[:, i]
                )

                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(
                    targets[:, i], binary_preds[:, i]
                ).ravel()

                # Sensitivity (Recall) and Specificity
                metrics[f'{disease}_sensitivity'] = tp / (tp + fn)
                metrics[f'{disease}_specificity'] = tn / (tn + fp)
                metrics[f'{disease}_precision'] = tp / (tp + fp)

        # Calculate mean metrics
        metric_types = ['auc', 'ap', 'f1', 'sensitivity', 'specificity', 'precision']
        for metric_type in metric_types:
            values = [metrics[f'{disease}_{metric_type}']
                      for disease in self.disease_names]
            metrics[f'mean_{metric_type}'] = np.mean(values)

        # Calculate exact match ratio
        metrics['exact_match'] = np.mean(
            np.all(binary_preds == targets, axis=1)
        )

        return metrics

    def calculate_confidence_intervals(self, targets, predictions,
                                       n_bootstrap=1000, alpha=0.05):
        """Calculate confidence intervals using bootstrap"""
        n_samples = len(targets)
        bootstrap_metrics = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_metrics.append(
                self.calculate_metrics(
                    targets[indices], predictions[indices]
                )
            )

        # Calculate confidence intervals for each metric
        ci_metrics = {}
        for metric in bootstrap_metrics[0].keys():
            values = [m[metric] for m in bootstrap_metrics]
            ci_lower = np.percentile(values, alpha / 2 * 100)
            ci_upper = np.percentile(values, (1 - alpha / 2) * 100)
            ci_metrics[f'{metric}_ci'] = (ci_lower, ci_upper)

        return ci_metrics

    def plot_metrics(self, metrics_history, save_dir):
        """Plot training metrics history"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot([m['loss'] for m in metrics_history], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(save_dir / 'loss_curve.png')
        plt.close()

        # Plot AUC curves
        plt.figure(figsize=(12, 8))
        for disease in self.disease_names:
            plt.plot([m[f'{disease}_auc'] for m in metrics_history],
                     label=disease)
        plt.xlabel('Epoch')
        plt.ylabel('AUC-ROC')
        plt.title('AUC-ROC per Disease')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'auc_curves.png')
        plt.close()



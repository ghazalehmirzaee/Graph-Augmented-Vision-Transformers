# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import torch
import pandas as pd
from pathlib import Path
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class VisualizationManager:
    def __init__(self, save_dir, disease_names):
        self.save_dir = Path(save_dir)
        self.disease_names = disease_names
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(disease_names)))

        # Create directories
        self.dirs = {
            'roc': self.save_dir / 'roc_curves',
            'pr': self.save_dir / 'pr_curves',
            'confusion': self.save_dir / 'confusion_matrices',
            'attention': self.save_dir / 'attention_maps',
            'error': self.save_dir / 'error_analysis',
            'training': self.save_dir / 'training_progress'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def plot_roc_curves(self, targets, predictions, title_suffix=''):
        """Plot ROC curves for all diseases"""
        plt.figure(figsize=(12, 8))

        for i, disease in enumerate(self.disease_names):
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=self.colors[i],
                     label=f'{disease} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves{title_suffix}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.dirs['roc'] / f'roc_curves{title_suffix}.pdf')
        plt.close()

    def plot_precision_recall_curves(self, targets, predictions, title_suffix=''):
        """Plot precision-recall curves"""
        plt.figure(figsize=(12, 8))

        for i, disease in enumerate(self.disease_names):
            precision, recall, _ = precision_recall_curve(
                targets[:, i], predictions[:, i]
            )
            ap = average_precision_score(targets[:, i], predictions[:, i])

            plt.plot(recall, precision, color=self.colors[i],
                     label=f'{disease} (AP = {ap:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves{title_suffix}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.dirs['pr'] / f'pr_curves{title_suffix}.pdf')
        plt.close()

    def plot_confusion_matrices(self, targets, predictions, threshold=0.5):
        """Plot confusion matrices for each disease"""
        binary_preds = (predictions > threshold).astype(float)

        n_rows = (len(self.disease_names) + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
        axes = axes.flatten()

        for i, (disease, ax) in enumerate(zip(self.disease_names, axes)):
            cm = confusion_matrix(targets[:, i], binary_preds[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(disease)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        for ax in axes[len(self.disease_names):]:
            ax.remove()

        plt.tight_layout()
        plt.savefig(self.dirs['confusion'] / 'confusion_matrices.pdf')
        plt.close()

    def plot_attention_maps(self, model, image_tensor, layer_name):
        """Plot attention maps from transformer"""

        # Get attention weights
        def get_attention(name):
            def hook(model, input, output):
                attention_maps[name] = output.detach()

            return hook

        attention_maps = {}
        hooks = []
        for name, module in model.named_modules():
            if layer_name in name:
                hooks.append(module.register_forward_hook(get_attention(name)))

        # Forward pass
        with torch.no_grad():
            _ = model(image_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Plot attention maps
        for name, attention in attention_maps.items():
            attention = attention.mean(dim=1)  # Average over heads

            plt.figure(figsize=(10, 10))
            sns.heatmap(attention[0].cpu(), cmap='viridis')
            plt.title(f'Attention Map - {name}')
            plt.savefig(self.dirs['attention'] / f'attention_map_{name}.pdf')
            plt.close()

    def plot_grad_cam(self, model, image_tensor, target_layer, class_idx):
        """Plot Grad-CAM visualizations"""
        grad_cam = GradCAM(model=model, target_layer=target_layer)

        # Generate cam mask
        cam_mask = grad_cam(input_tensor=image_tensor, target_category=class_idx)

        # Convert to RGB
        image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Overlay
        visualization = show_cam_on_image(image_np, cam_mask[0])

        plt.figure(figsize=(10, 10))
        plt.imshow(visualization)
        plt.title(f'Grad-CAM - {self.disease_names[class_idx]}')
        plt.axis('off')
        plt.savefig(self.dirs['attention'] / f'grad_cam_{class_idx}.pdf')
        plt.close()

    def plot_training_progress(self, history):
        """Plot training metrics over time"""
        metrics = ['loss', 'auc', 'ap', 'f1']

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(history[f'train_{metric}'], label='Train')
            plt.plot(history[f'val_{metric}'], label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.title(f'Training Progress - {metric.upper()}')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.dirs['training'] / f'{metric}_progress.pdf')
            plt.close()

    def plot_error_analysis(self, error_analysis):
        """Plot error analysis visualizations"""
        # Error rates per disease
        plt.figure(figsize=(12, 6))
        error_rates = [error_analysis[disease]['error_rate']
                       for disease in self.disease_names]
        plt.bar(self.disease_names, error_rates)
        plt.xticks(rotation=45)
        plt.ylabel('Error Rate')
        plt.title('Error Rates by Disease')
        plt.tight_layout()
        plt.savefig(self.dirs['error'] / 'error_rates.pdf')
        plt.close()

        # Error co-occurrence matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(error_analysis['cooccurrence_matrix'],
                    xticklabels=self.disease_names,
                    yticklabels=self.disease_names,
                    cmap='YlOrRd')
        plt.title('Error Co-occurrence Matrix')
        plt.tight_layout()
        plt.savefig(self.dirs['error'] / 'error_cooccurrence.pdf')
        plt.close()

        # False positive vs false negative rates
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.disease_names))
        width = 0.35

        fp_rates = [error_analysis[d]['false_positive_rate'] for d in self.disease_names]
        fn_rates = [error_analysis[d]['false_negative_rate'] for d in self.disease_names]

        plt.bar(x - width / 2, fp_rates, width, label='False Positives')
        plt.bar(x + width / 2, fn_rates, width, label='False Negatives')

        plt.xticks(x, self.disease_names, rotation=45)
        plt.ylabel('Rate')
        plt.title('False Positive vs False Negative Rates by Disease')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dirs['error'] / 'fp_fn_rates.pdf')
        plt.close()

    def create_performance_tables(self, metrics, ci_metrics=None):
        """Create performance summary tables"""
        # Main performance table
        performance_df = pd.DataFrame({
            'Disease': self.disease_names,
            'AUC-ROC': [metrics[f'{d}_auc'] for d in self.disease_names],
            'AP': [metrics[f'{d}_ap'] for d in self.disease_names],
            'F1': [metrics[f'{d}_f1'] for d in self.disease_names],
            'Sensitivity': [metrics[f'{d}_sensitivity'] for d in self.disease_names],
            'Specificity': [metrics[f'{d}_specificity'] for d in self.disease_names]
        })

        if ci_metrics:
            for d in self.disease_names:
                performance_df.loc[performance_df['Disease'] == d, 'AUC-ROC CI'] = \
                    f"({ci_metrics[f'{d}_auc_ci'][0]:.3f}-{ci_metrics[f'{d}_auc_ci'][1]:.3f})"

        # Save as LaTeX and CSV
        performance_df.to_latex(self.save_dir / 'performance_table.tex', index=False)
        performance_df.to_csv(self.save_dir / 'performance_table.csv', index=False)

        return performance_df

    def plot_learning_dynamics(self, history):
        """Plot learning dynamics"""
        # Learning rate schedule
        plt.figure(figsize=(10, 6))
        plt.plot(history['learning_rates'])
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(self.dirs['training'] / 'lr_schedule.pdf')
        plt.close()

        # Loss landscape
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Landscape')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.dirs['training'] / 'loss_landscape.pdf')
        plt.close()

    def visualize_predictions(self, images, targets, predictions,
                              indices=None, max_images=16):
        """Visualize model predictions with Grad-CAM"""
        if indices is None:
            indices = np.random.choice(len(images),
                                       size=min(max_images, len(images)),
                                       replace=False)

        ncols = 4
        nrows = (len(indices) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()

        for idx, ax in zip(indices, axes):
            image = images[idx].squeeze().cpu().numpy()
            target = targets[idx]
            pred = predictions[idx]

            # Show image
            ax.imshow(image, cmap='gray')

            # Add true and predicted labels
            true_diseases = [self.disease_names[i] for i, t in enumerate(target) if t]
            pred_diseases = [self.disease_names[i] for i, p in enumerate(pred) if p > 0.5]

            ax.set_title(f'True: {", ".join(true_diseases)}\n' + \
                         f'Pred: {", ".join(pred_diseases)}',
                         fontsize=8)
            ax.axis('off')

        # Remove empty subplots
        for ax in axes[len(indices):]:
            ax.remove()

        plt.tight_layout()
        plt.savefig(self.dirs['error'] / 'prediction_examples.pdf')
        plt.close()
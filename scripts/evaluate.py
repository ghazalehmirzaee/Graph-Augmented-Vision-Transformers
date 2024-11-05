# scripts/evaluate.py
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd

from src.data.dataset import ChestXrayDataset
from src.data.transforms import ChestXrayTransforms
from src.models.vit import VisionTransformer
from src.utils.metrics import MetricCalculator
from torch.utils.data import DataLoader


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    model = VisionTransformer(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_chans=config['model']['in_chans'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        drop_rate=0.0  # No dropout during evaluation
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)


def plot_roc_curves(predictions, targets, disease_names, save_dir):
    """Plot ROC curves for each disease"""
    plt.figure(figsize=(15, 10))

    for i, disease in enumerate(disease_names):
        fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{disease} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Diseases')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(predictions, targets, disease_names, save_dir):
    """Plot confusion matrices for each disease"""
    binary_preds = (predictions > 0.5).astype(np.int64)

    n_rows = (len(disease_names) + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, (disease, ax) in enumerate(zip(disease_names, axes)):
        cm = confusion_matrix(targets[:, i], binary_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(disease)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # Remove empty subplots
    for ax in axes[len(disease_names):]:
        ax.remove()

    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_error_patterns(predictions, targets, disease_names):
    """Analyze error patterns and disease co-occurrence"""
    binary_preds = (predictions > 0.5).astype(np.int64)
    errors = binary_preds != targets

    # Calculate error rates per disease
    error_rates = errors.mean(axis=0)
    error_df = pd.DataFrame({
        'Disease': disease_names,
        'Error Rate': error_rates
    })

    # Analyze co-occurrence of errors
    error_cooccurrence = np.zeros((len(disease_names), len(disease_names)))
    for i in range(len(disease_names)):
        for j in range(len(disease_names)):
            error_cooccurrence[i, j] = np.mean(errors[:, i] & errors[:, j])

    return error_df, error_cooccurrence


def evaluate(model, dataloader, device, save_dir):
    """Evaluate model and generate comprehensive analysis"""
    model.eval()
    metric_calculator = MetricCalculator(dataloader.dataset.disease_names)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Calculate metrics
    metrics = metric_calculator.calculate_metrics(targets, predictions)

    # Calculate confidence intervals
    ci_metrics = metric_calculator.calculate_confidence_intervals(
        targets, predictions
    )

    # Generate visualizations
    plot_roc_curves(predictions, targets,
                    dataloader.dataset.disease_names, save_dir)
    plot_confusion_matrices(predictions, targets,
                            dataloader.dataset.disease_names, save_dir)

    # Analyze errors
    error_df, error_cooccurrence = analyze_error_patterns(
        predictions, targets, dataloader.dataset.disease_names
    )

    # Save results
    results = {
        'metrics': metrics,
        'confidence_intervals': ci_metrics,
        'error_analysis': {
            'per_disease_errors': error_df.to_dict(),
            'error_cooccurrence': error_cooccurrence.tolist()
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', choices=['nih', 'chexpert'], required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['paths']['save_dir']) / 'evaluation'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Create dataset
    if args.dataset == 'nih':
        dataset = ChestXrayDataset(
            image_dir=config['data']['val_dir'],
            label_file=config['data']['val_labels'],
            transform=ChestXrayTransforms.get_val_transforms(config)
        )
    else:  # chexpert
        dataset = ChestXrayDataset(
            image_dir=config['data']['chexpert_dir'],
            label_file=config['data']['chexpert_labels'],
            transform=ChestXrayTransforms.get_val_transforms(config)
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    # Evaluate
    results = evaluate(model, dataloader, device, save_dir)

    # Save results
    import json
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()

    
# train.py
import os
import yaml
import torch
import wandb
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.models.vit import VisionTransformer
from src.training.trainer import Trainer
from src.utils.logging import setup_logging, log_system_info, log_dataset_info
from src.utils.metrics import MetricCalculator
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision Transformer for Chest X-ray Classification')
    parser.add_argument('--config', type=str, default='configs/baseline_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def create_transforms(config):
    """Create data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def setup_wandb(config):
    """Initialize Weights & Biases"""
    run = wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        entity=config['wandb']['entity'],
        config=config,
        reinit=True
    )
    return run


def create_dataloaders(config, train_transform, val_transform):
    """Create training and validation dataloaders"""
    train_dataset = ChestXrayDataset(
        image_dir=config['data']['train_dir'],
        label_file=config['data']['train_labels'],
        transform=train_transform
    )

    val_dataset = ChestXrayDataset(
        image_dir=config['data']['val_dir'],
        label_file=config['data']['val_labels'],
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset


def create_model(config, device):
    """Create and initialize the model"""
    model = VisionTransformer(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_chans=config['model']['in_chans'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        drop_rate=config['model']['drop_rate']
    )

    # Load pre-trained weights
    if os.path.exists(config['model']['pretrained_path']):
        model.load_mae_weights(config['model']['pretrained_path'])

    return model.to(device)


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup directories
    for dir_name in ['save_dir', 'log_dir']:
        Path(config['paths'][dir_name]).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(config)
    log_system_info()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        # Initialize wandb
        run = setup_wandb(config)

        # Create transforms and dataloaders
        train_transform, val_transform = create_transforms(config)
        train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
            config, train_transform, val_transform
        )

        # Log dataset information
        log_dataset_info(train_dataset, val_dataset)

        # Create model
        model = create_model(config, device)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
            else:
                logger.error(f"No checkpoint found at {args.resume}")

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )

        # Train model
        final_metrics = trainer.train()

        # Log final results
        logger.info("Training completed. Final metrics:")
        for metric_name, value in final_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

        # Create final visualizations
        metric_calculator = MetricCalculator(train_dataset.disease_names)
        metric_calculator.plot_metrics(
            trainer.train_metrics_history,
            save_dir=os.path.join(config['paths']['save_dir'], 'figures')
        )

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    finally:
        # Clean up
        wandb.finish()
        logger.info("Training script completed")


if __name__ == '__main__':
    main()


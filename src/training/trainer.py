# src/training/trainer.py
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import numpy as np
import logging
import os
from datetime import datetime
from ..utils.metrics import MetricCalculator
from .losses import DynamicWeightedLoss
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.current_epoch = 0

        # Parse numerical parameters
        try:
            self.learning_rate = float(config['training']['learning_rate'])
            self.weight_decay = float(config['training']['weight_decay'])
            self.beta1 = float(config['optimizer']['beta1'])
            self.beta2 = float(config['optimizer']['beta2'])
            self.eps = float(config['optimizer']['eps'])
            self.max_grad_norm = float(config['training'].get('max_grad_norm', 1.0))
        except ValueError as e:
            raise ValueError(f"Error converting optimizer parameters to float: {e}")

        # Initialize metrics calculator
        self.metric_calculator = MetricCalculator(train_loader.dataset.disease_names)

        # Setup combined loss with class weights
        class_weights = train_loader.dataset.class_weights.to(device)
        self.criterion = DynamicWeightedLoss(
            num_classes=config['model']['num_classes'],
            class_weights=class_weights
        ).to(device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.eps
        )

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # Training monitoring
        self.best_val_auc = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stop_patience = config['training']['early_stopping_patience']
        self.min_delta = 1e-4

        # Initialize history
        self.train_metrics_history = []
        self.val_metrics_history = []

    def _setup_scheduler(self):
        warmup_steps = len(self.train_loader) * self.config['training']['warmup_epochs']
        total_steps = len(self.train_loader) * self.config['training']['epochs']

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self):
        self.model.train()
        epoch_predictions = []
        epoch_targets = []
        epoch_losses = []
        component_losses = {'wbce': [], 'focal': [], 'asl': []}

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(images)
                loss, loss_components = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            self.scheduler.step()

            # Store predictions and losses
            with torch.no_grad():
                epoch_predictions.append(torch.sigmoid(outputs).cpu().numpy())
                epoch_targets.append(targets.cpu().numpy())
                epoch_losses.append(loss.item())

                # Store component losses
                for k, v in loss_components.items():
                    component_losses[k].append(v.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

        # Calculate metrics
        predictions = np.vstack(epoch_predictions)
        targets = np.vstack(epoch_targets)
        metrics = self.metric_calculator.calculate_metrics(targets, predictions)

        # Add losses to metrics
        metrics['loss'] = np.mean(epoch_losses)
        for k, v in component_losses.items():
            metrics[f'loss_{k}'] = np.mean(v)

        return metrics

    def validate(self):
        self.model.eval()
        val_predictions = []
        val_targets = []
        val_losses = []

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    loss, _ = self.criterion(outputs, targets)

                val_predictions.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_losses.append(loss.item())

        predictions = np.vstack(val_predictions)
        targets = np.vstack(val_targets)
        metrics = self.metric_calculator.calculate_metrics(targets, predictions)
        metrics['loss'] = np.mean(val_losses)

        return metrics

    def save_checkpoint(self, metrics, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'metrics': metrics,
            'config': self.config
        }

        checkpoint_dir = os.path.join(self.config['paths']['save_dir'])
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save latest checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch}_auc_{metrics["mean_auc"]:.4f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model with AUC: {metrics['mean_auc']:.4f}")

    def train(self):
        logger.info("Starting training...")

        try:
            for epoch in range(self.config['training']['epochs']):
                self.current_epoch = epoch

                # Train epoch
                train_metrics = self.train_epoch()
                self.train_metrics_history.append(train_metrics)

                # Validate
                val_metrics = self.validate()
                self.val_metrics_history.append(val_metrics)

                # Log metrics
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/mean_auc': train_metrics['mean_auc'],
                    'val/loss': val_metrics['loss'],
                    'val/mean_auc': val_metrics['mean_auc'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })

                # Log per-disease metrics
                for disease in self.train_loader.dataset.disease_names:
                    wandb.log({
                        f'train/auc_{disease}': train_metrics[f'{disease}_auc'],
                        f'val/auc_{disease}': val_metrics[f'{disease}_auc']
                    })

                # Print epoch summary
                logger.info(
                    f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train AUC: {train_metrics['mean_auc']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val AUC: {val_metrics['mean_auc']:.4f}"
                )

                # Save checkpoint if best model
                if val_metrics['mean_auc'] > self.best_val_auc:
                    self.best_val_auc = val_metrics['mean_auc']
                    self.patience_counter = 0
                    self.save_checkpoint(val_metrics, is_best=True)
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.early_stop_patience:
                    logger.info("Early stopping triggered")
                    break

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise

        finally:
            logger.info("Training completed!")

        return {
            'best_val_auc': float(self.best_val_auc),
            'final_train_loss': float(self.train_metrics_history[-1]['loss']),
            'final_train_auc': float(self.train_metrics_history[-1]['mean_auc']),
            'final_val_loss': float(self.val_metrics_history[-1]['loss']),
            'final_val_auc': float(self.val_metrics_history[-1]['mean_auc'])
        }


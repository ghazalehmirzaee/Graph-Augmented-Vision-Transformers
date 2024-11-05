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
from torch.cuda.amp import autocast, GradScaler
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
        self.metric_calculator = MetricCalculator(
            train_loader.dataset.disease_names
        )

        # Setup combined loss with dynamic weights
        class_weights = train_loader.dataset.class_weights.to(device)
        self.criterion = DynamicWeightedLoss(
            num_classes=config['model']['num_classes'],
            class_weights=class_weights
        ).to(device)

        # Optimizer setup
        self.optimizer = self._setup_optimizer()

        # Learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Mixed precision training
        self.scaler = GradScaler('cuda')

        # Enable gradient checkpointing if configured
        if config['model'].get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Training monitoring
        self.best_val_auc = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stop_patience = config['training']['early_stopping_patience']
        self.min_delta = 1e-4

        # Initialize comprehensive history tracking
        self.history = {
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'loss_weights': [],
            'grad_norms': [],
            'per_class_metrics': {
                disease: {'train': [], 'val': []}
                for disease in train_loader.dataset.disease_names
            }
        }

    def _setup_optimizer(self):
        """Setup optimizer with parameter groups"""
        # Separate parameter groups for different learning rates
        decoder_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'head' in name:
                decoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': other_params},
            {'params': decoder_params, 'lr': self.learning_rate * 10}  # Higher LR for decoder
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.eps
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        warmup_steps = len(self.train_loader) * self.config['training']['warmup_epochs']
        total_steps = len(self.train_loader) * self.config['training']['epochs']

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _log_gradients(self):
        """Log gradient norms"""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.history['grad_norms'].append(total_norm)
        wandb.log({'grad_norm': total_norm})

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_predictions = []
        epoch_targets = []
        epoch_losses = {
            'total': [], 'wbce': [], 'focal': [], 'asl': []
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Mixed precision training
            with autocast('cuda'):
                outputs = self.model(images)
                loss, loss_components = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step
            self.scheduler.step()

            # Store predictions and losses
            epoch_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
            epoch_targets.append(targets.cpu().numpy())
            epoch_losses['total'].append(loss.item())
            for k, v in loss_components.items():
                epoch_losses[k].append(v.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

            # Log training progress
            if batch_idx % 100 == 0:
                self._log_gradients()
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/grad_norm': grad_norm,
                    **{f'train/loss_{k}': v[-1] for k, v in epoch_losses.items()}
                })

        # Calculate epoch metrics
        predictions = np.vstack(epoch_predictions)
        targets = np.vstack(epoch_targets)
        metrics = self.metric_calculator.calculate_metrics(targets, predictions)
        metrics.update({
            f'loss_{k}': np.mean(v) for k, v in epoch_losses.items()
        })

        return metrics

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_predictions = []
        val_targets = []
        val_losses = []

        val_pbar = tqdm(self.val_loader, desc='Validation')
        with torch.no_grad():
            for images, targets in val_pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss, _ = self.criterion(outputs, targets)

                val_predictions.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_losses.append(loss.item())

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate metrics
        predictions = np.vstack(val_predictions)
        targets = np.vstack(val_targets)
        metrics = self.metric_calculator.calculate_metrics(targets, predictions)
        metrics['loss'] = np.mean(val_losses)

        return metrics

    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'best_val_auc': self.best_val_auc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'metrics': metrics,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_dir = os.path.join(self.config['paths']['save_dir'])
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch}_auc_{metrics["mean_auc"]:.4f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model with AUC: {metrics['mean_auc']:.4f}")

        # Save periodic checkpoints
        if self.current_epoch % self.config['training']['save_freq'] == 0:
            periodic_path = os.path.join(
                checkpoint_dir,
                f'periodic_checkpoint_epoch_{self.current_epoch}.pt'
            )
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']

        logger.info(f"Resumed from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        try:
            for epoch in range(self.current_epoch, self.config['training']['epochs']):
                self.current_epoch = epoch

                # Train epoch
                train_metrics = self.train_epoch()
                self.history['train_metrics'].append(train_metrics)

                # Validate
                val_metrics = self.validate()
                self.history['val_metrics'].append(val_metrics)

                # Update history
                self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
                self.history['loss_weights'].append(
                    self.criterion.get_loss_weights()
                )

                # Log metrics
                self._log_epoch_metrics(train_metrics, val_metrics)

                # Check early stopping
                stop_training, is_improved = self._check_early_stopping(val_metrics)

                if is_improved:
                    self.save_checkpoint(val_metrics, is_best=True)

                if stop_training:
                    logger.info("Early stopping triggered")
                    break

                # Regular checkpoint saving
                if (epoch + 1) % self.config['training']['save_freq'] == 0:
                    self.save_checkpoint(val_metrics)

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise

        finally:
            logger.info("Training completed!")

        return self._get_final_metrics()

    def _log_epoch_metrics(self, train_metrics, val_metrics):
        """Log metrics for current epoch"""
        metrics = {
            'epoch': self.current_epoch,
            'train/loss': train_metrics['loss'],
            'train/mean_auc': train_metrics['mean_auc'],
            'val/loss': val_metrics['loss'],
            'val/mean_auc': val_metrics['mean_auc']
        }

        # Log per-disease metrics
        for disease in self.train_loader.dataset.disease_names:
            metrics.update({
                f'train/auc_{disease}': train_metrics[f'{disease}_auc'],
                f'val/auc_{disease}': val_metrics[f'{disease}_auc']
            })

        # Log loss components if available
        for k in ['wbce', 'focal', 'asl']:
            if f'loss_{k}' in train_metrics:
                metrics[f'train/loss_{k}'] = train_metrics[f'loss_{k}']

        wandb.log(metrics)

        # Log to console
        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.config['training']['epochs']} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train AUC: {train_metrics['mean_auc']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val AUC: {val_metrics['mean_auc']:.4f}"
        )

    def _get_final_metrics(self):
        """Get final metrics in a simple format"""
        return {
            'best_val_auc': float(self.best_val_auc),
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': self.best_epoch,
            'final_train_loss': float(self.history['train_metrics'][-1]['loss']),
            'final_train_auc': float(self.history['train_metrics'][-1]['mean_auc']),
            'final_val_loss': float(self.history['val_metrics'][-1]['loss']),
            'final_val_auc': float(self.history['val_metrics'][-1]['mean_auc'])
        }



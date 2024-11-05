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

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Initialize metrics calculator
        self.metric_calculator = MetricCalculator(
            train_loader.dataset.disease_names
        )

        # Setup criterion with class weights
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=train_loader.dataset.class_weights.to(device)
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['optimizer']['beta1'],
                   config['optimizer']['beta2']),
            eps=config['optimizer']['eps']
        )

        # Setup scheduler with warmup
        self.scheduler = self.get_scheduler()

        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize tracking variables
        self.best_val_auc = 0
        self.patience_counter = 0
        self.current_epoch = 0

        # Initialize metrics history
        self.train_metrics_history = []
        self.val_metrics_history = []

    def get_scheduler(self):
        """Create learning rate scheduler with warmup"""
        warmup_steps = len(self.train_loader) * self.config['training']['warmup_epochs']
        total_steps = len(self.train_loader) * self.config['training']['epochs']

        def lr_lambda(step):
            # Linear warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_predictions = []
        epoch_targets = []
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            self.scheduler.step()

            # Store predictions and targets
            epoch_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
            epoch_targets.append(targets.cpu().numpy())
            epoch_losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

            # Log to wandb
            if batch_idx % 100 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                })

        # Calculate epoch metrics
        predictions = np.vstack(epoch_predictions)
        targets = np.vstack(epoch_targets)
        metrics = self.metric_calculator.calculate_metrics(targets, predictions)
        metrics['loss'] = np.mean(epoch_losses)

        return metrics

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_predictions = []
        val_targets = []
        val_losses = []

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                val_predictions.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_losses.append(loss.item())

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
            'best_val_auc': self.best_val_auc,
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

        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model with AUC: {metrics['mean_auc']:.4f}")

    def train(self):
        """Main training loop"""
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
                metrics = {
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/mean_auc': train_metrics['mean_auc'],
                    'val/loss': val_metrics['loss'],
                    'val/mean_auc': val_metrics['mean_auc']
                }
                wandb.log(metrics)

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
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info("Early stopping triggered")
                    break

                # Regular checkpoint saving
                if (epoch + 1) % self.config['training']['save_freq'] == 0:
                    self.save_checkpoint(val_metrics)

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise

        finally:
            # Save final results
            final_metrics = {
                'best_val_auc': self.best_val_auc,
                'final_train_metrics': self.train_metrics_history[-1],
                'final_val_metrics': self.val_metrics_history[-1]
            }
            wandb.log({"final_metrics": final_metrics})

        logger.info("Training completed!")
        return final_metrics


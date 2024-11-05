# src/utils/logging.py
import logging
import sys
from pathlib import Path
import yaml
from datetime import datetime
import torch



def setup_logging(config):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create handlers
    file_handler = logging.FileHandler(
        log_dir / f'training_{timestamp}.log'
    )
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Get root logger
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Log config
    logger = logging.getLogger(__name__)
    logger.info("Starting new training run")
    logger.info(f"Config:\n{yaml.dump(config)}")

    return logger


def log_system_info():
    """Log system and environment information"""
    logger = logging.getLogger(__name__)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"CUDA Device {i}: {props.name}")
            logger.info(f"Memory: {props.total_memory / 1024 ** 3:.2f} GB")
    else:
        logger.warning("No CUDA devices available, using CPU")


def log_dataset_info(train_dataset, val_dataset):
    """Log dataset information"""
    logger = logging.getLogger(__name__)

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    # Log class distribution
    for dataset, name in [(train_dataset, 'Training'), (val_dataset, 'Validation')]:
        pos_counts = np.sum(dataset.labels, axis=0)
        logger.info(f"\n{name} set class distribution:")
        for disease, count in zip(dataset.disease_names, pos_counts):
            percentage = (count / len(dataset)) * 100
            logger.info(f"{disease}: {count} ({percentage:.2f}%)")


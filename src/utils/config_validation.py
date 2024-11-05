# utils/config_validation.py

def validate_config(config):
    """Validate configuration parameters and convert to appropriate types."""
    try:
        # Training parameters
        config['training']['learning_rate'] = float(config['training']['learning_rate'])
        config['training']['weight_decay'] = float(config['training']['weight_decay'])
        config['training']['epochs'] = int(config['training']['epochs'])
        config['training']['warmup_epochs'] = int(config['training']['warmup_epochs'])
        config['training']['eval_freq'] = int(config['training']['eval_freq'])
        config['training']['save_freq'] = int(config['training']['save_freq'])
        config['training']['early_stopping_patience'] = int(config['training']['early_stopping_patience'])

        # Optimizer parameters
        config['optimizer']['beta1'] = float(config['optimizer']['beta1'])
        config['optimizer']['beta2'] = float(config['optimizer']['beta2'])
        config['optimizer']['eps'] = float(config['optimizer']['eps'])

        # Model parameters
        config['model']['mlp_ratio'] = float(config['model']['mlp_ratio'])
        config['model']['drop_rate'] = float(config['model']['drop_rate'])

        return config
    except (ValueError, KeyError) as e:
        raise ValueError(f"Error in configuration: {str(e)}")


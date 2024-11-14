# test_checkpoint.py
import os
import torch
import numpy as np
import io


def print_debug(msg):
    print(f"[DEBUG] {msg}")


def inspect_tensor_data(data):
    """Inspect tensor data structure"""
    if isinstance(data, dict):
        print_debug("Dictionary found:")
        for key, value in data.items():
            print_debug(f"  Key: {key}")
            if isinstance(value, dict):
                print_debug("  Nested dictionary:")
                for k, v in value.items():
                    print_debug(f"    {k}: {type(v)}")
            else:
                print_debug(f"  Value type: {type(value)}")
    else:
        print_debug(f"Data type: {type(data)}")


def main():
    checkpoint_path = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/best_model.pt'
    print_debug(f"Testing checkpoint: {checkpoint_path}")

    # Try loading with bytes buffer
    try:
        print_debug("Attempting to load with bytes buffer...")
        with open(checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            data = torch.load(buffer, map_location='cpu')
            print_debug("Successfully loaded checkpoint!")
            inspect_tensor_data(data)
            return True
    except Exception as e:
        print_debug(f"Bytes buffer loading failed: {str(e)}")

    return False


if __name__ == '__main__':
    main()


# test_checkpoint.py
import os
import torch
import numpy as np
import io
import pickle
import sys
from datetime import datetime


def print_debug(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            if name == 'scalar':
                return float
            module = 'numpy'
        if module.startswith('numpy._core'):
            module = 'numpy'
        return super().find_class(module, name)


def test_checkpoint(path):
    print_debug(f"Testing checkpoint: {path}")

    # Method 1: Custom unpickler
    try:
        print_debug("Testing custom unpickler...")
        with open(path, 'rb') as f:
            data = CustomUnpickler(f).load()
        print_debug("Custom unpickler successful!")
        if isinstance(data, dict):
            print_debug(f"Keys found: {list(data.keys())}")
        return True
    except Exception as e:
        print_debug(f"Custom unpickler failed: {str(e)}")

    # Method 2: Memory mapping
    try:
        print_debug("Testing memory-mapped loading...")
        with open(path, 'rb') as f:
            f.seek(2)  # Skip magic number and protocol
            data = torch.load(f, map_location='cpu', weights_only=True)
        print_debug("Memory-mapped loading successful!")
        return True
    except Exception as e:
        print_debug(f"Memory-mapped loading failed: {str(e)}")

    # Method 3: Raw reading
    try:
        print_debug("Testing raw file reading...")
        with open(path, 'rb') as f:
            content = f.read()
        buffer = io.BytesIO(content)
        data = CustomUnpickler(buffer).load()
        print_debug("Raw file reading successful!")
        return True
    except Exception as e:
        print_debug(f"Raw file reading failed: {str(e)}")

    return False


if __name__ == '__main__':
    checkpoint_path = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/best_model.pt'
    test_checkpoint(checkpoint_path)


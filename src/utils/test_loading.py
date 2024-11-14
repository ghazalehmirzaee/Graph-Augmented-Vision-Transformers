# test_loading.py
import torch
import pickle
import numpy as np


def print_status(msg):
    print(f"[DEBUG] {msg}")


def test_checkpoint(path):
    print_status(f"Testing checkpoint: {path}")

    # Try direct loading
    try:
        print_status("Attempting direct load...")
        data = torch.load(path, map_location='cpu', weights_only=False)
        print_status("Direct load successful!")
        if isinstance(data, dict):
            print_status(f"Checkpoint keys: {list(data.keys())}")
        return True
    except Exception as e:
        print_status(f"Direct load failed: {str(e)}")

    # Try custom unpickler
    try:
        print_status("Attempting custom unpickler...")

        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "numpy._core.multiarray" and name == "scalar":
                    return np.float32
                return super().find_class(module, name)

        with open(path, 'rb') as f:
            data = CustomUnpickler(f).load()
        print_status("Custom unpickler successful!")
        return True
    except Exception as e:
        print_status(f"Custom unpickler failed: {str(e)}")

    return False


if __name__ == '__main__':
    checkpoint_path = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/best_model.pt'
    test_checkpoint(checkpoint_path)
    
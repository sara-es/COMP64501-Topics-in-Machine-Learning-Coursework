"""
Main model calls for this assignment. DO NOT MODIFY THIS FILE.
Your container must run this script and output results to stdout.
This script will be replaced during marking, so any changes you make will not be saved! 

This script:
1. Loads your model and counts parameters
2. Performs a training sanity check (1 iteration on 3 samples)
3. Evaluates on the Fashion-MNIST test set
"""
import os
import torch
import torch.nn as nn
import torchvision

import utils
from submission.fashion_training import get_transforms
from submission.fashion_model import Net
from submission.STUDENT_ID import STUDENT_ID


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Returns:
        float: Accuracy on test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def count_parameters(model):
    """Count all parameters (trainable or not) and buffers, except BatchNorm running statistics."""
    bn_stats = ('running_mean', 'running_var', 'num_batches_tracked')
    n = sum(p.numel() for p in model.parameters())
    return n + sum(b.numel() for name, b in model.named_buffers() if not name.endswith(bn_stats))


def main():
    # Check STUDENT_ID is set
    if STUDENT_ID == "your_student_id_here":
        print(f"STUDENT_ID: UNKNOWN")
        print(f"ERROR: STUDENT ID not set, please set it in submission/STUDENT_ID.py first.")
        return
    if not os.path.exists('submission/model_weights.pth'):
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: I couldn't find the model weights at submission/model_weights.pth! Did you forget to train your model first?")
        return
    if os.path.getsize('submission/model_weights.pth') == 0:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: The model weights file at submission/model_weights.pth is empty. Check that your model was trained and saved correctly.")
        return
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the model
    model = Net()
    
    # Count parameters BEFORE loading weights
    num_params = count_parameters(model)
    
    # Load trained weights once and keep them in memory, in case the checks below overwrite the file
    try:
        weights = torch.load('submission/model_weights.pth', weights_only=True, map_location=device)
        model.load_state_dict(weights)
    except Exception as e:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: Failed to load model weights: {e}")
        print(f"PARAMETERS: {num_params}")
        return
    
    model.to(device)
    
    # Verify parameter count after loading (should be same)
    num_params_after = count_parameters(model)
    if num_params != num_params_after:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: Parameter count changed after loading weights ({num_params} -> {num_params_after})")
        print(f"PARAMETERS: {num_params}")
        return
    
    # Define training transforms and check validity
    transforms_train = get_transforms(mode='train')
    passed, error = utils.check_valid_transforms(transforms_train, strict=False) # any torchvision transform is fine for training
    if not passed:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: {error}")
        print(f"PARAMETERS: {num_params}")
        return
    
    # Model architecture check
    passed, error = utils.model_check(model, device, transforms=transforms_train)
    if not passed:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: {error}")
        print(f"PARAMETERS: {num_params}")
        return
    
    # Training function check
    passed, error = utils.training_check(Net, device, transforms=transforms_train)
    if not passed:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: {error}")
        print(f"PARAMETERS: {num_params}")
        return
    
    # Reload weights after sanity check (in case model state changed)
    model.load_state_dict(weights)
    model.to(device)

    # Define evaluation transforms and check validity and determinism
    transform = get_transforms(mode='eval')
    passed, error = utils.check_valid_transforms(transform)
    if not passed:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR in eval transforms: {error}")
        print(f"PARAMETERS: {num_params}")
        return
    passed, error = utils.test_transform_determinism(transform)
    if not passed:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR in eval transforms: {error}")
        print(f"PARAMETERS: {num_params}")
        return

    # Load Fashion-MNIST test set
    # NOTE: during marking, dataset and dataloader may change! 
    # The results you get here may not be the same as during marking!
    # However the format will be the same - if this works, so will the marking setup.
    try:
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    except OSError as e:
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"PARAMETERS: {num_params}")
        print(f"TRAINING_CHECK: PASSED")
        print(f"ERROR: Failed to download Fashion-MNIST test dataset: {e}")
        print(f"       Have you built Docker as read-only?")
        return
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['read-only', 'network', 'connection', 'download', 'name resolution']):
            print(f"STUDENT_ID: {STUDENT_ID}")
            print(f"PARAMETERS: {num_params}")
            print(f"TRAINING_CHECK: PASSED")
            print(f"Well done! Your model passed the training checks.")
            print(f"   To check that inference works, we need to give Docker network access to download the test dataset.")
            print(f"   Please run this again with network access:")
            print(f"       docker run test")
            return
        else:
            print(f"STUDENT_ID: {STUDENT_ID}")
            print(f"PARAMETERS: {num_params}")
            print(f"TRAINING_CHECK: PASSED")
            raise
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    # Evaluate model
    accuracy = evaluate_model(model, test_loader, device)
    
    # Output in standardized format
    print(f"STUDENT_ID: {STUDENT_ID}")
    print(f"ACCURACY_CHECK: {'PASSED' if accuracy >= 0.88 else 'FAILED'} (88% on the public test set)")
    print(f"PARAMETERS: {num_params}")
    print(f"TRAINING_CHECK: PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e: # make sure the parser always gets a result
        print(f"STUDENT_ID: {STUDENT_ID}")
        print(f"ERROR: {type(e).__name__}: {e}")
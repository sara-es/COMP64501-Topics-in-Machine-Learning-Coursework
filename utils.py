"""
Utility functions for the assignment. DO NOT MODIFY THIS FILE.
This script will be replaced during marking, so any changes you make will not be saved! 
"""
import torch, torchvision
from torch import nn
from PIL import Image

from submission.fashion_training import train_fashion_model, get_transforms


def model_check(model, device, transforms=None):
    """
    Small script that checks if the model as defined can perform a single
    training step on dummy data. This script will be used during marking.
    
    This checks:
    - Forward pass works
    - Backward pass works
    - Model has trainable parameters
    
    Returns:
        bool: True if sanity check passes, False otherwise
        str: Error message if check fails, None otherwise
    """
    try:
        model.train()
        
        # Create dummy dataset (3 samples)
        dummy_data = torch.rand(3, 1, 28, 28).to(device)
        dummy_labels = torch.tensor([0, 1, 2]).to(device)

        if transforms is not None:
            # Apply transforms to each sample
            transformed_data = []
            for i in range(dummy_data.shape[0]):
                # Convert to PIL for transforms that expect it
                img = torchvision.transforms.ToPILImage()(dummy_data[i].cpu())
                img = transforms(img)
                transformed_data.append(img)
            dummy_data = torch.stack(transformed_data).to(device)
        
        # Forward pass
        outputs = model(dummy_data)
        
        # Check output shape
        if outputs.shape != (3, 10):
            return False, f"Expected output shape (3, 10), got {outputs.shape}"
        
        # Note this does not need to be the same as loss used in training,
        # just using CrossEntropyLoss as a standard check
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, dummy_labels)
        if torch.isnan(loss) or torch.isinf(loss):
            return False, f"Loss is {loss.item()}, expected finite value"
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for at least some parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if not has_grad:
            return False, "No gradients computed during backward pass"
        
        return True, None
        
    except Exception as e:
        return False, f"Model check failed: {str(e)}"
    

def training_check(model_class, device, transforms=None):
    """
    Verify train_fashion_model() runs and returns valid state_dict.
    Uses small dummy dataset (30 samples) and 1 epoch.
    
    This checks:
    - train_fashion_model() can run without errors
    - Returns a valid state_dict (dict type)
    - Returned state_dict can be loaded into model
    - Training code matches submitted architecture
    
    Returns:
        bool: True if integrity check passes, False otherwise
        str: Error message if check fails, None otherwise
    """
    try:
        # Create small dummy data (30 samples, 28x28 grayscale, 10 classes)
        dummy_images = torch.randint(0, 256, (30, 28, 28), dtype=torch.uint8)
        dummy_labels = torch.arange(30) % 10
        
        # Create a mock dataset that behaves like FashionMNIST
        class DummyDataset:
            def __init__(self, images, labels):
                self.data = images
                self.targets = labels
                self.transform = None # raw images: train_fashion_model adds its own transforms
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                image = Image.fromarray(self.data[idx].numpy())  # PIL image, as in FashionMNIST
                label = self.targets[idx]
                if self.transform:
                    image = self.transform(image)
                return image, label
        
        dummy_dataset = DummyDataset(dummy_images, dummy_labels)
        
        # USE_GPU=False to ensure consistent behavior
        result = train_fashion_model(dummy_dataset, n_epochs=1, USE_GPU=False, return_state_dict_only=True)
        
        # Verify return type is dict (state_dict)
        if not isinstance(result, dict):
            return False, f"train_fashion_model() returned {type(result).__name__}, should be dict (state_dict)"
        
        # Verify state_dict is not empty
        if not result:
            return False, "train_fashion_model() returned empty state_dict"
        
        # Verify we can load the state_dict into a fresh model
        test_model = model_class()
        test_model.load_state_dict(result)
        test_model.to(device)
        
        # Verify the loaded model can do a forward pass
        test_input = torch.stack([transforms(dummy_dataset[i][0]) for i in range(2)]).to(device)
        test_output = test_model(test_input)
        
        if test_output.shape != (2, 10):
            return False, f"Model with trained weights produced wrong output shape: {test_output.shape}, expected (2, 10)"
        
        return True, None
        
    except Exception as e:
        return False, f"Training integrity check failed: {str(e)}"
    

def test_transform_determinism(transforms):
    """
    Verify that transforms returned by get_transforms('eval') are deterministic.
    
    Returns:
        bool: True if transforms are deterministic, False otherwise
        str: Error message if check fails, None otherwise
    """
    try:     
        # Apply the transforms 20 times to each of 3 dummy images (PIL): all outputs must match
        for _ in range(3):
            dummy_image = torchvision.transforms.ToPILImage()(torch.rand(1, 28, 28))
            output1 = transforms(dummy_image)
            for _ in range(20):
                if not torch.allclose(output1, transforms(dummy_image)):
                    return False, "Non-deterministic transforms in eval mode"
        
        return True, None
    
    except Exception as e:
        return False, f"Transform determinism check failed: {str(e)}"
    

def validate_transform_object(transform_obj, visited=None, strict=True):
    """
    Recursively validate a transform object to ensure it only uses allowed torchvision transforms.
    With strict=False (training transforms), any torchvision transform not in BLACKLIST is allowed.

    Returns:
        bool: True if transforms are valid, False otherwise
        str: Error message if check fails, None otherwise
    """
    ALLOWED_TRANSFORMS = {
        # Basic preprocessing 
        'Normalize', 'Resize', 'CenterCrop', 'Pad', 
        'Grayscale', 'ToTensor', 'ToPILImage',
        'ToImage', 'ToDtype', 'ConvertImageDtype', # v2 replacements for ToTensor
        
        # Augmentation (eval transforms must still be deterministic)
        'RandomHorizontalFlip', 'RandomVerticalFlip', 
        'RandomRotation', 'RandomCrop', 'ColorJitter',
        'RandomGrayscale', 'RandomPerspective', 'GaussianBlur',
        
        # Compositions, should not be used with arbitrary functions or in eval
        'Compose', 'RandomApply', 'RandomChoice', 'RandomOrder',
    }

    BLACKLIST = {
        'Lambda', 'LinearTransformation'
    }

    if visited is None:
        visited = set()
    
    obj_id = id(transform_obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    
    # Check it's from torchvision.transforms
    module = type(transform_obj).__module__
    if not module.startswith('torchvision.transforms'):
        raise ValueError(
            f"Transform {type(transform_obj).__name__} is not from "
            f"torchvision.transforms (found in {module}). "
            f"Please avoid using arbitrary functions or lambdas. "
        )
    
    # Check it's on whitelist
    class_name = type(transform_obj).__name__
    if class_name in BLACKLIST:
        raise ValueError(
                         f"Transform '{class_name}' is not allowed. "
                         f"Please avoid using arbitrary functions or lambdas. "
                         f"Allowed: {sorted(ALLOWED_TRANSFORMS)}"
                         )
    if strict and class_name not in ALLOWED_TRANSFORMS:
        raise ValueError(
            f"Transform '{class_name}' is not on the allowed list. "
            f"Please contact a course leader if you have a legitimate use case. "
        )
    
    # Recursively check containers
    if isinstance(transform_obj, torchvision.transforms.Compose):
        for t in transform_obj.transforms:
            validate_transform_object(t, visited, strict)
    
    if hasattr(transform_obj, 'transforms'):
        for t in transform_obj.transforms:
            validate_transform_object(t, visited, strict)


def check_valid_transforms(transforms, strict=True):
    """
    Verify that transforms are composed of valid torchvision transforms only.
    Arbitrary functions or lambdas are not allowed.

    Returns:
        bool: True if transforms are valid, False otherwise
        str: Error message if check fails, None otherwise
    """
    try:
        validate_transform_object(transforms, strict=strict)
    except ValueError as e:
        return False, f"Disallowed transform: {str(e)}"
    except Exception as e:
        return False, f"Validate transforms check failed: {str(e)}"
    return True, None
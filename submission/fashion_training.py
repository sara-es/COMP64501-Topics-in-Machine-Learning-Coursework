"""
Feel free to replace this code with your own model training code. 
This is just a simple example to get you started.

This training script uses imports relative to the project root directory.
To run this training script with uv, ensure you're in the project root directory
and execute: uv run -m submission.fashion_training
"""
import os
import numpy as np
import torch, torchvision

from submission import engine
from submission.fashion_model import Net


def train_fashion_model(fashion_mnist, 
                        n_epochs, 
                        batch_size=32,
                        learning_rate=0.001,
                        USE_GPU=False,
                        seed=None,
                        model_kwargs=None,
                        return_state_dict_only=True,):
    """
    You can modify the contents of this function as needed, but DO NOT CHANGE the arguments,
    the function name, or the return value when return_state_dict_only=True, as this will be called during marking!
    (You can change the default values or add additional keyword arguments if needed.)
    During marking this is also run on a small dummy dataset (30 samples, 1 epoch), so it must work on that too.
    """
    # Optionally fix the random seed, so that results can be repeated
    if seed is not None:
        engine.set_seed(seed)

    # Optionally use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create train-val split
    train_size = int(0.8 * len(fashion_mnist))
    val_size = len(fashion_mnist) - train_size
    train_data, val_data = torch.utils.data.random_split(fashion_mnist, [train_size, val_size])

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )

    # Initialize model, loss function, and optimizer
    model = Net(**(model_kwargs or {}))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
        val_loss, accuracy = engine.evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    if return_state_dict_only:
        return model.state_dict()
    # Add whatever else you want to return here - return_state_dict_only=True will be enforced at runtime
    return model.state_dict(), train_losses, val_losses, val_accuracies


def get_transforms(mode='train'):
    """
    Define any data augmentations or preprocessing here if needed.
    Only standard torchvision transforms are permitted (no lambda functions), please check that 
    these pass by running model_calls.py before submission. Transforms will be set to .eval()
    (deterministic) mode during evaluation, so avoid using stochastic transforms like RandomCrop
    or RandomHorizontalFlip unless they can be set to p=0 during eval.
    """
    if mode == 'train':
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
        ])
    elif mode == 'eval': # no stochastic transforms, or use p=0
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
        ])
        for tf in tfs.transforms:
            if hasattr(tf, 'train'):
                tf.eval()  # set to eval mode if applicable # type: ignore
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.ipynb! 
    # To use them for training or inference, we need to transform them to tensors. 
    # We set this transform here, as well as any other data preprocessing or augmentation you 
    # wish to apply.
    fashion_mnist.transform = get_transforms(mode='train')
    return fashion_mnist


def main():
    # example usage
    # you could create a separate file that calls train_fashion_model with different parameters
    # or modify this as needed to add cross-validation, hyperparameter tuning, etc.
    fashion_mnist = load_training_data()

    # TODO: create data splits

    # TODO: implement hyperparameter search

    # Train model 
    # TODO: this may be done within a loop for hyperparameter search / cross-validation
    model_weights = train_fashion_model(fashion_mnist, n_epochs=1, seed=0)

    # Save model weights
    # However you tune and evaluate your model, make sure to save the final weights 
    # to submission/model_weights.pth before submission!
    model_save_path = os.path.join('submission', 'model_weights.pth')
    torch.save(model_weights, f=model_save_path)


if __name__ == "__main__":
    main()
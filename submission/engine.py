"""
Feel free to replace this code with your own model training code. 
This is just a simple example to get you started.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    # Fix the random number generators, so that runs can be repeated (on the CPU)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model, data_loader, criterion, optimizer, device):
    model.to(device)
    model.train()

    epoch_loss = 0.0
    n_samples = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0) # loss.item() is the mean over the batch
        n_samples += inputs.size(0)

    avg_loss = epoch_loss / max(n_samples, 1)
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    model.to(device)
    model.eval()

    targets = []
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            targets.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            total_loss += loss.item() * inputs.size(0)

    accuracy = (np.array(predictions) == np.array(targets)).mean()
    avg_loss = total_loss / max(len(targets), 1)
    return avg_loss, accuracy


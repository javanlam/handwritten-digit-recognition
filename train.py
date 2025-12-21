import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Tuple, Dict, List


def train_epoch(
        model: nn.Module, 
        train_loader: DataLoader, 
        criterion: callable, 
        optimizer: Optimizer,
        device: torch.device 
    ) -> Tuple[float, float]:
    """
    Trains model for one epoch.
    
    args:
    - model (nn.Module): the model to train
    - train_loader (DataLoader): a DataLoader object for loading the training set
    - criterion: the loss function to be used
    - optimizer (Optimizer): the optimizer to be used
    - device (torch.device): specifies device to use model on

    returns:
    - a tuple of the training loss and training accuracy
    """
    model.to(device)
    model.train()

    train_loss = 0.0
    correct_pred = 0
    total_samples = 0


    for i, data in enumerate(train_loader, 0):
        inputs, labels = data                   # data is a list of [inputs, labels]
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()                   # gradients to optimizer are zero

        outputs = model(inputs)                 # forward pass

        loss = criterion(outputs, labels)       # backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        _, pred = torch.max(outputs, 1)

        total_samples += labels.size(0)
        correct_pred += (pred == labels).sum().item()

        if i % 500 == 499:                      # show progress every 500 steps
            cur_item = i * len(inputs)
            print(f"Step {cur_item}: Loss {loss}")

    avg_loss = train_loss / total_samples
    accuracy = correct_pred / total_samples

    return avg_loss, accuracy


def validate_epoch(
        model: nn.Module, 
        val_loader: DataLoader, 
        criterion: callable, 
        device: torch.device
    ) -> Tuple[float, float]:
    """
    Validates model on the validation set.
    
    args:
    - model (nn.Module): the model to train
    - val_loader (DataLoader): a DataLoader object for loading the validation set
    - criterion: the loss function to be used
    - device (torch.device): specifies device to use model on

    returns:
    - a tuple of the validation loss and validation accuracy
    """
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    correct_pred = 0
    total_samples = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data                   # data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, pred = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct_pred += (pred == labels).sum().item()

    avg_loss = test_loss / total_samples
    accuracy = correct_pred / total_samples

    return avg_loss, accuracy


def training_loop(
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        criterion: callable, 
        optimizer: Optimizer, 
        scheduler: any, 
        epochs: int, 
        device: torch.device, 
        save_path: str = 'best_model.pth'
    ) -> Dict[str, List]:
    """
    Completes the training pipeline.

    args:
    - model (nn.Module): the model to train
    - train_loader (DataLoader): a DataLoader object for loading the training set
    - val_loader (DataLoader): a DataLoader object for loading the validation set
    - criterion: the loss function to be used
    - optimizer (Optimizer): the optimizer to be used
    - scheduler: the learning rate scheduler
    - epochs (int): number of epochs to train the model for
    - device (torch.device): specifies device to use model on
    - save_path (str): the path to which the model will be saved to

    returns:
    - a Python dictionary of lists containing training logs
    """
    best_val_accuracy = 0.0
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        print("=" * 40)

        train_loss, train_accuracy = train_epoch(
            model=model, 
            train_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device
        )

        val_loss, val_accuracy = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.params_groups[0]["lr"]
            print(f"Current learning rate: {current_lr}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "train_accuracy": train_accuracy,
            }, save_path)
            print(f"Best model saved with validation accuracy: {val_accuracy:.4f}")

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_accuracy)
        training_history["val_loss"].append(val_loss)
        training_history["val_accuracy"].append(val_accuracy)

        print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        print()

        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Training completed. Best validation accuracy: {best_val_accuracy}")

        return training_history
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from typing import Dict, Tuple, List
import random


def evaluate_model(
        model: nn.Module, 
        test_loader: DataLoader,
        criterion: callable,  
        device: torch.device
    ) -> Dict:
    """
    Evaluates model on test set.
    
    args:
    - model (nn.Module): the model to evaluate
    - test_loader (DataLoader): a DataLoader object for loading the test set
    - criterion: the loss function to be used
    - device (torch.device): specifies device to use model on

    returns:
    - a Python dictionary containing evaluation metrics
    """
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    correct_pred = 0
    total_samples = 0

    test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data                                           # data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            total_samples += labels.size(0)
            correct_pred += (pred == labels).sum().item()

    avg_loss = test_loss / len(test_loader)                                 # calculates metrics
    accuracy = correct_pred / total_samples

    class_names = [f"{i}" for i in range(len(np.unique(all_labels)))]
    report = classification_report(                                         # classification report using sklearn.metrics
        y_true=all_labels,
        y_pred=all_predictions,
        target_names=class_names,
        output_dict=True
    )

    con_matrix = confusion_matrix(                                          # confusion matrix
        y_true=all_labels,
        y_pred=all_predictions
    )

    precision, recall, f1, support = precision_recall_fscore_support(       # precision, recall, F1 score, support
        y_true=all_labels,
        y_pred=all_predictions,
        average="weighted"
    )

    per_class_accuracy = []                                                 # accuracy for each class
    for class_idx in range(len(class_names)):
        class_mask = (all_labels == class_idx)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_predictions[class_mask] == class_idx)
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0.0)

    evaluation_results = {
        "accuracy": accuracy,
        "test_loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "F1_score": f1,
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
        "confusion_matrix": con_matrix,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "labels": all_labels
    }

    return evaluation_results


def visualize_predictions(
        model: nn.Module, 
        test_loader: DataLoader, 
        device: torch.device, 
        num_samples: int = 10
    ) -> Tuple[float, float]:
    """
    Visualizes model predictions with ground truth.
    
    args:
    - model (nn.Module): the model to evaluate
    - test_loader (DataLoader): a DataLoader object for loading the test set
    - device (torch.device): specifies device to use model on
    - num_samples (int): number of samples to visualize

    returns:
    - a tuple (number of correct items, total number of samples)
    """
    class_names = [f"{i}" for i in range(10)]

    model.to(device)
    model.eval()

    dataiter = iter(test_loader)                        # gets a batch of data
    images, labels = next(dataiter)

    num_samples = min(num_samples, len(images))         # avoid trying to visualize more than available

    if len(images) > num_samples:                       # sample data items for visualization
        indices = random.sample(range(len(images)), num_samples)
    else:
        indices = range(len(images))

    images = images[indices]
    labels = labels[indices]

    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    images = images.cpu()
    outputs = outputs.cpu()
    probabilities = probabilities.cpu()
    pred = pred.cpu()
    labels = labels.cpu()

    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    correct_count = 0

    for idx, (img, true_label, prediction, prob) in enumerate(zip(
        images, labels, pred, probabilities
    )):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        img = img.squeeze(0)                            # removes channel dimension since MNIST images are greyscale
        ax.imshow(img.numpy(), cmap="gray")
        
        is_correct = (prediction == true_label).item()
        title_color = "green" if is_correct else "red"
        
        if is_correct:
            correct_count += 1
        
        confidence = prob[prediction].item()
        title = (f"True: {class_names[true_label]}\n"
                f"Pred: {class_names[prediction]} ({confidence:.2f})\n"
                f"{"Correct" if is_correct else "Incorrect"}")
        
        ax.set_title(title, color=title_color, fontsize=10, fontweight="bold")
        ax.axis("off")

    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            axes[row, col].axis("off")
        else:
            axes[col].axis("off")

    plt.suptitle(
        "Model Predictions on Test Samples", 
        fontsize=16, 
        fontweight="bold", 
        y=1.02
    )

    plt.tight_layout()

    plt.show()
    
    return correct_count, num_samples


def plot_training_history(
        train_losses: List[float], 
        val_losses: List[float], 
        train_accs: List[float], 
        val_accs: List[float]
    ):
    """
    Plots training and validation metrics over epochs.
    
    args:
    - train_losses (List[float]): a list of training losses over epochs
    - val_losses (List[float]): a list of validation losses over epochs
    - train_accs (List[float]): a list of training accuracies over epochs
    - val_accs (List[float]): a list of validation accuracies over epochs
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', alpha=0.2)

    plt.suptitle('Model Training History', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.show()
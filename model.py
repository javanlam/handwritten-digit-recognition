import torch
import torch.nn as nn


class DigitRecognizer(nn.Module):
    """
    CNN model for digit recognition.
    """
    def __init__(self, num_classes: int = 10):
        """
        Initializes model layers.
        
        args:
        - num_classes (int): number of output classes
        """
        super(DigitRecognizer, self).__init__()
        
        self.model = nn.Sequential(
            # input shape: (1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape after convolution: (32, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # shape after pooling: (32, 14, 14)
            nn.Dropout2d(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape after convolution: (64, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # shape after pooling: (64, 7, 7)
            nn.Dropout2d(p=0.25),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features=64*7*7, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=128, out_features=num_classes)
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass.
        
        args:
        - x (torch.Tensor): input image (shape (1, 28, 28)) from MNIST dataset

        returns:
        - a tensor of logits (shape (num_classes, ))
        """
        features = self.model(x)
        logits = self.classifier(features)

        return logits


def initialize_model(device: torch.device = torch.device("cuda")) -> DigitRecognizer:
    """
    Creates model instance and moves to appropriate device.
    
    args:
    - device (torch.device): specifies device to use model on

    returns:
    - a DigitRecognizer model object
    """
    model = DigitRecognizer(num_classes=10)
    model.to(device)

    return model
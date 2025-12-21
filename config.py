import torch
import torch.nn as nn
from typing import Tuple, Literal


class Config:
    """Configuration class for model hyperparameters."""
    def __init__(
            self,
            batch_size: int = 64,
            val_split: float = 0.2,
            num_classes: int = 10,
            input_size: Tuple[int, int, int] = (1, 28, 28),
            learning_rate: float = 1e-3,
            epochs: int = 10,
            momentum: float = 0.9,
            optimizer: Literal["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"] = "Adam",
            scheduler: Literal["StepLR", "MultiStepLR", "ExponentialLR", "ReduceLRonPlateau"] = "StepLR",
            model_path: str = "./saved_models/",
            log_dir: str = "./logs/"
        ):
        """
        Initializes a class instance for model hyperparameters.

        args:
        - batch_size (int): batch size
        - val_split (float): proportion of training data to be used as validation set
        - num_classes (int): number of output classes
        - input_size (int): dimensions of input image
        - learning_rate (float): learning rate during training
        - epochs (int): number of epochs to train for
        - momentum (float): momentum parameter
        - optimizer (str): type of optimizer to use
        - scheduler (str): type of learning rate scheduler to use
        - model_path (str): path to save the model to
        - log_dir (str): directory to save training logs
        """
        self.batch_size = batch_size
        self.val_split = val_split

        self.num_classes = num_classes
        self.input_size = input_size

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model_save_path = model_path
        self.log_dir = log_dir

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
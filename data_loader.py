import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from typing import Tuple, Dict


def prepare_data_loaders(
        batch_size: int = 64, 
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare MNIST dataset with transformations and split into train/val/test loaders.
    
    args:
    - batch_size (int): batch size
    - validation_split (float): proportion of training data to be used as validation set

    returns:
    - a tuple of Dataset and DataLoader objects for training set, validation set, and test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),              # transforms data to a torch.FloatTensor (shape C x H x W) in the range of [0.0, 1.0]
        transforms.Normalize(
            mean=(0.5,), 
            std=(0.5,)
        )                                   # scales the data to [-1.0, 1.0] to zero-centre the data
    ])

    # load training dataset and prepare DataLoader

    train_dataset = datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )

    lengths=[int(len(train_dataset) * (1-validation_split)), int(len(train_dataset) * validation_split)]
    generator = torch.Generator().manual_seed(12345)

    train, val = random_split(
        dataset=train_dataset,
        lengths=lengths,
        generator=generator
    )

    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,                       # shuffles the data          
        num_workers=2                       # loads data in parallel
    )

    val_loader = DataLoader(
        dataset=val,
        batch_size=batch_size,
        shuffle=True,                       # shuffles the data          
        num_workers=2                       # loads data in parallel
    )

    # load test dataset and prepare DataLoader

    test_dataset = datasets.MNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,                       # shuffles the data          
        num_workers=2                       # loads data in parallel
    )

    return train_loader, val_loader, test_loader
    

def get_data_statistics(
        data_loader: DataLoader
    ) -> Dict[str, float]:
    """
    Calculate and display dataset statistics.
    
    args:
    - data_loader (DataLoader): a DataLoader object initialized for loading a training/validation/test set
    
    returns:
    - a Python dictionary containing the mean and standard deviation of image data
    """
    sum = 0
    sum_squared = 0

    num_batches = 0

    for data, _ in data_loader:
        # iterates through batches
        num_batches += 1

        sum += torch.mean(data, dim=[0,2,3])
        sum_squared += torch.mean(data**2, dim=[0,2,3])

    mean = sum / num_batches
    std = torch.sqrt(sum_squared / num_batches - mean ** 2)     # std = sqrt(E(X^2) - (E(X))^2)

    statistics = {
        "mean": mean,
        "std": std
    }

    return statistics
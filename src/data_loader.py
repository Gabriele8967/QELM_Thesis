import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

def get_mnist_dataloaders(
    batch_size: int, 
    data_dir: str = "./data/raw", 
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    
    """
    Downloads and prepares the MNIST dataset.
    
    Args:
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory where the dataset will be stored.
        num_workers (int): Number of subprocesses for data loading. 
                           Set to 0 on Windows/Laptops to avoid crashes.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    
    # Transformation pipeline:
    # 1. Convert image to PyTorch Tensor (Scales pixel values to [0, 1])
    # Note: We do NOT use extra normalization (mean/std) here because 
    # the Autoencoder uses a Sigmoid activation (output [0, 1]).
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load Training Set
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load Test Set
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create DataLoaders
    # shuffle=True for training to break correlations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # shuffle=False for testing (evaluation must be deterministic)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader
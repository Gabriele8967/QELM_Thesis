import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def get_mnist_dataloaders(batch_size: int, data_dir = None, num_workers : int = 0) -> tuple[DataLoader, DataLoader]:

    if data_dir is None:
        base_path = Path(__file__).resolve().parent.parent
        data_dir = base_path / "data" / "raw"
    else:
        data_dir = Path(data_dir).resolve()
    
    data_dir.mkdir(parents=True, exist_ok=True)

    train_data = datasets.MNIST(
        root=str(data_dir),
        train = True,
        download = True,
        transform = ToTensor(),
        target_transform = None
        )


    test_data = datasets.MNIST(
        root = str(data_dir),
        train = False,
        download = True,
        transform = ToTensor()
    )

    train_dataloader = DataLoader(train_data,
                                batch_size = batch_size,
                                shuffle=True,
                                num_workers=num_workers)

    test_dataloader = DataLoader(test_data,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers=num_workers)
    
    return train_dataloader, test_dataloader

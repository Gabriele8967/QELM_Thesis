import sys
import os
import time
import yaml 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path

file_dir = Path(__file__).resolve().parent

project_root = file_dir.parent.parent

sys.path.append(str(project_root))

from src.models.hybrid_model import HybridModel


def load_config(config_filename="config.yaml"):
    full_path = project_root / "configs" / config_filename
    
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")
    
    print(f"Config loaded from: {full_path}")
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)
    

def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

def train_qelm():
    cfg = load_config()
    SEED = cfg['experiment']['seed']
    DEVICE_STR = cfg['experiment']['device']
    
    BATCH_SIZE = cfg['data']['batch_size']
    DATA_PATH = cfg['data']['path']
    
    N_QUBITS = cfg['model']['latent_dim'] 
    
    N_EPOCHS = cfg['training']['epochs']
    LR = cfg['training']['lr']
    AE_WEIGHTS_PATH = cfg['training']['save_path']
    
    QELM_SAVE_PATH = "./data/checkpoints/qelm_final_model.pth"

    torch.manual_seed(SEED)
    device = get_device(DEVICE_STR)

    transform = transforms.Compose([
        transforms.ToTensor()  
    ])

    os.makedirs(DATA_PATH, exist_ok=True)
    train_dataset = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    
    ae_full_path = os.path.join(project_root, AE_WEIGHTS_PATH)
    
    if not os.path.exists(ae_full_path):
        print(f"Weight not found in: {ae_full_path}")
        return

    model = HybridModel(model_path=ae_full_path, n_qbits=N_QUBITS)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)


    history = {'train_acc': [], 'train_loss': []}

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"   Ep {epoch+1} | Bt {batch_idx} | Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        history['train_acc'].append(epoch_acc)
        history['train_loss'].append(epoch_loss)

        print(f"Epoch {epoch+1} in {epoch_time:.1f}s | Acc: {epoch_acc:.2f}% | Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

    final_path = os.path.join(project_root, QELM_SAVE_PATH)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\n Model QELM saved in: {final_path}")


    plt.plot(history['train_acc'], label='Accuracy')
    plt.plot(history['train_loss'], label='Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_qelm()


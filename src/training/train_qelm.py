import sys
import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

file_dir = Path(__file__).resolve().parent
project_root = file_dir.parent.parent
sys.path.append(str(project_root))

from src.models.hybrid_model import HybridModel

def load_config(config_filename="config.yaml"):
    full_path = project_root / "configs" / config_filename
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_qelm_pseudoinverse():
    print("--- AVVIO TRAINING QELM (METODO PSEUDO-INVERSA) ---")

    cfg = load_config()
    DEVICE_STR = cfg['experiment']['device']
    BATCH_SIZE = cfg['data']['batch_size']
    DATA_PATH = cfg['data']['path']
    AE_WEIGHTS_PATH = cfg['training']['save_path']

    latent_dim_config = cfg['model']['latent_dim']
    N_QUBITS = latent_dim_config // 2

    device = get_device(DEVICE_STR)
    print(f"Device: {device}")
    print(f"Qubits: {N_QUBITS} (Feature Space: {2**N_QUBITS})")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ae_full_path = os.path.join(project_root, AE_WEIGHTS_PATH)
    model = HybridModel(model_path=ae_full_path, n_qbits=N_QUBITS)
    model.to(device)
    model.eval()

    print("\n[Fase 1] Estrazione delle feature quantistiche (H)...")

    H_list = []
    Y_list = []

    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            latent = model.ae.encode(images)
            latent = latent * torch.pi

            features = model.quantum_layer(latent)

            H_list.append(features.cpu())
            Y_list.append(labels.cpu())

            if i % 50 == 0:
                print(f"   Batch {i}/{len(train_loader)} estratto.")

    H = torch.cat(H_list, dim=0).to(device)
    labels_tensor = torch.cat(Y_list, dim=0).to(device)

    ones_col = torch.ones(H.size(0), 1, device=device)
    H_aug = torch.cat([H, ones_col], dim=1)

    print(f"Dimensione Matrice H: {H_aug.shape}")

    Y = torch.nn.functional.one_hot(labels_tensor, num_classes=10).float()

    print("\n[Fase 2] Calcolo Pseudo-Inversa (One-Shot Learning)...")

    dim_aug = H_aug.shape[1]
    reg_lambda = 1e-2
    I = torch.eye(dim_aug, device=device)

    lhs = (H_aug.T @ H_aug) + reg_lambda * I
    rhs = H_aug.T @ Y

    beta_aug = torch.linalg.solve(lhs, rhs)

    beta_T = beta_aug.T
    weights = beta_T[:, :-1]

    bias = beta_T[:, -1]

    print("   Pesi calcolati!")

    print("\n[Fase 3] Applicazione pesi al classificatore...")

    with torch.no_grad():
        model.classifier.weight.data = weights
        model.classifier.bias.data = bias

    print("   Modello aggiornato.")
    training_time = time.time() - start_time
    print(f"   Tempo totale training: {training_time:.2f} secondi")

    print("\n[Fase 4] Validazione Test Set...")
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print("-" * 40)
    print(f"RISULTATO FINALE (Pseudo-Inverse): {acc:.2f}%")
    print("-" * 40)

    save_path = "./data/checkpoints/qelm_pinv_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in {save_path}")

if __name__ == "__main__":
    train_qelm_pseudoinverse()

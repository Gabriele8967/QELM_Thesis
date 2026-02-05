import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.autoencoder import ConvAutoencoder
from src.data_loader import get_mnist_dataloaders
from src.utils import set_seed
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_config(path="configs/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels_list = []

    print("Estrazione features in corso...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            latent = model.encode(images)
            latent = latent.view(latent.size(0), -1)

            features.append(latent.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels_list)

def main():
    config = load_config()
    device = config['experiment']['device']
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'

    latent_dim = config['model']['latent_dim']
    batch_size = config['data']['batch_size']
    ae_path = config['training']['save_path']

    print(f"--- BENCHMARK CLASSICO (Solo Autoencoder) ---")
    print(f"Latent Dim: {latent_dim}")

    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    model = ConvAutoencoder(latent_dim=latent_dim)
    try:
        model.load_state_dict(torch.load(ae_path, map_location=device))
        print("Autoencoder caricato con successo.")
    except FileNotFoundError:
        print(f"ERRORE: Non trovo il file {ae_path}. Addestra prima l'AE!")
        return

    model.to(device)

    print("\nEstrazione Training Set...")
    X_train, y_train = extract_features(model, train_loader, device)

    print("Estrazione Test Set...")
    X_test, y_test = extract_features(model, test_loader, device)

    print(f"\nDimensioni Dataset Classico:")
    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")

    print("\nAddestramento Classificatore Lineare (Baseline)...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print("-" * 30)
    print(f"RISULTATI BASELINE (NO QUANTUM)")
    print(f"Accuratezza Training: {train_acc*100:.2f}%")
    print(f"Accuratezza Test:     {test_acc*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()

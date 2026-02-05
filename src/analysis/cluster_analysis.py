import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm

file_dir = Path(__file__).resolve().parent
project_root = file_dir.parent.parent
sys.path.append(str(project_root))

from src.models.hybrid_model import HybridModel

def run_cluster_analysis():
    print("--- ANALISI CLUSTERING: K-MEANS, INERTIA & ARI ---")

    N_QUBITS = 10
    N_LAYERS = 5
    BATCH_SIZE = 256
    DEVICE = torch.device("cpu")
    MODEL_PATH = os.path.join(project_root, "data/checkpoints/qelm_pinv_model.pth")

    print("Caricamento MNIST Test Set...")
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Caricamento Modello Ibrido...")
    model = HybridModel(model_path=None, n_qbits=N_QUBITS, n_layers=N_LAYERS)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Checkpoint caricato con successo.")
    else:
        raise FileNotFoundError(f"Non trovo il modello in {MODEL_PATH}")

    model.to(DEVICE)
    model.eval()

    print("Estrazione delle feature dal Reservoir...")
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)

            latent = model.ae.encode(images)
            latent = latent * torch.pi

            q_features = model.quantum_layer(latent)

            features_list.append(q_features.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(features_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0)

    print(f"Spazio delle Feature estratto: {X.shape}")

    print("\n--- Calcolo Elbow Curve (Inertia) ---")
    inertias = []
    k_range = range(2, 16)

    for k in k_range:
        print(f"Fitting K-Means con k={k}...", end="\r")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    print("\nCalcolo completato.")

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Numero di Cluster (k)', fontsize=12)
    plt.ylabel('Inertia (Somma distanze quadrate)', fontsize=12)
    plt.title('Elbow Method per determinare k ottimale', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)

    if 10 in k_range:
        idx_10 = list(k_range).index(10)
        plt.plot(10, inertias[idx_10], 'r*', markersize=15, label='Ground Truth (k=10)')
        plt.legend()

    plt.savefig('kmeans_elbow.png')
    print("Grafico salvato: kmeans_elbow.png")

    print("\n--- Calcolo ARI per k=10 ---")
    kmeans_10 = KMeans(n_clusters=10, random_state=42, n_init=10)
    y_pred = kmeans_10.fit_predict(X)

    ari_score = adjusted_rand_score(y_true, y_pred)

    print(f"\nRISULTATI:")
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print("-" * 30)
    if ari_score > 0.8:
        print("ECCELLENTE. I cluster quantistici corrispondono quasi perfettamente alle cifre.")
    elif ari_score > 0.6:
        print("BUONO. C'è una struttura chiara, ma alcune cifre simili si confondono.")
    else:
        print("BASSO. Lo spazio delle feature è ancora confuso.")

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.xlabel('Predicted Cluster ID')
    plt.ylabel('True Digit Label')
    plt.title('Confusion Matrix: True Labels vs K-Means Clusters')
    plt.savefig('kmeans_confusion.png')
    print("Grafico salvato: kmeans_confusion.png")

if __name__ == "__main__":
    run_cluster_analysis()

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm

file_dir = Path(__file__).resolve().parent
project_root = file_dir.parent.parent
sys.path.append(str(project_root))

from src.models.hybrid_model import HybridModel

def visualize_quantum_manifold():
    print("--- VISUALIZZAZIONE MANIFOLD (t-SNE) ---")

    N_QUBITS = 10
    N_LAYERS = 5
    LIMIT_SAMPLES = 2000
    DEVICE = torch.device("cpu")
    MODEL_PATH = os.path.join(project_root, "data/checkpoints/qelm_pinv_model.pth")

    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    subset_indices = torch.randperm(len(test_dataset))[:LIMIT_SAMPLES]
    subset = torch.utils.data.Subset(test_dataset, subset_indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False)

    model = HybridModel(model_path=None, n_qbits=N_QUBITS, n_layers=N_LAYERS)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("Estrazione feature quantistiche...")
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            latent = model.ae.encode(images) * torch.pi
            q_features = model.quantum_layer(latent)

            features_list.append(q_features.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    print("Calcolo t-SNE (riduzione dimensionale 1024 -> 2)...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, verbose=1)
    X_embedded = tsne.fit_transform(X)

    print("Generazione grafico...")
    plt.figure(figsize=(16, 10))

    sns.scatterplot(
        x=X_embedded[:,0], y=X_embedded[:,1],
        hue=y,
        palette=sns.color_palette("tab10", 10),
        legend="full",
        alpha=0.7,
        s=60
    )

    plt.title(f"t-SNE del Reservoir Quantistico (Accuracy Modello: ~96%)\n"
              f"Visualizzazione 2D dello spazio a {2**N_QUBITS} dimensioni", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Cifra Reale", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = "tsne_manifold.png"
    plt.savefig(save_path)
    print(f"Grafico salvato in: {save_path}")

if __name__ == "__main__":
    visualize_quantum_manifold()

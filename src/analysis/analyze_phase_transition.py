import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm

file_dir = Path(__file__).resolve().parent
project_root = file_dir.parent.parent
sys.path.append(str(project_root))

from src.models.hybrid_model import HybridModel
from src.quantum.gates import get_ry_matrix, get_phase_matrix, get_x_matrix

def measure_system_properties(n_layers, device, train_loader, test_loader, ae_state_dict):
    N_QUBITS = 10

    model = HybridModel(model_path=None, n_qbits=N_QUBITS, n_layers=n_layers)
    model.ae.load_state_dict(ae_state_dict)
    model.to(device)
    model.eval()

    H_list = []
    Y_list = []

    limit = 2000
    count = 0

    with torch.no_grad():
        for images, labels in train_loader:
            if count >= limit: break
            images = images.to(device)
            latent = model.ae.encode(images) * torch.pi
            features = model.quantum_layer(latent)

            H_list.append(features.cpu())
            Y_list.append(labels.cpu())
            count += images.size(0)

    H = torch.cat(H_list, dim=0).to(device)
    labels_tensor = torch.cat(Y_list, dim=0).to(device)
    Y_onehot = torch.nn.functional.one_hot(labels_tensor, num_classes=10).float()

    reg_lambda = 1e-2
    I = torch.eye(H.shape[1], device=device)
    beta = torch.linalg.solve(H.T @ H + reg_lambda * I, H.T @ Y_onehot)

    model.classifier.weight.data = beta.T
    model.classifier.bias.data.fill_(0)

    correct = 0
    total = 0
    limit_test = 1000

    with torch.no_grad():
        for images, labels in test_loader:
            if total >= limit_test: break
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    accuracy = 100 * correct / total

    images, _ = next(iter(test_loader))
    images = images.to(device)
    model.eval()

    latent = model.ae.encode(images) * torch.pi
    latent_np = latent.detach().cpu().numpy()

    sim = model.quantum_layer.simulator
    total_entropy = 0

    for i in range(len(images)):
        sim.reset_state()
        feats = latent_np[i]
        thetas, phis = feats[:N_QUBITS], feats[N_QUBITS:]

        for q in range(N_QUBITS):
            sim.apply_gate(get_ry_matrix(thetas[q]), q)
            if len(phis) > q: sim.apply_gate(get_phase_matrix(phis[q]), q)

        for l in range(n_layers):
            for q in range(N_QUBITS - 1):
                sim.apply_gate(get_x_matrix(), q+1, q)
            if N_QUBITS > 1:
                sim.apply_gate(get_x_matrix(), 0, N_QUBITS-1)
            for q in range(N_QUBITS):
                angle = np.random.uniform(0, 2*np.pi)
                sim.apply_gate(get_ry_matrix(angle), q)

        img_avg_ent = np.mean([sim.get_single_qubit_entropy(q) for q in range(N_QUBITS)])
        total_entropy += img_avg_ent

    avg_entropy = total_entropy / len(images)

    return accuracy, avg_entropy

def run_phase_analysis():
    print("--- ANALISI TRANSIZIONE DI FASE: ACCURACY VS ENTROPIA ---")

    DEVICE = torch.device("cpu")
    FULL_PATH = os.path.join(project_root, "data/checkpoints/qelm_pinv_model.pth")
    full_state = torch.load(FULL_PATH, map_location=DEVICE)

    ae_state = {k.replace('ae.', ''): v for k, v in full_state.items() if k.startswith('ae.')}

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)

    layers_range = [0, 1, 2, 3, 4, 5, 8, 12, 15, 20]
    accuracies = []
    entropies = []

    print("Inizio scansione layer...")
    for l in tqdm(layers_range):
        acc, ent = measure_system_properties(l, DEVICE, train_loader, test_loader, ae_state)
        accuracies.append(acc)
        entropies.append(ent)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Numero di Layers (Profondità Temporale)')
    ax1.set_ylabel('Test Accuracy (%)', color=color)
    ax1.plot(layers_range, accuracies, 'o-', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Entropia di Von Neumann Media', color=color)
    ax2.plot(layers_range, entropies, 's--', color=color, linewidth=2, label='Entanglement')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=np.log(2), color='green', linestyle=':', label='Max (ln 2)')

    plt.title("Transizione di Fase nel Reservoir:\nCorrelazione tra Entanglement e Apprendimento")
    fig.tight_layout()
    plt.savefig('phase_transition.png')
    print("Grafico salvato: phase_transition.png")

if __name__ == "__main__":
    run_phase_analysis()

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchvision import datasets, transforms

file_dir = Path(__file__).resolve().parent
project_root = file_dir.parent.parent
sys.path.append(str(project_root))

from src.models.hybrid_model import HybridModel

def analyze_entanglement():
    print("--- ANALISI ENTANGLEMENT (VON NEUMANN ENTROPY) ---")

    N_QUBITS = 10
    N_LAYERS = 5
    AE_PATH = "./data/checkpoints/qelm_pinv_model.pth"

    device = torch.device("cpu")

    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
    images, labels = next(iter(loader))

    model = HybridModel(model_path=None, n_qbits=N_QUBITS, n_layers=N_LAYERS)
    model.eval()

    entropies_per_sample = []

    print("Calcolo entropia per il batch...")

    with torch.no_grad():
        latent = model.ae.encode(images)
        latent = latent * torch.pi
        latent_np = latent.cpu().numpy()

        q_layer = model.quantum_layer
        simulator = q_layer.simulator

        for i in range(len(images)):
            features = latent_np[i]

            simulator.reset_state()
            thetas = features[:q_layer.n_qubits]
            phis = features[q_layer.n_qubits:]

            from src.quantum.gates import get_ry_matrix, get_phase_matrix, get_x_matrix

            for q in range(q_layer.n_qubits):
                ry = get_ry_matrix(thetas[q])
                simulator.apply_gate(ry, q)
                if len(phis) > q:
                    phase_gate = get_phase_matrix(phis[q])
                    simulator.apply_gate(phase_gate, q)

            for l in range(q_layer.n_layers):
                for q in range(q_layer.n_qubits - 1):
                    simulator.apply_gate(q_layer.X_mat, target_qubit=q+1, control_qubit=q)
                if q_layer.n_qubits > 1:
                    simulator.apply_gate(q_layer.X_mat, target_qubit=0, control_qubit=q_layer.n_qubits-1)

                for q in range(q_layer.n_qubits):
                    theta_res = q_layer.reservoir_angles[l, q]
                    ry_res = get_ry_matrix(theta_res)
                    simulator.apply_gate(ry_res, q)

            sample_entropies = []
            for q in range(q_layer.n_qubits):
                S = simulator.get_single_qubit_entropy(q)
                sample_entropies.append(S)

            entropies_per_sample.append(sample_entropies)

    entropies_np = np.array(entropies_per_sample)
    mean_entropies = np.mean(entropies_np, axis=0)
    std_entropies = np.std(entropies_np, axis=0)

    plt.figure(figsize=(12, 6))

    x = np.arange(N_QUBITS)
    plt.bar(x, mean_entropies, yerr=std_entropies, capsize=5, color='#4a90e2', alpha=0.8, label='Von Neumann Entropy')

    plt.axhline(y=np.log(2), color='r', linestyle='--', label='Max Entanglement (ln 2)')

    plt.xlabel('Qubit Index', fontsize=12)
    plt.ylabel('Von Neumann Entropy S(ρ)', fontsize=12)
    plt.title(f'Distribuzione Entanglement nel Reservoir\n(Media su {len(images)} immagini, {N_LAYERS} layers)', fontsize=14)
    plt.xticks(x)
    plt.ylim(0, 0.75)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.savefig('entanglement_analysis.png')

    plt.figure(figsize=(10, 8))
    sns.heatmap(entropies_np, annot=True, cmap="viridis", fmt=".2f",
                xticklabels=[f"Q{i}" for i in range(N_QUBITS)],
                yticklabels=[f"Img {i}" for i in range(len(images))])
    plt.title("Heatmap Entropia: Images vs Qubit")
    plt.xlabel("Qubit Index")
    plt.ylabel("Sample Index")
    plt.savefig('entanglement_heatmap.png')

if __name__ == "__main__":
    analyze_entanglement()

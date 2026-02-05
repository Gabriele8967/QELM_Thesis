from src.quantum.gates import get_hadamard_matrix, get_ry_matrix, get_rx_matrix, get_x_matrix, get_phase_matrix
from src.quantum.quantum_simulator import CustomQuantumSimulator
import torch
import torch.nn as nn
import numpy as np

class CustomQuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers=5):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.simulator = CustomQuantumSimulator(n_qubits)

        self.X_mat = get_x_matrix()

        self.reservoir_angles = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits))

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []

        x_np = x.detach().cpu().numpy()

        for b in range(batch_size):
            self.simulator.reset_state()
            features = x_np[b]

            thetas = features[:self.n_qubits]
            phis = features[self.n_qubits:]

            for q in range(self.n_qubits):
                ry = get_ry_matrix(thetas[q])
                self.simulator.apply_gate(ry, q)

                if len(phis) > q:
                    phase_gate = get_phase_matrix(phis[q])
                    self.simulator.apply_gate(phase_gate, q)

            for l in range(self.n_layers):

                for q in range(self.n_qubits - 1):
                    self.simulator.apply_gate(self.X_mat, target_qubit=q+1, control_qubit=q)

                if self.n_qubits > 1:
                    self.simulator.apply_gate(self.X_mat, target_qubit=0, control_qubit=self.n_qubits-1)

                for q in range(self.n_qubits):
                    theta_res = self.reservoir_angles[l, q]
                    ry_res = get_ry_matrix(theta_res)
                    self.simulator.apply_gate(ry_res, q)

            probs = self.simulator.get_probabilities()
            outputs.append(probs)

        return torch.tensor(np.array(outputs), dtype=torch.float32).to(x.device)

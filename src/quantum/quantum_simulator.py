import numpy as np

class CustomQuantumSimulator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0j

        self.indices = np.arange(self.dim, dtype=np.int32)

    def reset_state(self):
        self.state.fill(0.0)
        self.state[0] = 1.0 + 0j

    def apply_gate(self, gate_matrix, target_qubit, control_qubit=None):
        target_bit = 1 << target_qubit

        if control_qubit is None:
            mask_target_0 = (self.indices & target_bit) == 0
            i0 = self.indices[mask_target_0]
        else:
            control_bit = 1 << control_qubit
            mask = ((self.indices & target_bit) == 0) & ((self.indices & control_bit) != 0)
            i0 = self.indices[mask]

        i1 = i0 | target_bit

        a0 = self.state[i0]
        a1 = self.state[i1]

        u00, u01 = gate_matrix[0, 0], gate_matrix[0, 1]
        u10, u11 = gate_matrix[1, 0], gate_matrix[1, 1]

        new_a0 = u00 * a0 + u01 * a1
        new_a1 = u10 * a0 + u11 * a1

        self.state[i0] = new_a0
        self.state[i1] = new_a1

    def get_probabilities(self):
        return np.abs(self.state) ** 2

    def get_single_qubit_entropy(self, qubit_idx):
        state_tensor = self.state.reshape([2] * self.n_qubits)

        psi_moved = np.moveaxis(state_tensor, qubit_idx, 0)

        psi_2d = psi_moved.reshape(2, -1)

        rho_reduced = np.dot(psi_2d, psi_2d.conj().T)

        eigenvalues = np.linalg.eigvalsh(rho_reduced)

        entropy = 0.0
        for p in eigenvalues:
            if p > 1e-15:
                entropy -= p * np.log(p)

        return entropy

import torch
import torch.nn as nn
from pathlib import Path

from src.models.autoencoder import ConvAutoencoder
from src.quantum.custom_layer import CustomQuantumLayer

class HybridModel(nn.Module):
    def __init__(self, model_path=None, n_qbits=10, n_layers=2):
        super(HybridModel, self).__init__()

        self.ae = ConvAutoencoder(latent_dim=n_qbits * 2)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cpu')
            self.ae.load_state_dict(state_dict)

        for param in self.ae.parameters():
            param.requires_grad = False

        self.quantum_layer = CustomQuantumLayer(n_qubits=n_qbits, n_layers=5)

        for param in self.quantum_layer.parameters():
            param.requires_grad = False

        input_dim = 2**n_qbits

        self.classifier = nn.Linear(input_dim, 10)

    def forward(self, x):
        x = self.ae.encode(x)

        x = x * torch.pi

        x = self.quantum_layer(x)

        x = self.classifier(x)
        return x

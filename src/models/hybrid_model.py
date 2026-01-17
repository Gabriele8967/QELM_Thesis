import torch
import torch.nn as nn
from pathlib import Path

from src.models.autoencoder import ConvAutoencoder
from src.quantum.reservoir import get_quantum_reservoir

class HybridModel(nn.Module):
    def __init__(self, model_path, n_qbits = 10, n_layers=2):
        super(HybridModel,self).__init__()

        self.ae = ConvAutoencoder(latent_dim=n_qbits)

        state_dict = torch.load(model_path, map_location='cpu')
        self.ae.load_state_dict(state_dict)


        for param in self.ae.parameters():
            param.requires_grad = False

        
        self.quantum_layer = get_quantum_reservoir(n_qbit=n_qbits, n_layers=n_layers)

        for param in self.quantum_layer.parameters():
            param.requires_grad = False

        

        input_dim = 2**n_qbits

        self.classifier = nn.Linear(input_dim, 10)


    def forward(self, x):
        x = self.ae.encode(x)

        x = x * 2 * torch.pi 
        
       
        x = self.quantum_layer(x)

        
        x = self.classifier(x)
        return x
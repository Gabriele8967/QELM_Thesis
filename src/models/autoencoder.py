import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim= 10):
        """
        Autoencoder Convoluzionale per MNIST.
        
        Args:
            latent_dim (int): Dimensione dello spazio latente (collo di bottiglia).
                              Questo vettore sarà l'input per il Quantum Reservoir.
        """

        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # --- First Block ---
            # Input: (Batch, 1, 28, 28) -> Output: (Batch, 16, 14, 14)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 3, stride = 1, padding= 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),

            # --- Second Block ---
            # Input : (Batch, 16, 14 ,14) -> Output: (Batch, 32, 7, 7)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten_dim = 32 * 7 * 7

        # --- Latent space ---
        #linear projection to reduct space
        self.fc_encoder = nn.Linear(self.flatten_dim, latent_dim)


        ### Decorder - reconstruction

        self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            # Input (Batch, 32, 7, 7) -> Output : (Batch, 16, 14, 14)
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Input (Batch, 16, 14, 14 ) -> Output (Batch, 1, 28,28)
            nn.ConvTranspose2d(16,1,kernel_size=3, stride = 2, padding=1, output_padding= 1),

            nn.Sigmoid()
        )
    
    def forward(self,x):

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        latent_code = self.fc_encoder(x)

        x = self.fc_decoder(latent_code)

        x = x.view(x.size(0), 32, 7, 7)

        reconstructed = self.decoder(x)

        return latent_code, reconstructed
    
    def encode(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_encoder(x)




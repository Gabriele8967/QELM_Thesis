import yaml
import torch
from src.data_loader import get_mnist_dataloaders
from src.models.autoencoder import ConvAutoencoder
from src.training.train_ae import train_autoencoder

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    config = load_config("configs/config.yaml")
    batch_size = config['data']['batch_size']   
    latent_dim = config['model']['latent_dim']  
    lr = float(config['training']['lr'])        
    epochs = config['training']['epochs']       
    device = config['experiment']['device']     
    save_path = config['training']['save_path']

    if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA non disponibile. Passo a CPU.")
            device = 'cpu'
        
    
    print("Caricamento Dati")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
    
    
    print(f"Creazione Autoencoder (Latent Dim: {latent_dim})")
    model = ConvAutoencoder(latent_dim=latent_dim)
    
    
    print("Avvio Training")
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader = test_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path=save_path
    )
    
    print("\n Autoencoder Addestrato!")

if __name__ == "__main__":
    main()
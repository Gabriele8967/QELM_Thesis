import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # La barra di caricamento (fondamentale per non impazzire)
import os

def train_autoencoder(model, train_loader, test_loader, epochs, lr, device, save_path):

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, _ in progress_bar:
            
            images = images.to(device)

            optimizer.zero_grad()

            _, reconstructed = model(images)

            loss = criterion(reconstructed, images)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                _, reconstructed = model(images)

                loss = criterion(reconstructed, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_val_loss:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in: {save_path}")
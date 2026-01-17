# Quantum Extreme Learning Machines (QELM) for MNIST Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![QuTiP](https://img.shields.io/badge/QuTiP-Quantum%20Sim-orange)
![Status](https://img.shields.io/badge/Status-Thesis%20In%20Progress-green)

## Overview

This repository contains the official implementation for my **Bachelor's Thesis in Computer Science**.

The project investigates **Quantum Extreme Learning Machines (QELM)** applied to computer vision. It implements a hybrid architecture combining **Deep Learning** (Convolutional Autoencoders) for dimensionality reduction with **Quantum Reservoir Computing** for non-linear feature mapping.

The goal is to demonstrate that a quantum-enhanced pipeline can achieve competitive accuracy on the **MNIST dataset** while utilizing a fraction of the trainable parameters required by classical deep networks.

## The Architecture

The pipeline addresses the "NISQ bottleneck" (limited qubits) through a 4-step process:

1.  **Dimensionality Reduction:** A **Convolutional Autoencoder (CAE)** compresses $28 \times 28$ images into a compact latent vector (e.g., $N=4$ or $8$ features).
2.  **Quantum Encoding:** Latent features are mapped to a quantum state via **Dense Angle Encoding**.
3.  **Reservoir Evolution:** The system evolves under a Transverse Field Ising Hamiltonian, exploiting quantum entanglement to project data into a higher-dimensional Hilbert space.
4.  **Readout:** A classical linear layer (Ridge Regression) is trained on the quantum measurements.



## Project Structure

```text
QELM_Thesis/
├── configs/                 # Configuration files 
│   └── config.yaml          # Central control for parameters 
│
├── data/                    # Dataset storage
│   ├── raw/                 # MNIST raw binaries (downloaded automatically)
│   └── checkpoints/         # Saved model weights (.pth)
│
├── notebooks/               # Experimental Playground
│   └── 01_Exploration.ipynb # Visualization and prototyping
│
├── src/                     # Source Code Modules
│   ├── models/              # Neural Network Architectures
│   │   └── autoencoder.py   # PyTorch Convolutional Autoencoder
│   ├── quantum/             # Quantum Physics Logic
│   │   └── reservoir.py     # Qiskit Simulation & Hamiltonian
│   ├── training/            # Training Loops
│   │   ├── train_ae.py      # Classical Pre-training
│   │   └── train_qelm.py    # Quantum Pipeline Training
│   ├── data_loader.py       # Data ingestion pipeline
│   └── utils.py             # Reproducibility & Logging tools
│
├── main.py                  # Entry Point (Run this to train AE)
└── requirements.txt         # Dependencies
```

## Insallation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Gabriele8967/QELM_Thesis.git](https://github.com/Gabriele8967/QELM_Thesis.git)
   cd QELM_Thesis
   ```
2. **Install dependencies**: It is recommended to use a virtual environment.

## Configuration and Usage

1. **Configure the Experiment**

   Open ```configs/config.yaml``` to adjust parameters:

      ``` YAML
      model:
         latent_dim: 4      # Number of Qubits Features
      training:
         epochs: 10
      lr: 0.001
         experiment:
      device: "auto"     # Automatically detects CUDA/CPU
      ```
2. **Run the Training**
   To download the dataset and train the Autoencoder (Phase 1), simply run:

      ```Bash
      python main.py
      ```
      The script will:

      1. Load the config.

      2. Set the random seed (for reproducibility).

      3. Train the model.

      4. Save the weights to  ```data/checkpoints/ae_mnist.pth ```.


## Methodology & References
This implementation strictly follows the theoretical framework and experimental setup proposed by the Quantum Computing Group at the University of Calabria (Unical).

Specifically, we implement the **Transverse Field Ising Model (TFIM)** Hamiltonian for the reservoir dynamics, as established in the 2025 reference paper.

### Key References

1.  **De Lorenzis, A., Plastina, F., et al.** (2025). *Harnessing quantum extreme learning machines for image classification*. Physical Review Applied.
2.  **Mujal, P., et al.** (2021). *Opportunities in Quantum Reservoir Computing and Extreme Learning Machines*. Advanced Quantum Technologies.
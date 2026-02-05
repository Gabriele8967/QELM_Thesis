import numpy as np

def get_hadamard_matrix():
    
    return np.array([[1, 1], 
                     [1, -1]], dtype=np.complex128) / np.sqrt(2)

def get_phase_matrix(theta):
    return np.array([[1, 0], 
                     [0, np.exp(1j * theta)]], dtype=np.complex128)

def get_ry_matrix(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], 
                     [s, c]], dtype=np.complex128)

def get_x_matrix():
    return np.array([[0, 1], 
                     [1, 0]], dtype=np.complex128)

def get_rx_matrix(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], 
                     [-1j * s, c]], dtype=np.complex128)
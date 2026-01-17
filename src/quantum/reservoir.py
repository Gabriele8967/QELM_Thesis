from qiskit.circuit.library import zz_feature_map, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

def get_quantum_reservoir(n_qbit, n_layers=1):
    feature_map = zz_feature_map(n_qbit,1)
    ansatz = RealAmplitudes(n_qbit, entanglement='full', reps=n_layers)
    circuit = feature_map.compose(ansatz)
    qnn = SamplerQNN(circuit=circuit, input_params=feature_map.parameters, weight_params=ansatz.parameters, input_gradients=True)
    return TorchConnector(qnn)



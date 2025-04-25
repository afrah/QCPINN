## Imports
import numpy as np

# import qiskit
# from qiskit import transpile, assemble
# from qiskit.visualization import *
# from qiskit_aer import Aer
# from qiskit.circuit import Parameter
# from qiskit_algorithms.optimizers import SPSA

import pickle
import sys
import os
from scipy.optimize import minimize


PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../../"))
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qnn.activation_function.activation_function_factory import (
    ActivationFunctionFactory,
)
from qnn.ansatz.abbas import Abbas
from qnn.ansatz.farhi_ansatz import FarhiAnsatz
from qnn.ansatz.sim_circ_13 import SimCirc13
from qnn.ansatz.sim_circ_13_half import SimCirc13Half
from qnn.ansatz.sim_circ_14 import SimCirc14
from qnn.ansatz.sim_circ_14_half import SimCirc14Half
from qnn.ansatz.sim_circ_15 import SimCirc15
from qnn.ansatz.sim_circ_19 import SimCirc19
from qnn.ansatz.alternating_layer_tdcnot_ansatz import AlternatingLayerTDCnotAnsatz

from src.utils.logger import Logging
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
# from sklearn.linear_model import LinearRegression


def custom_polynomial_feature_map(n_qubits, coefficients, degree=2):
    """
    Create a custom polynomial feature map for a given number of qubits.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        coefficients (list): Coefficients for the polynomial terms.
        degree (int): Degree of the polynomial.

    Returns:
        QuantumCircuit: The custom polynomial feature map circuit.
    """
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        # Apply RZ gate with a polynomial function of x_i
        for j, coeff in enumerate(coefficients[:degree]):
            circuit.rz(coeff * (i + 1) ** (j + 1), i)
    return circuit


def custom_ansatz(n_qubits, n_layers):
    """
    Create a custom ansatz circuit with parameterized Rx, Ry rotations and CX gates.

    Parameters:
    - n_qubits: Number of qubits in the circuit.
    - n_layers: Number of layers in the circuit.

    Returns:
    - qc: A parameterized QuantumCircuit.
    """
    qc = QuantumCircuit(n_qubits + 1)

    # Define symbolic parameters for the circuit
    ansatz_params = [Parameter(f"ansatz_{i}") for i in range(4 * n_layers)]

    # Build the circuit layer by layer
    param_index = 0
    for layer in range(n_layers):
        # Apply Rx and Ry gates with parameters
        for qubit in range(n_qubits):
            qc.rx(ansatz_params[param_index], qubit)
            param_index += 1
            qc.ry(ansatz_params[param_index], qubit)
            param_index += 1
        # Add CX gates (entangling layer)
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)

    return qc


class Ansatz:
    def __init__(self, args, n_qubits, backend=None, shots=None):

        # all_qubits = [i for i in range(n_qubits)]
        if args.get("activation") == "null":
            self.circuit = qiskit.QuantumCircuit(n_qubits)
        else:
            self.circuit = qiskit.QuantumCircuit(n_qubits, 2)
        # self._circuit.h(all_qubits)
        self.circuit.barrier()

        # Add feature map
        feature_map_type = args.get(
            "feature_map", "zz_feature_map"
        )  # Default to ZZFeatureMap
        self.add_feature_map(feature_map_type, n_qubits)

        # Add barrier to separate feature map and ansatz
        self.circuit.barrier()
        self.shots = args["shots"]
        self.activation = args["activation"]
        activation_function = ActivationFunctionFactory(self.activation)
        function = activation_function.get()
        ansatz = None
        if args["q_ansatz"] == "farhi":
            ansatz = FarhiAnsatz(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "alternating_layer_tdcnot":
            ansatz = AlternatingLayerTDCnotAnsatz(
                args["layers"], args["q_sweeps"], function
            )
        elif args["q_ansatz"] == "abbas":
            ansatz = Abbas(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "sim_circ_13_half":
            ansatz = SimCirc13Half(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "sim_circ_13":
            ansatz = SimCirc13(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "sim_circ_14_half":
            ansatz = SimCirc14Half(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "sim_circ_14":
            ansatz = SimCirc14(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "sim_circ_15":
            ansatz = SimCirc15(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "sim_circ_19":
            ansatz = SimCirc19(args["layers"], args["q_sweeps"], function)
        elif args["q_ansatz"] == "custom":
            self.circuit_ansatz = custom_ansatz(n_qubits, args["layers"])
        else:
            raise ValueError(f"Unknown ansatz: {args['q_ansatz']}")

        if args["q_ansatz"] != "custom":
            self.circuit_ansatz = ansatz.get_quantum_circuit(n_qubits)
        self.circuit.compose(self.circuit_ansatz, inplace=True)
        # Assign the parameters to the circuit

        # self.circuit.cx(n_qubits - 1, n_qubits)

        # self.circuit.barrier()

        # self.circuit.measure_all()
        # self.circuit.measure(n_qubits - 1, 0)

        self.backend = backend
        self.shots = shots

    def add_feature_map(self, feature_map_type, num_qubits):
        """Add a feature map to the circuit based on args."""

        # num_qubits = self.circuit.num_qubits

        if feature_map_type == "zz_feature_map":
            feature_map = qiskit.circuit.library.ZZFeatureMap(num_qubits, 
                                                              reps=4, 
                                                              entanglement='linear')
        elif feature_map_type == "z_feature_map":
            feature_map = qiskit.circuit.library.ZFeatureMap(num_qubits, reps=4)
        elif feature_map_type == "pauli_feature_map":
            feature_map = qiskit.circuit.library.PauliFeatureMap(
                num_qubits, reps=2, paulis=["X", "Z"]
            )

        elif feature_map_type == "polynomial_feature_map":
            # Use PolynomialFeatureMap
            # Example coefficients: [1.0, 2.0, 1.5] (adjust based on your needs)
            feature_map = custom_polynomial_feature_map(
                num_qubits,
                coefficients=[1.0, 2.0, 1.5],  # Polynomial coefficients
                degree=2,  # Degree of the polynomial
            )

        elif feature_map_type == "custom":
            # Create a custom feature map with RX gates as parameters
            feature_map = qiskit.QuantumCircuit(num_qubits)

            # Define parameters for the RX gates
            rx_params = [Parameter(f"x{i}") for i in range(num_qubits)]

            # Add parameterized RX rotations
            for i, param in enumerate(rx_params):
                feature_map.rx(param, i)

            # # Add entanglement (example: chain CNOTs)
            # for i in range(num_qubits - 1):
            #     feature_map.cx(i, i + 1)

            # Store the RX parameters so they can be bound later
            self.rx_parameters = rx_params
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")

        self.circuit.compose(feature_map, inplace=True)

    def bind_circuit(self, thetas):
        """Bind the parameters to the circuit."""

        # Ensure the number of parameters matches the number of values
        if len(thetas) != len(self.circuit.parameters):
            raise ValueError(
                f"Parameter vector is not the correct size. Expected {len(self.circuit.parameters)}, "
                f"got {len(thetas)}."
            )
        # Create a mapping of parameters to their values
        binding_dict = {
            param: value for param, value in zip(self.circuit.parameters, thetas)
        }

        # Assign parameters to the circuit
        circuit = self.circuit.assign_parameters(binding_dict)
        return circuit

    # def process_results(self, result):
    #     """Process the results of the quantum circuit."""
    #     total_counts = sum(result.values())
    #     normalized_probabilities = {
    #         (int(state, 2) / ((2 ** self.circuit.num_qubits)-1)): (count / total_counts) for state, count in result.items()
    #     }
    #     return normalized_probabilities

    def process_results(self, result):
        """Improved processing of quantum circuit results."""
        total_counts = sum(result.values())
        probabilities = {
                    state: count / total_counts for state, count in result.items()
                }
        # Map binary states to normalized continuous values
        states = [int(state, 2) for state in probabilities.keys()]
        normalized_states = [state / (2 ** self.circuit.num_qubits - 1) for state in states]
        predictions = sum(p * probabilities[state] for p, state in zip(normalized_states, probabilities))

      
        # # Ensure the probabilities are 2D for regression
        # probabilities = np.array(
        #     [count / sum(result.values()) for count in result.values()]
        # )
        return predictions , probabilities

    def run(self, thetas):
        """Run the circuit and return the expectation value of the Z operator."""
        if isinstance(thetas, np.ndarray):
            thetas = thetas.flatten().tolist()

        circuit = self.bind_circuit(thetas)

        try:
            transpiled_circuit = transpile(
                circuit, backend=self.backend, optimization_level=0
            )
            job = self.backend.run(transpiled_circuit, shots=self.shots)
            result = job.result().get_counts()
            # print("result: ", result)
        except Exception as e:
            print(f"Error during backend execution: {e}")
            raise
        expec , probabilities = self.process_results(result)

        # print("probablitites: " ,probabilities.shape )
        # print("probablitites: " ,probabilities )
        return np.array([expec]) , probabilities

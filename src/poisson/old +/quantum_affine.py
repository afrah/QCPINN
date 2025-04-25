import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


class CVNeuralNetwork(nn.Module):
    """
    Implementation of CV Neural Network based on https://arxiv.org/pdf/1806.06871
    Following equation 26 structure: Linear -> Non-linear -> Linear
    """

    def __init__(
        self,
        num_qumodes: int,
        num_layers: int,
        device: str = "cpu",
        cutoff_dim: int = 2,
    ):
        super().__init__()
        self.num_qumodes = num_qumodes
        self.num_layers = num_layers
        self.cutoff_dim = cutoff_dim
        self.device = device
        # Initialize trainable parameters
        # Parameters for interferometers (linear transformations)
        self.theta_1 = nn.Parameter(
            torch.randn(num_layers, num_qumodes, num_qumodes, device=self.device),
            requires_grad=True,
        )
        self.phi_1 = nn.Parameter(
            torch.randn(num_layers, num_qumodes, num_qumodes, device=self.device),
            requires_grad=True,
        )
        self.theta_2 = nn.Parameter(
            torch.randn(num_layers, num_qumodes, num_qumodes, device=self.device),
            requires_grad=True,
        )
        self.phi_2 = nn.Parameter(
            torch.randn(num_layers, num_qumodes, num_qumodes, device=self.device),
            requires_grad=True,
        )

        # Parameters for non-linear transformations
        self.displacement_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device), requires_grad=True
        )
        self.displacement_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device), requires_grad=True
        )
        self.squeezing_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device), requires_grad=True
        )
        self.squeezing_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device), requires_grad=True
        )

        # Add Kerr parameters
        self.kerr_params = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device), requires_grad=True
        )

        # Create quantum device
        self.dev = qml.device(
            "strawberryfields.fock", wires=num_qumodes, cutoff_dim=cutoff_dim
        )

        # Create quantum node
        self.circuit = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

        self.activation = nn.Tanh()
        self._initialize_weights()

    def _interferometer(self, theta: torch.Tensor, phi: torch.Tensor):
        """
        Implements a linear interferometer using beamsplitters and phase shifters
        Following the universal interferometer design that requires both beamsplitters
        and phase shifts for full unitary transformation capability
        """
        n = self.num_qumodes
        # # Apply individual phase shifts first
        for i in range(n):
            qml.Rotation(phi[i, i], wires=i)

        # Then apply beamsplitters between modes
        for i in range(n):
            for j in range(i + 1, n):
                # Parameterized beamsplitter
                qml.Beamsplitter(theta[i, j], phi[i, j], wires=[i, j])
                # Add phase shift after each beamsplitter
                qml.Rotation(phi[j, i], wires=j)

    def _squeeze(self, layer_idx: int):
        """
        Applies non-linear transformations (displacement and squeezing)
        """
        for wire in range(self.num_qumodes):
            # Apply squeezing
            qml.Squeezing(
                self.squeezing_r[layer_idx, wire],
                self.squeezing_phi[layer_idx, wire],
                wires=wire,
            )

    def _dispalcement(self, layer_idx: int):
        """
        Applies non-linear transformations (displacement and squeezing)
        """
        for wire in range(self.num_qumodes):
            # Convert to polar coordinates for displacement
            r = self.displacement_r[layer_idx, wire]
            phi = self.displacement_phi[layer_idx, wire]
            # Apply displacement
            qml.Displacement(r * torch.cos(phi), r * torch.sin(phi), wires=wire)

    def _quantum_circuit(self, inputs):
        """
        Implements the full quantum circuit according to equation 26 based on the architecture described in the paper arxiv:1806.06871.
        """
        # Encode inputs
        for i, input_val in enumerate(inputs):
            # print(f"input_val: {input_val}")
            qml.Displacement(input_val, 0.0, wires=i)

        # Apply layers
        for layer_idx in range(self.num_layers):
            # First linear transformation
            self._interferometer(self.theta_1[layer_idx], self.phi_1[layer_idx])

            # Non-linear transformations
            self._squeeze(layer_idx)

            # Second linear transformation
            self._interferometer(self.theta_2[layer_idx], self.phi_2[layer_idx])

            self._dispalcement(layer_idx)

            # Add Kerr operations at the end for each qumode
            for wire in range(self.num_qumodes):
                qml.Kerr(self.kerr_params[layer_idx, wire], wires=wire)

        # Measure quadratures for all modes
        return [
            qml.expval(qml.NumberOperator(wire)) for wire in range(self.num_qumodes)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum neural network
        """
        batch_size = x.shape[0]
        outputs = []

        for sample in x:
            # print(f"sample: {sample}")
            result = self.circuit(sample)
            outputs.append(result)

        results = torch.stack(outputs)
        # print(f"results: {results.shape}")
        return results

    def _initialize_weights(self):
        """
        Initialize parameters using Xavier uniform initialization
        Provides more bounded and predictable parameter ranges
        """
        gain = nn.init.calculate_gain("tanh")

        # Interferometer parameters (bounded initialization)
        for param in [self.theta_1, self.phi_1, self.theta_2, self.phi_2]:
            fan_in = param.size(1)
            fan_out = param.size(1)
            std = gain * np.sqrt(2.0 / (fan_in + fan_out))
            scale = np.pi / std
            nn.init.xavier_uniform_(param)
            param.data.mul_(scale)

        # Quantum operation amplitudes (need careful bounding)
        for param in [self.displacement_r, self.squeezing_r]:
            nn.init.xavier_uniform_(param, gain=gain)
            param.data.mul_(0.1)  # Controlled initial magnitudes

        # Phase parameters
        for param in [self.displacement_phi, self.squeezing_phi]:
            nn.init.uniform_(param, -np.pi, np.pi)

        nn.init.uniform_(
            self.kerr_params, -0.1, 0.1
        )  # Small initial values for stability

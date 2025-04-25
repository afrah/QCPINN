import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import strawberryfields as sf
from strawberryfields import ops


class CVNeuralNetwork(nn.Module):
    """
    Implementation of CV Neural Network based on https://arxiv.org/pdf/1806.06871
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

        # Standard deviation for normal distribution
        real_sd = 0.1  # Real-numbered parameters from N(0, 0.1)
        angle_range = (0, 2 * np.pi)  # Angles from U(0, 2π)

        # Trainable Parameters Initialization
        self.num_interfermoter_params = int(
            self.num_qumodes * (self.num_qumodes - 1)
        ) + max(1, self.num_qumodes - 1)

        # Interferometer angles (Uniform distribution in [0, 2π])
        self.theta_1 = nn.Parameter(
            torch.rand(num_layers, self.num_interfermoter_params, device=self.device)
            * (2 * np.pi)
        )
        self.theta_2 = nn.Parameter(
            torch.rand(num_layers, self.num_interfermoter_params, device=self.device)
            * (2 * np.pi)
        )

        # Squeezing gate parameters
        self.squeezing_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * real_sd
        )  # N(0, 0.1)
        self.squeezing_phi = nn.Parameter(
            torch.rand(num_layers, num_qumodes, device=self.device) * (2 * np.pi)
        )  # U(0, 2π)

        # Displacement gate parameters
        self.displacement_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * real_sd
        )  # N(0, 0.1)
        self.displacement_phi = nn.Parameter(
            torch.rand(num_layers, num_qumodes, device=self.device) * (2 * np.pi)
        )  # U(0, 2π)

        # Kerr gate parameters
        self.kerr_params = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * real_sd
        )  # N(0, 0.1)

        # Quantum device
        self.dev = qml.device(
            "strawberryfields.fock", wires=num_qumodes, cutoff_dim=cutoff_dim
        )

        # Quantum node
        self.circuit = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum neural network"""
        return torch.stack([self.circuit(sample) for sample in x])

    def _quantum_circuit(self, inputs):
        """Quantum circuit definition"""
        for i, input_val in enumerate(inputs):
            qml.Displacement(input_val, 0.0, wires=i)

        for layer_idx in range(self.num_layers):
            self.qnn_layer(layer_idx)

        return [
            qml.expval(qml.QuadOperator(wires=wire, phi=0.0))
            for wire in range(self.num_qumodes)
        ]

    def qnn_layer(self, layer_idx):
        """CV quantum neural network layer"""
        self.interferometer(self.theta_1[layer_idx])

        for wire in range(self.num_qumodes):
            qml.Squeezing(
                self.squeezing_r[layer_idx, wire],
                self.squeezing_phi[layer_idx, wire],
                wires=wire,
            )

        self.interferometer(self.theta_2[layer_idx])

        for wire in range(self.num_qumodes):
            qml.Displacement(
                self.displacement_r[layer_idx, wire],
                self.displacement_phi[layer_idx, wire],
                wires=wire,
            )
            qml.Kerr(self.kerr_params[layer_idx, wire], wires=wire)

    def interferometer(self, params):
        """Parameterized interferometer"""
        qumode_list = list(range(self.num_qumodes))
        theta = params[: self.num_qumodes * (self.num_qumodes - 1) // 2]

        phi = params[
            (self.num_qumodes * (self.num_qumodes - 1) // 2) : (
                self.num_qumodes * (self.num_qumodes - 1)
            )
        ]
        rphi = params[-self.num_qumodes + 1 :]

        if self.num_qumodes == 1:
            qml.Rotation(rphi[0], wires=0)
            return

        n = 0
        for l in range(self.num_qumodes):
            for k, (q1, q2) in enumerate(zip(qumode_list[:-1], qumode_list[1:])):
                if (l + k) % 2 != 1:
                    qml.Beamsplitter(
                        theta[n], phi[n], wires=[q1, q2]
                    )  # No phi parameter
                    n += 1

        for i in range(max(1, self.num_qumodes - 1)):
            qml.Rotation(rphi[i], qumode_list[i])

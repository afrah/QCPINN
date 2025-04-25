import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import strawberryfields as sf
from strawberryfields import ops


class CVNeuralNetwork(nn.Module):
    def __init__(
        self,
        num_qumodes: int,
        num_layers: int,
        device: str = "cpu",
        cutoff_dim: int = 2,
        use_cubic_phase: bool = True,
        use_cross_kerr: bool = True,
        learnable_input_encoding: bool = True,
    ):
        super().__init__()
        self.num_qumodes = num_qumodes
        self.num_layers = num_layers
        self.cutoff_dim = cutoff_dim
        self.device = device
        self.use_cubic_phase = use_cubic_phase
        self.use_cross_kerr = use_cross_kerr

        # Improved initialization with different scales
        active_sd = 0.0001  # For active gates (squeezing, displacement)
        passive_sd = 0.1  # For passive gates (beam splitters, phases)

        # Interferometer parameters
        self.num_interferometer_params = int(
            self.num_qumodes * (self.num_qumodes - 1)
        ) + max(1, self.num_qumodes - 1)

        self.theta_1 = nn.Parameter(
            torch.randn(num_layers, self.num_interferometer_params, device=self.device)
            * passive_sd
        )
        self.theta_2 = nn.Parameter(
            torch.randn(num_layers, self.num_interferometer_params, device=self.device)
            * passive_sd
        )

        # Enhanced non-linear operations
        self.squeezing_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd
        )
        self.squeezing_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * passive_sd
        )
        self.displacement_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd
        )
        self.displacement_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * passive_sd
        )
        self.kerr_params = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd
        )

        # New parameters for additional operations
        if use_cubic_phase:
            self.cubic_phase = nn.Parameter(
                torch.randn(num_layers, num_qumodes, device=self.device) * active_sd
            )

        if use_cross_kerr:
            self.cross_kerr = nn.Parameter(
                torch.randn(num_layers, num_qumodes, num_qumodes, device=self.device)
                * active_sd
            )

        if learnable_input_encoding:
            self.input_scaling = nn.Parameter(
                torch.ones(num_qumodes, device=self.device)
            )
            self.input_phase = nn.Parameter(
                torch.zeros(num_qumodes, device=self.device)
            )

        # Create quantum device
        self.dev = qml.device(
            "strawberryfields.fock", wires=num_qumodes, cutoff_dim=cutoff_dim
        )

        # Create quantum node
        self.circuit = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        outputs = []
        for sample in x:
            result = self.circuit(sample)
            outputs.append(result)

        results = torch.stack(outputs)
        return results

    def _quantum_circuit(self, inputs):
        # Learnable input encoding
        if hasattr(self, "input_scaling"):
            for i, input_val in enumerate(inputs):
                qml.Displacement(
                    input_val * self.input_scaling[i], self.input_phase[i], wires=i
                )
        else:
            for i, input_val in enumerate(inputs):
                qml.Displacement(input_val, 0.0, wires=i)

        # Iterative quantum layers
        for layer_idx in range(self.num_layers):
            self.qnn_layer(layer_idx)

        # Enhanced measurement scheme
        measurements = []
        for wire in range(self.num_qumodes):
            measurements.append(qml.expval(qml.NumberOperator(wire)))

        return measurements

    def qnn_layer(self, layer_idx):
        """Enhanced CV quantum neural network layer"""
        # First interferometer
        self.interferometer(self.theta_1[layer_idx])

        # Non-linear operations
        for wire in range(self.num_qumodes):
            qml.Squeezing(
                self.squeezing_r[layer_idx, wire],
                self.squeezing_phi[layer_idx, wire],
                wires=wire,
            )

        # Second interferometer
        self.interferometer(self.theta_2[layer_idx])

        # Enhanced non-linear operations
        for wire in range(self.num_qumodes):
            qml.Displacement(
                self.displacement_r[layer_idx, wire],
                self.displacement_phi[layer_idx, wire],
                wires=wire,
            )
            qml.Kerr(self.kerr_params[layer_idx, wire], wires=wire)

            if self.use_cubic_phase:
                qml.CubicPhase(self.cubic_phase[layer_idx, wire], wires=wire)

        if self.use_cross_kerr:
            for i in range(self.num_qumodes):
                for j in range(i + 1, self.num_qumodes):
                    qml.CrossKerr(self.cross_kerr[layer_idx, i, j], wires=[i, j])

    def interferometer(self, params):
        """Parameterized interferometer implementation"""
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
                    qml.Beamsplitter(theta[n], phi[n], wires=[q1, q2])
                    n += 1

        for i in range(max(1, self.num_qumodes - 1)):
            qml.Rotation(rphi[i], qumode_list[i])

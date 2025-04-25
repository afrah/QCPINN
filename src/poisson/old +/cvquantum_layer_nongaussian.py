import os
import torch
import torch.nn as nn
import pennylane as qml


class CVNonGaussianQuantumLayer(nn.Module):
    """True Continuous Variable Quantum Layer combining Gaussian and non-Gaussian operations"""

    def __init__(self, num_qubits: int, num_layers: int, device, cutoff_dim: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = device
        self.cutoff_dim = cutoff_dim

        # Initialize Gaussian operation parameters
        self.displacements = nn.Parameter(
            torch.randn(
                2,
                self.num_layers,
                self.num_qubits,
                2,
                requires_grad=True,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.squeezing = nn.Parameter(
            torch.randn(
                2,
                self.num_layers,
                self.num_qubits,
                2,
                requires_grad=True,
                dtype=torch.float32,
                device=self.device,
            )
        )

        # Beam splitter parameters for mode mixing
        self.beamsplitter = nn.Parameter(
            torch.randn(
                2,
                self.num_layers,
                self.num_qubits - 1,
                2,
                requires_grad=True,
                dtype=torch.float32,
                device=self.device,
            )
        )

        # Create quantum devices using strawberryfields.fock backend
        # This backend supports both Gaussian and non-Gaussian operations
        self.dev_x = qml.device(
            "strawberryfields.fock", wires=self.num_qubits, cutoff_dim=self.cutoff_dim
        )

        self.dev_p = qml.device(
            "strawberryfields.fock", wires=self.num_qubits, cutoff_dim=self.cutoff_dim
        )

        # Create quantum nodes
        self.circuit_X = qml.QNode(
            lambda inputs, d, s, b, w: self._quantum_circuit(inputs, d, s, b, w, qml.X),
            self.dev_x,
            interface="torch",
        )

        self.circuit_P = qml.QNode(
            lambda inputs, d, s, b, w: self._quantum_circuit(inputs, d, s, b, w, qml.P),
            self.dev_p,
            interface="torch",
        )

    def _initialize_weights(self):
        """Apply Xavier initialization to all parameters."""
        for param in [self.displacements, self.squeezing, self.beamsplitter]:
            torch.nn.init.xavier_normal_(param.view(2, -1).T)

    def _apply_gaussian_operations(
        self,
        layer_idx: int,
        wire: int,
        displacements: torch.Tensor,
        squeezing: torch.Tensor,
    ):
        """Apply Gaussian operations."""
        # Displacement
        qml.Displacement(
            displacements[layer_idx, wire, 0],
            displacements[layer_idx, wire, 1],
            wires=wire,
        )

        # Squeezing (magnitude must be positive)
        qml.Squeezing(
            torch.abs(squeezing[layer_idx, wire, 0]),
            squeezing[layer_idx, wire, 1],
            wires=wire,
        )

    def _apply_non_gaussian_operations(self, wire: int):
        """Apply non-Gaussian operations."""
        # Kerr non-linearity - a key non-Gaussian operation in CV
        qml.Kerr(0.1, wires=wire)

        # Cross-Kerr interaction between adjacent modes
        if wire < self.num_qubits - 1:
            qml.CrossKerr(0.05, wires=[wire, wire + 1])

    def _quantum_circuit(
        self, inputs, displacements, squeezing, beamsplitter, wire_idx, measurement_op
    ):
        """Quantum circuit combining Gaussian and non-Gaussian operations"""
        # Encode inputs using displacement
        for i in range(self.num_qubits):
            qml.Displacement(inputs[i], 0.0, wires=i)

        # Apply quantum layers
        for layer in range(self.num_layers):
            # # First apply Gaussian operations
            # for wire in range(self.num_qubits):
            #     self._apply_gaussian_operations(
            #         layer, wire, displacements, squeezing
            #     )

            # Apply non-Gaussian operations
            for wire in range(self.num_qubits):
                self._apply_non_gaussian_operations(wire)

            # Mode mixing with beam splitters
            for wire in range(self.num_qubits - 1):
                qml.Beamsplitter(
                    torch.sigmoid(beamsplitter[layer, wire, 0]),
                    beamsplitter[layer, wire, 1],
                    wires=[wire, wire + 1],
                )

        # Measure quadrature
        return qml.expval(measurement_op(wire_idx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum layer"""
        batch_outputs = []

        for sample in x:
            sample = sample.float()
            measurements = []

            for wire in range(self.num_qubits):
                # X quadrature measurement
                x_measurement = self.circuit_X(
                    sample,
                    self.displacements[0],
                    self.squeezing[0],
                    self.beamsplitter[0],
                    wire,
                )

                # P quadrature measurement
                p_measurement = self.circuit_P(
                    sample,
                    self.displacements[1],
                    self.squeezing[1],
                    self.beamsplitter[1],
                    wire,
                )

                measurements.extend([x_measurement, p_measurement])

            batch_outputs.append(torch.stack(measurements))

        return torch.stack(batch_outputs)

import torch
import torch.nn as nn
import pennylane as qml


class CVQuantumLayer(nn.Module):
    """Continuous Variable Quantum Layer using only Gaussian operations"""

    def __init__(self, num_qubits: int, num_layers: int, device):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = device

        # initialize quantum parameters
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

        self.dev_x = qml.device("default.gaussian", wires=self.num_qubits)
        self.dev_p = qml.device("default.gaussian", wires=self.num_qubits)

        self.circuit_X = qml.QNode(
            self.quantum_circuit_X, self.dev_x, interface="torch"
        )
        self.circuit_P = qml.QNode(
            self.quantum_circuit_P, self.dev_p, interface="torch"
        )

    def _initialize_weights(self):
        """Apply Xavier initialization to all parameters."""

        torch.nn.init.xavier_normal_(
            self.displacements.view(2, self.num_layers, self.num_qubits, 2)
        )

        torch.nn.init.xavier_normal_(
            self.squeezing.view(2, self.num_layers, self.num_qubits, 2)
        )

        torch.nn.init.xavier_normal_(
            self.beamsplitter.view(2, self.num_layers, self.num_qubits - 1, 2)
        )

    def quantum_circuit_X(
        self, inputs, displacements, squeezing, beamsplitter, wire_idx
    ):
        """Quantum circuit measuring X quadrature for a specific wire"""

        for i in range(self.num_qubits):
            qml.Displacement(inputs[i], 0.0, wires=i)

        for layer in range(self.num_layers):

            for wire in range(self.num_qubits):
                #
                qml.Displacement(
                    displacements[layer, wire, 0],
                    displacements[layer, wire, 1],
                    wires=wire,
                )

                qml.Squeezing(
                    torch.abs(squeezing[layer, wire, 0]),
                    squeezing[layer, wire, 1],
                    wires=wire,
                )

            for wire in range(self.num_qubits - 1):

                qml.Beamsplitter(
                    torch.sigmoid(beamsplitter[layer, wire, 0]),
                    beamsplitter[layer, wire, 1],
                    wires=[wire, wire + 1],
                )

        return qml.expval(qml.X(wire_idx))

    def quantum_circuit_P(
        self, inputs, displacements, squeezing, beamsplitter, wire_idx
    ):
        """Quantum circuit measuring P quadrature for a specific wire"""
        for i in range(self.num_qubits):
            qml.Displacement(inputs[i], 0.0, wires=i)

        for layer in range(self.num_layers):
            for wire in range(self.num_qubits):
                qml.Displacement(
                    displacements[layer, wire, 0],
                    displacements[layer, wire, 1],
                    wires=wire,
                )
                qml.Squeezing(
                    torch.abs(squeezing[layer, wire, 0]),
                    squeezing[layer, wire, 1],
                    wires=wire,
                )

            for wire in range(self.num_qubits - 1):
                qml.Beamsplitter(
                    torch.sigmoid(beamsplitter[layer, wire, 0]),
                    beamsplitter[layer, wire, 1],
                    wires=[wire, wire + 1],
                )

        return qml.expval(qml.P(wire_idx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum layer"""
        quantum_outputs = []

        for sample in x:
            sample = sample.float()

            x_measurements = []
            for wire in range(self.num_qubits):
                out_x = self.circuit_X(
                    sample,
                    self.displacements[0],
                    self.squeezing[0],
                    self.beamsplitter[0],
                    wire,
                )
                x_measurements.append(out_x)

            p_measurements = []
            for wire in range(self.num_qubits):
                out_p = self.circuit_P(
                    sample,
                    self.displacements[1],
                    self.squeezing[1],
                    self.beamsplitter[1],
                    wire,
                )
                p_measurements.append(out_p)

            combined = torch.cat(
                [torch.stack(x_measurements), torch.stack(p_measurements)]
            )
            quantum_outputs.append(combined)

        return torch.stack(quantum_outputs)

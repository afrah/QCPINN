import os
import torch
import torch.nn as nn
import pennylane as qml

class CVNonGaussianQuantumLayer(nn.Module):
    """Continuous Variable Quantum Layer using non-Gaussian operations"""
    def __init__(self, num_qubits: int, num_layers: int, device):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = device
        
        # Initialize quantum parameters
        self.rotations = nn.Parameter(torch.randn(2, self.num_layers, 
                                                 self.num_qubits, 3, 
                                                 requires_grad=True,
                                                 dtype=torch.float32, 
                                                 device=self.device))
        
        # Non-Gaussian operation parameters
        self.cx_strengths = nn.Parameter(torch.randn(2, self.num_layers, 
                                                    self.num_qubits - 1, 
                                                    requires_grad=True,
                                                    dtype=torch.float32, 
                                                    device=self.device))
        
        self.t_gates = nn.Parameter(torch.randn(2, self.num_layers, 
                                               self.num_qubits,
                                               requires_grad=True,
                                               dtype=torch.float32, 
                                               device=self.device))
            
        # Create separate quantum devices for each measurement basis
        self.dev_x = qml.device("default.qubit", wires=self.num_qubits)
        self.dev_z = qml.device("default.qubit", wires=self.num_qubits)
        
        # Create separate circuits for different measurement bases
        self.circuit_X = qml.QNode(
            lambda inputs, r, cx, t, w: self._quantum_circuit(inputs, r, cx, t, w, 'X'),
            self.dev_x,
            interface="torch"
        )
        self.circuit_Z = qml.QNode(
            lambda inputs, r, cx, t, w: self._quantum_circuit(inputs, r, cx, t, w, 'Z'),
            self.dev_z,
            interface="torch"
        )

    def _initialize_weights(self):
        """Apply Xavier initialization to all parameters."""
        for param in [self.rotations, self.cx_strengths, self.t_gates]:
            torch.nn.init.xavier_normal_(param.view(2, -1).T)

    def _apply_non_gaussian_layer(self, layer_idx: int, wire: int,
                                rotations: torch.Tensor,
                                cx_strengths: torch.Tensor,
                                t_gates: torch.Tensor):
        """Apply non-Gaussian quantum operations using supported gates."""
        # Apply rotation gates
        qml.RX(rotations[layer_idx, wire, 0], wires=wire)
        qml.RY(rotations[layer_idx, wire, 1], wires=wire)
        qml.RZ(rotations[layer_idx, wire, 2], wires=wire)
        
        # Apply T gate (non-Gaussian phase gate)
        t_strength = torch.sigmoid(t_gates[layer_idx, wire]) * torch.pi
        qml.PhaseShift(t_strength, wires=wire)
        
        # Apply controlled-X gates between adjacent qubits
        if wire < self.num_qubits - 1:
            cx_strength = torch.sigmoid(cx_strengths[layer_idx, wire]) * torch.pi
            qml.CRX(cx_strength, wires=[wire, wire + 1])

    def _quantum_circuit(self, inputs, rotations, cx_strengths, t_gates, 
                        wire_idx, measurement_basis):
        """Generic quantum circuit with non-Gaussian operations"""
        # Encode inputs using rotation gates
        for i in range(self.num_qubits):
            angle = torch.arctan2(inputs[i], torch.ones_like(inputs[i])) * torch.pi
            qml.RY(angle, wires=i)
        
        # Apply quantum layers
        for layer in range(self.num_layers):
            # Apply non-Gaussian operations to each qubit
            for wire in range(self.num_qubits):
                self._apply_non_gaussian_layer(
                    layer, wire, rotations, cx_strengths, t_gates
                )
            
            # Add entangling operations between adjacent qubits
            for wire in range(self.num_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        
        # Return expectation value in the specified basis
        if measurement_basis == 'X':
            return qml.expval(qml.PauliX(wire_idx))
        elif measurement_basis == 'Z':
            return qml.expval(qml.PauliZ(wire_idx))
        else:
            raise ValueError(f"Unsupported measurement basis: {measurement_basis}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum layer"""
        batch_outputs = []
        
        for sample in x:
            sample = sample.float()
            
            # Measure in both X and Z bases for each wire
            measurements = []
            for wire in range(self.num_qubits):
                # X basis measurement
                x_measurement = self.circuit_X(
                    sample, self.rotations[0], self.cx_strengths[0],
                    self.t_gates[0], wire
                )
                
                # Z basis measurement
                z_measurement = self.circuit_Z(
                    sample, self.rotations[1], self.cx_strengths[1],
                    self.t_gates[1], wire
                )
                
                measurements.extend([x_measurement, z_measurement])
            
            batch_outputs.append(torch.stack(measurements))
        
        return torch.stack(batch_outputs)

    def get_circuit_statistics(self, wire: int) -> dict:
        """Get circuit statistics for a specific wire"""
        return {
            'mean_rotation_x': torch.mean(self.rotations[:, :, wire, 0]),
            'mean_rotation_y': torch.mean(self.rotations[:, :, wire, 1]),
            'mean_rotation_z': torch.mean(self.rotations[:, :, wire, 2]),
            'mean_t_gate_strength': torch.mean(torch.sigmoid(self.t_gates[:, :, wire]) * torch.pi),
            'mean_cx_strength': torch.mean(torch.sigmoid(self.cx_strengths[:, :, wire]) * torch.pi) if wire < self.num_qubits - 1 else None
        }
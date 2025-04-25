import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
## Imports 
import os
import sys
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torchviz import make_dot
import matplotlib.pyplot as plt
import pickle
from src.utils.logger import Logging
from src.data.helmholtz_dataset import generate_training_dataset
from src.nn.pde import helmholtz_operator , helmholtz_quantum_operator
from poisson.dv_quantum_layer import DVQuantumLayer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fetch_minibatch(sampler, N):
    X, Y = sampler.sample(N)
    return X, Y


class HybridPINN(nn.Module):
    def __init__(self, args , logger :Logging , classic_network:list, 
                  num_qubits=4, num_layers=2,
                 data=None , 
                  quantum_reps=None , device=DEVICE):
        super().__init__()
        self.logger = logger
        self.device = data.device if data is not None else device
        self.args = args
        self.data = data
        self.draw_quantum_circuit =  True
        # self.targets = targets
        self.classic_network = classic_network # example [2 , 50 , 50 , 50 , 1]
        # Classical layers (enforcing double precision)
        self.activation = nn.Tanh()

        # Classical layers: Dynamically create input, hidden, and output layers
        self.layers = nn.ModuleList()

        # Hidden layers
        for index in range(len(self.classic_network)-1):
            self.layers.append(nn.Linear(self.classic_network[index], 
                                         self.classic_network[index+1]).to(self.device))

        # Quantum parameters
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.quantum_reps = quantum_reps
        self.quantum_layer = DVQuantumLayer( self.num_qubits , self.num_layers, args)
        if self.args["mode"] != "classical":
            # self.params = nn.Parameter(torch.randn(num_qubits, quantum_reps, dtype=torch.float32))
            if self.args["q_ansatz"] == "layered_circuit":
                self.params = nn.Parameter(torch.empty(self.num_layers * self.num_qubits * 2,
                                                        requires_grad=True , dtype=torch.float32))
            elif self.args["q_ansatz"] == "alternating_layer_tdcnot":
                self.params = nn.Parameter(torch.empty(self.num_layers * self.num_qubits * 4,
                                                        requires_grad=True , dtype=torch.float32))

            elif self.args["q_ansatz"] == "sim_circ_19":
                self.params = nn.Parameter(torch.empty(self.num_layers * self.num_qubits * 3,
                                                        requires_grad=True , dtype=torch.float32))
            # self.optimizer = torch.optim.Adam(list(self.parameters()) + [self.params], lr=0.005)
            self.optimizer = torch.optim.Adam(
                [
                    {"params": [self.params], "lr": 0.001},  # quantum parameters
                    {"params": filter(lambda p: p is not self.params, self.parameters()), "lr": 0.005},  # Exclude self.params
                ]
            )
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005 , weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.9, patience=1000
        )


        self.loss_fn = nn.MSELoss()
        self.loss_history = []
        self._initialize_logging()
        self._initialize_weights()


    def _initialize_weights(self):
        """Apply Xavier initialization to all layers."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)  # Or use xavier_normal_ for normal distribution
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Set biases to zero

        if self.args["mode"] in ["quantum", "hybrid"]:
            if self.args["q_ansatz"] == "layered_circuit":
                torch.nn.init.xavier_normal_(self.params.view(self.num_layers, self.num_qubits, 2))
            elif self.args["q_ansatz"] == "alternating_layer_tdcnot":
                torch.nn.init.xavier_normal_(self.params.view(self.num_layers, self.num_qubits, 4))
            elif self.args["q_ansatz"] == "sim_circ_19":
                torch.nn.init.xavier_normal_(self.params.view(self.num_layers, self.num_qubits, 3))
        else:
            self.params = None  # No quantum parameters in classical mode

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        self.logger.print(f"checkpoint path: {self.log_path=}")

    
    def train(self , epochs , batch_size):

        [bcs_sampler, res_sampler] = generate_training_dataset(DEVICE)

        def objective_fn():
            self.optimizer.zero_grad()
            # Fetch boundary mini-batches
            X_bc1_batch, u_bc1_batch = fetch_minibatch(bcs_sampler[0], batch_size)
            X_bc2_batch, u_bc2_batch = fetch_minibatch(bcs_sampler[1], batch_size)
            X_bc3_batch, u_bc3_batch = fetch_minibatch(bcs_sampler[2], batch_size)
            X_bc4_batch, u_bc4_batch = fetch_minibatch(bcs_sampler[3], batch_size)
            # Fetch residual mini-batch
            X_res_batch, f_res_batch = fetch_minibatch(res_sampler, batch_size)
            u_bc1_pred = self.forward(X_bc1_batch)
            u_bc2_pred = self.forward(X_bc2_batch)
            u_bc3_pred = self.forward(X_bc3_batch)
            u_bc4_pred = self.forward(X_bc4_batch)
            
            x1_r, x2_r = X_res_batch[:, 0:1], X_res_batch[:, 1:2]
            [_, r_pred] = helmholtz_quantum_operator(self, x1_r, x2_r)

            loss_r = self.loss_fn(r_pred, f_res_batch)
            loss_bc1 = self.loss_fn(u_bc1_pred, u_bc1_batch)
            loss_bc2 = self.loss_fn(u_bc2_pred, u_bc2_batch)
            loss_bc3 = self.loss_fn(u_bc3_pred, u_bc3_batch)
            loss_bc4 = self.loss_fn(u_bc4_pred, u_bc4_batch)

            loss_bc = loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
            loss = loss_r + 10.0 * (loss_bc)

            if it % self.args["print_every"] == 0:
                self.logger.print(
                    "Iteration: %d, loss_r = %.1e ,  loss_bc = %.1e,  lr = %0.1e"
                    % (
                        it,
                        loss_r.item(),
                        loss_bc.item(),
                        self.optimizer.param_groups[0]["lr"],
                    )
                )
                self.save_state()
            return loss
        
        for it in range(epochs):
            loss = objective_fn()
            # print(f"{loss.item()=}")
            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step(loss)  # Step the learning rate scheduler
            self.loss_history.append(loss.item())

            if it % self.args["print_every"] == 0:
                if self.args["mode"] != "classical":
                    # With this safer version
                    if self.params is not None and self.params.grad is not None:
                        quantum_grad_norm = torch.norm(self.params.grad).item()
                        self.logger.print(f"Quantum params grad norm: {quantum_grad_norm}")

                    for param in self.parameters():
                        if param.grad is not None:
                            classical_grad_norm = torch.norm(param.grad).item()
                            self.logger.print(f"Classical params grad norm: {classical_grad_norm}")


    def forward(self, x):
        """
        Enhanced forward pass implementation supporting classical, quantum, and hybrid modes.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
                
            # Store original device for consistent output
            original_device = x.device
            batch_size = x.shape[0]
            
            # Common classical layer processing
            def process_classical_layers(x):
                for i, layer in enumerate(self.layers[:-1]):
                    x = layer(x)
                    if i > 0 or self.args["mode"] != "quantum":  # Skip activation for first layer in quantum mode
                        x = self.activation(x)
                return self.layers[-1](x)
                
            # Quantum processing function
            def process_quantum(x):
                # Draw circuit only once if enabled
                # print(f"x shape: {x.shape}")
                if self.draw_quantum_circuit:
                    try:
                        self.logger.print("The circuit used in the study:")
                        fig, ax = qml.draw_mpl(self.quantum_layer.qnode)(self.params, x[0])
                        plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                        plt.close()  # Clean up matplotlib resources
                        self.draw_quantum_circuit = False
                    except Exception as e:
                        self.logger.print(f"Failed to draw quantum circuit: {str(e)}")

                # Batch quantum processing with error handling
                quantum_outputs = []
                for sample in x:
                    try:
                        quantum_output = self.quantum_layer.qnode(self.params, sample)
                        quantum_outputs.append(
                            torch.stack(quantum_output).to(
                                dtype=torch.float32, 
                                device=original_device
                            )
                        )
                    except Exception as e:
                        self.logger.print(f"Quantum processing failed for sample: {str(e)}")
                        raise

                # Efficient tensor operations
                quantum_outputs = torch.stack(quantum_outputs)
                quantum_outputs = quantum_outputs.view(batch_size, self.num_qubits)
                
                # Apply post-processing
                quantum_mean = quantum_outputs.mean(dim=1, keepdim=True)
                
                # Optional: Add noise resilience
                if hasattr(self, 'noise_threshold'):
                    quantum_mean = torch.where(
                        torch.abs(quantum_mean) < self.noise_threshold,
                        torch.zeros_like(quantum_mean),
                        quantum_mean
                    )
                    
                return quantum_mean

            # Mode-specific processing
            if self.args["mode"] == "classical":
                output = process_classical_layers(x)
                
            elif self.args["mode"] == "quantum":
                output = process_quantum(process_classical_layers(x))
                
            elif self.args["mode"] == "hybrid":
                classical_output = process_classical_layers(x)
                quantum_output = process_quantum(classical_output)
                
                # Combine classical and quantum outputs
                alpha = getattr(self, 'hybrid_weight', 0.5)
                output = alpha * classical_output + (1 - alpha) * quantum_output
                
            else:
                raise ValueError(f"Unknown mode: {self.args['mode']}")

            # Ensure output is on the correct device
            output = output.to(device=original_device)
            
            # Optional: Add output validation
            if torch.isnan(output).any():
                self.logger.warning("NaN values detected in output")
                output = torch.nan_to_num(output, nan=0.0)
                
            return output

        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            raise

    def set_hybrid_weight(self, alpha):
        """Set the weight for hybrid classical-quantum combination."""
        self.hybrid_weight = torch.clamp(alpha, 0.0, 1.0)

    def set_noise_threshold(self, threshold):
        """Set threshold for noise filtering in quantum outputs."""
        self.noise_threshold = abs(float(threshold))

    # Save model state to a file
    def save_state(self):
        state = {
            "args": self.args,
            "n_qubits":self.num_qubits,
            "num_layers":self.num_layers,
            "classic_network":self.classic_network,
            "parameter_values": self.params.detach().cpu().numpy() if self.args["mode"] != "classical" else None,  # Include only if mode is not classical
            "classic_network_state_dict": self.layers.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "log_path":self.log_path
        }

        model_path = os.path.join(self.log_path, "model.pth")
        quantum_layer_path = os.path.join(self.log_path, "quantum_layer.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(state, f)

        self.logger.print(f"Model state saved to {model_path}")

    # Load model state from a file
    @classmethod
    def load_state(cls, file_path):
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        print(f"Model state loaded from {file_path}")

        return state



    # def forward(self, x):
    #     # Pass through classical layers sequentially with Tanh activation
    #     if self.args["mode"] == "classical":
    #         for layer in self.layers[:-1]:  # Apply activation to all except the output layer
    #             x = self.activation(layer(x))
    #             x = (layer(x))
    #         x = self.layers[-1](x)  # Final layer without activation
    #         return x
    #     elif self.args["mode"] == "quantum":
    #         for layer in self.layers[:-1]:  # Apply activation to all except the output layer
    #             x = (layer(x))
    #         x = self.layers[-1](x)  # Final layer without activation
    #         # Draw the quantum circuit once and save as a PDF (if enabled)
    #         if self.draw_quantum_circuit:
    #             self.logger.print("The circuit used in the study:")
    #             fig, ax = qml.draw_mpl(self.quantum_layer.qnode)(self.params, x[0])
    #             plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
    #             self.draw_quantum_circuit = False

    #         # Quantum forward pass for each sample in the batch
    #         quantum_outputs = []
    #         for sample in x:
    #             quantum_output = self.quantum_layer.qnode(self.params, sample)
    #             quantum_outputs.append(torch.stack(quantum_output).to(dtype=torch.float32))

    #         # print(f"before calling stack{quantum_outputs=}")
    #         # Convert the flattened list to a tensor
    #         quantum_outputs = torch.stack(quantum_outputs)  # Shape: (batch_size * num_qubits,)
    #         # print(f"after calling stack{quantum_outputs=}")

    #         batch_size = x.shape[0]
    #         quantum_outputs = quantum_outputs.view(batch_size, self.num_qubits)

    #         # Optionally take the mean across qubits
    #         quantum_mean = quantum_outputs.mean(dim=1)  # Shape: (batch_size,)
    #         quantum_mean = quantum_mean.unsqueeze(1) 
    #         # Return the quantum mean or full quantum outputs
    #         # print(f"inside forward:{quantum_mean}")
    #         return  quantum_mean # Shape: (batch_size, 1)
        
    #     elif self.args["mode"] == "hybird":
    #         for layer in self.layers[:-1]:  # Apply activation to all except the output layer
    #             x = self.activation(layer(x))
    #             x = (layer(x))
    #         x = self.layers[-1](x)  # Final layer without activation
    #         # Draw the quantum circuit once and save as a PDF (if enabled)
    #         if self.draw_quantum_circuit:
    #             self.logger.print("The circuit used in the study:")
    #             fig, ax = qml.draw_mpl(self.quantum_layer.qnode)(self.params, x[0])
    #             plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
    #             self.draw_quantum_circuit = False

    #         # Quantum forward pass for each sample in the batch
    #         quantum_outputs = []
    #         for sample in x:
    #             quantum_output = self.quantum_layer.qnode(self.params, sample)
    #             quantum_outputs.append(torch.stack(quantum_output).to(dtype=torch.float32))

    #         # print(f"before calling stack{quantum_outputs=}")
    #         # Convert the flattened list to a tensor
    #         quantum_outputs = torch.stack(quantum_outputs)  # Shape: (batch_size * num_qubits,)
    #         # print(f"after calling stack{quantum_outputs=}")

    #         batch_size = x.shape[0]
    #         quantum_outputs = quantum_outputs.view(batch_size, self.num_qubits)

    #         # Optionally take the mean across qubits
    #         quantum_mean = quantum_outputs.mean(dim=1)  # Shape: (batch_size,)
    #         quantum_mean = quantum_mean.unsqueeze(1) 
    #         # Return the quantum mean or full quantum outputs
    #         # print(f"inside forward:{quantum_mean}")
    #         return  quantum_mean # Shape: (batch_size, 1)
        
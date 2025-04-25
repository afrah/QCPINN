import pennylane as qml

# from pennylane import numpy as np
import torch
import torch.nn as nn

## Imports
import os

# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # from torchviz import make_dot
import matplotlib.pyplot as plt

# import pickle
from src.utils.logger import Logging
from src.poisson.dv_quantum_layer import DVQuantumLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DVPDESolver(nn.Module):
    def __init__(self, args, logger: Logging, data=None, device=DEVICE):
        super().__init__()
        self.logger = logger
        self.device = device
        self.args = args
        self.data = data
        self.batch_size = self.args["batch_size"]
        self.num_qubits = self.args["num_qubits"]
        self.epochs = self.args["epochs"]
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.encoding = self.args.get("encoding", "angle")
        self.draw_quantum_circuit_flag = True
        # self.targets = targets
        self.classic_network = self.args["classic_network"]  # [3, 50, 50, 50, 4] #

        if self.encoding == "amplitude":
            self.preprocessor = nn.Sequential(
                nn.Linear(self.classic_network[0], self.classic_network[-2]).to(
                    self.device
                ),
                nn.Tanh(),
                nn.Linear(self.classic_network[-2], self.num_qubits).to(self.device),
            ).to(self.device)
        else:

            self.preprocessor = nn.Sequential(
                nn.Linear(self.classic_network[0], self.classic_network[-2]).to(
                    self.device
                ),
                nn.Tanh(),
                nn.Linear(self.classic_network[-2], self.num_qubits).to(self.device),
            ).to(self.device)

        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, self.classic_network[-2]).to(
                self.device
            ),  
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.classic_network[-1]).to(
                self.device
            ),
        ).to(self.device)

        # 
        self.activation = nn.Tanh()

        # Quantum parameters
        self.num_qubits = args["num_qubits"]
        self.quantum_layer = DVQuantumLayer(self.args)
 
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args["lr"]
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=1000
        )

        self.loss_fn = torch.nn.MSELoss()

        self._initialize_logging()
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier initialization to all layers."""
        for layer in self.preprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(
                    layer.weight
                )  # Or use xavier_normal_ for normal distribution
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Set biases to zero

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        # self.logger.print(f"checkpoint path: {self.log_path=}")

        # # total number of parameters
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"Total number of parameters: {total_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network
        Args:
            x: Spatial coordinates
            t: Time coordinates
        Returns:
            PDE solution values
        """

        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            # Combine inputs
            # Classical preprocessing
            preprocessed = self.preprocessor(x)
            # print(f"preprocessed: {preprocessed.shape}")
            # Quantum processing

            if self.draw_quantum_circuit_flag:
                self.draw_quantum_circuit(preprocessed)
                self.draw_quantum_circuit_flag = False

            quantum_out = self.quantum_layer(preprocessed).to(
                dtype=torch.float32, device=self.device
            )
            # print(f"quantum_out: {quantum_out.shape}")

            classical_out = self.postprocessor(quantum_out)
            # print(f"classical_out: {classical_out.shape}")
            return classical_out

        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            raise

    # Save model state to a file
    def save_state(self):
        state = {
            "args": self.args,
            "classic_network": self.classic_network,
            "quantum_params": self.quantum_layer.state_dict(),
            "preprocessor": self.preprocessor.state_dict(),
            "quantum_layer": self.quantum_layer.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            # "classical_input_scale": self.classical_input_scale.detach().cpu().numpy(),
            # "classical_output_scale": self.classical_output_scale.detach().cpu().numpy(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "log_path": self.log_path,
        }

        model_path = os.path.join(self.log_path, "model.pth")

        with open(model_path, "wb") as f:
            torch.save(state, f)

        self.logger.print(f"Model state saved to {model_path}")

    # Load model state from a file
    @classmethod
    def load_state(cls, file_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cpu")
        with open(file_path, "rb") as f:
            state = torch.load(f, map_location=map_location)
            # state = pickle.load(f)
        print(f"Model state loaded from {file_path}")
        return state

    def draw_quantum_circuit(self, x):
        if self.draw_quantum_circuit_flag:
            try:
                self.logger.print("The circuit used in the study:")
                if self.quantum_layer.params is not None:
                    fig, ax = qml.draw_mpl(self.quantum_layer.circuit)(x[0])
                    plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                    plt.close()  # Clean up matplotlib resources
                    print(f"The circuit is saved in {self.log_path}")
            except Exception as e:
                self.logger.print(f"Failed to draw quantum circuit: {str(e)}")

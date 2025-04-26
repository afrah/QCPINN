import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pennylane as qml


class CVPDESolver(nn.Module):
    """Hybrid Classical-CV Quantum Neural Network for PDE Solving"""

    def __init__(self, args, logger, data=None, device="cpu"):
        super().__init__()

        self.logger = logger
        self.device = device
        self.data = data
        self.input_dim = args["input_dim"]
        self.num_qubits = args["num_qubits"]
        self.hidden_dim = args["hidden_dim"]
        self.output_dim = args["output_dim"]
        self.num_quantum_layers = args["num_quantum_layers"]
        self.epochs = args["epochs"]
        self.args = args
        self.batch_size = args["batch_size"]

        self.log_path = self.logger.get_output_dir()
        self.model_path = os.path.join(self.log_path, "model.pth")
        # Classical preprocessing

        try:
            if self.args.get("class") == "CVNeuralNetwork1":
                from src.nn.CVNeuralNetwork1 import CVNeuralNetwork1

                self.logger.print("Using CVNeuralNetwork1")

                self.quantum_layer = CVNeuralNetwork1(
                    self.num_qubits,
                    self.num_quantum_layers,
                    self.device,
                    cutoff_dim=args["cutoff_dim"],
                )
            elif self.args.get("class") == "CVNeuralNetwork2":
                from src.nn.CVNeuralNetwork2 import CVNeuralNetwork2

                self.logger.print("Using CVNeuralNetwork2 ")

                self.quantum_layer = CVNeuralNetwork2(
                    self.num_qubits,
                    self.num_quantum_layers,
                    self.device,
                    cutoff_dim=args["cutoff_dim"],
                )
            elif self.args.get("class") == "GSRandomCVQNN2":
                from src.nn.CVNeuralNetwork3 import CVNeuralNetwork3

                self.logger.print("Using CVNeuralNetwork3 ")

                self.quantum_layer = CVNeuralNetwork3(
                    self.num_qubits,
                    self.num_quantum_layers,
                    self.device,
                    cutoff_dim=args["cutoff_dim"],
                )
            else:
                from src.nn.CVNeuralNetwork1 import CVNeuralNetwork1

                self.logger.print("Using CVNeuralNetwork1")

                self.quantum_layer = CVNeuralNetwork1(
                    self.num_qubits,
                    self.num_quantum_layers,
                    self.device,
                    cutoff_dim=args["cutoff_dim"],
                )
        except Exception as e:
            self.logger.print(f"Failed to initialize quantum layer: {str(e)}")
            raise

        self.preprocessor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim).to(self.device),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_qubits).to(self.device),
        ).to(self.device)

        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, self.hidden_dim).to(
                self.device
            ),  # 2* for X and P quadratures
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim).to(self.device),
        ).to(self.device)

        if self.args.get("class", "CVNeuralNetwork") == "GSRandomCVQNN2":
            # Lower learning rate
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args["lr"]
            )  # from 5e-3

            # Add learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args["epochs"], weight_decay=0.001
            )

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.9, patience=800, min_lr=1e-6
            )

        self.loss_fn = torch.nn.MSELoss()
        self.loss_history = []
        self.params = None
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
                    nn.init.zeros_(layer.bias)  # Set b

        for layer in self.postprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(
                    layer.weight
                )  # Or use xavier_normal_ for normal distribution
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Set b

    def _initialize_logging(self):
        if self.num_qubits < 2:
            raise ValueError("Number of qubits must be at least 2")
        if self.num_quantum_layers < 1:
            raise ValueError("Number of layers must be at least 1")

        self.log_path = self.logger.get_output_dir()
        # self.logger.print(f"checkpoint path: {self.log_path=}")

        # # total number of parameters
        # total_params = sum(p.numel() for p in self.parameters())
        # self.logger.print(f"Total number of parameters: {total_params}")

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
            "preprocessor": self.preprocessor.state_dict(),
            "quantum_layer": self.quantum_layer.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "model_path": self.model_path,
        }

        with open(self.model_path, "wb") as f:
            torch.save(state, f)
            self.logger.print(f"Model state saved to {self.model_path}")

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
                if self.params is not None:
                    fig, ax = qml.draw_mpl(self.quantum_layer.qnode)(
                        self.params[0], x[0]
                    )
                    plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                    plt.close()  # Clean up matplotlib resources
                    self.draw_quantum_circuit_flag = False
                    self.logger.print(f"The circuit is saved in {self.log_path}")
            except Exception as e:
                self.logger.print(f"Failed to draw quantum circuit: {str(e)}")

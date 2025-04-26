
import torch
import torch.nn as nn
import os
from src.utils.logger import Logging



class ClassicalSolver2(nn.Module):
    def __init__(self, args, logger: Logging, data=None, device="cpu"):
        super().__init__()
        self.logger = logger
        self.device = device
        self.args = args
        self.data = data
        self.batch_size = self.args["batch_size"]
        self.epochs = self.args["epochs"]
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.encoding = self.args.get("encoding", "angle")
        self.draw_quantum_circuit_flag = True
        # self.targets = targets
        self.classic_network = self.args["classic_network"]  # [2, 50,50,1] #

        self.preprocessor = nn.Sequential(
            nn.Linear(self.classic_network[0], self.classic_network[-2]).to(
                self.device
            ),
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.classic_network[-2]).to(
                self.device
            ),
        ).to(self.device)

        self.hidden = nn.Sequential(
            nn.Linear(self.classic_network[-2], self.classic_network[-2]).to(
                self.device
            ),
        ).to(self.device)

        self.postprocessor = nn.Sequential(
            nn.Linear(self.classic_network[-2], self.classic_network[-2]).to(
                self.device
            ),  # 2* for X and P quadratures
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.classic_network[-1]).to(
                self.device
            ),
        ).to(self.device)
        self.activation = nn.Tanh()


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
        for layer in self.preprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(
                    layer.weight
                )  
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        self.logger.print(f"checkpoint path: {self.log_path=}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            # Classical preprocessing
            preprocessed = self.preprocessor(x)
            hidden = self.hidden(self.activation(preprocessed))
            # Postprocessing
            classical_out = self.postprocessor(self.activation(hidden))
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
            "preprocessor": self.preprocessor.state_dict(),
            "hidden_network": self.hidden.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
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
import pennylane as qml
import torch
import torch.nn as nn

import numpy as np
import strawberryfields as sf
from strawberryfields import ops


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
        cheby_order: int = 4,
    ):
        super().__init__()
        self.num_qumodes = num_qumodes
        self.num_layers = num_layers
        self.cutoff_dim = cutoff_dim
        self.device = device
        active_sd = 0.1
        passive_sd = (2 * np.pi)
        self.cheby_order = cheby_order  # Added this line to fix the error

        # Initialize trainable parameters
        # self.weights = self._initialize_weights()

        # Initialize trainable parameters
        # Parameters for interferometers (linear transformations)

        self.num_interfermoter_params = int(
            self.num_qumodes * (self.num_qumodes - 1)
        ) + max(1, self.num_qumodes - 1)

        self.theta_1 = nn.Parameter(
            torch.randn(num_layers, self.num_interfermoter_params, device=self.device)
            * passive_sd,
            requires_grad=True,
        )

        self.theta_2 = nn.Parameter(
            torch.randn(num_layers, self.num_interfermoter_params, device=self.device)
            * passive_sd,
            requires_grad=True,
        )
        self.squeezing_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd,
            requires_grad=True,
        )
        self.squeezing_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * passive_sd,
            requires_grad=True,
        )
        # Parameters for non-linear transformations
        self.displacement_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd,
            requires_grad=True,
        )
        self.displacement_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * passive_sd,
            requires_grad=True,
        )
        # Add Kerr parameters
        self.kerr_params = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd,
            requires_grad=True,
        )

        # Learnable weights for Chebyshev combination
        # self.cheb_weights = nn.Parameter(
        #     torch.ones(cheby_order + 1, num_qumodes, device=device) / (cheby_order + 1)
        # )

        self.cheb_weights = nn.Parameter(
            torch.empty(cheby_order + 1, num_qumodes, device=device)
        )
        nn.init.normal_(
            self.cheb_weights, mean=0.0, std=1 / (num_qumodes * (cheby_order + 1))
        )

        # Create quantum device
        self.dev = qml.device(
            "strawberryfields.fock", wires=num_qumodes, cutoff_dim=cutoff_dim
        )
        # Create quantum node
        self.circuit = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

    def chebyshev_transform(self, x):
        """
        Apply Chebyshev transformation while preserving input dimension
        """
        # Normalize x to [-1, 1]
        x_norm = 2.0 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1.0

        # Initialize list with first two Chebyshev polynomials
        cheb_polys = [torch.ones_like(x_norm)]  # T_0(x) = 1
        cheb_polys.append(x_norm)  # T_1(x) = x

        # Generate higher order Chebyshev polynomials
        for _ in range(2, self.cheby_order + 1):
            # T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
            tn = 2.0 * x_norm * cheb_polys[-1] - cheb_polys[-2]
            cheb_polys.append(tn)

        # Stack polynomials [batch_size, num_features, cheb_order+1]
        cheb_polys = torch.stack(cheb_polys, dim=-1)

        # Weighted combination to maintain original dimension
        # [batch_size, num_features, cheb_order+1] @ [cheb_order+1, num_features]
        transformed = torch.einsum("bnc,cn->bn", cheb_polys, self.cheb_weights)

        return transformed

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # print(f"x: {x.shape}")
        x_features = self.chebyshev_transform(x)
        # print(f"x_features: {x_features.shape}")
        # Vectorize circuit evaluation
        return torch.stack([self.circuit(sample) for sample in x_features])

    def _quantum_circuit(self, inputs):
        # Encode input x into quantum state
        # Encode inputs
        for i, input_val in enumerate(inputs):
            # print(f"input_val: {input_val}")
            qml.Displacement(input_val, 0.0, wires=i)

        # iterative quantum layers
        for layer_idx in range(self.num_layers):
            self.qnn_layer(layer_idx)

        return [
            qml.expval(qml.QuadOperator(wires=wire, phi=0.0)) for wire in range(self.num_qumodes)
        ]

    def qnn_layer(self, layer_idx):
        """CV quantum neural network layer acting on ``N`` modes.

        Args:
            params (list[float]): list of length ``2*(max(1, N-1) + N**2 + n)`` containing
                the number of parameters for the layer
            q (list[RegRef]): list of Strawberry Fields quantum registers the layer
                is to be applied to
        """

        # qumode_list = list(range(self.num_qumodes))

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
        """Parameterised interferometer acting on ``N`` modes.

        Args:
            params (list[float]): list of length ``max(1, N-1) + (N-1)*N`` parameters.

                * The first ``N(N-1)/2`` parameters correspond to the beamsplitter angles
                * The second ``N(N-1)/2`` parameters correspond to the beamsplitter phases
                * The final ``N-1`` parameters correspond to local rotation on the first N-1 modes

            q (list[RegRef]): list of Strawberry Fields quantum registers the interferometer
                is to be applied to
        """

        qumode_list = list(range(self.num_qumodes))

        theta = params[: self.num_qumodes * (self.num_qumodes - 1) // 2]
        phi = params[
            (self.num_qumodes * (self.num_qumodes - 1) // 2) : (
                self.num_qumodes * (self.num_qumodes - 1)
            )
        ]
        rphi = params[-self.num_qumodes + 1 :]

        if self.num_qumodes == 1:
            # the interferometer is a single rotation
            qml.Rotation(rphi[0], wires=0)
            return

        n = 0  # keep track of free parameters

        # Apply the rectangular beamsplitter array
        # The array depth is N
        for l in range(self.num_qumodes):
            for k, (q1, q2) in enumerate(zip(qumode_list[:-1], qumode_list[1:])):
                # skip even or odd pairs depending on layer
                if (l + k) % 2 != 1:
                    qml.Beamsplitter(
                        theta[n],
                        phi[n],
                        wires=[q1, q2],
                    )
                    n += 1

        # apply the final local phase shifts to all modes except the last one
        for i in range(max(1, self.num_qumodes - 1)):
            qml.Rotation(rphi[i], qumode_list[i])

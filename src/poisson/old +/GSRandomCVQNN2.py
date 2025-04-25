import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        gaussian_sigma: float = 0.1,  # Reduced sigma for smaller smoothing
        kernel_size: int = 3,  # Must be odd and smaller than input dimension
    ):
        super().__init__()
        self.num_qumodes = num_qumodes
        self.num_layers = num_layers
        self.cutoff_dim = cutoff_dim
        self.device = device
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = min(
            kernel_size, num_qumodes
        )  # Ensure kernel size isn't too large
        if self.kernel_size % 2 == 0:
            self.kernel_size -= 1  # Make sure kernel size is odd

        active_sd =  0.001
        passive_sd = (0.01* np.pi)

        # Calculate number of interferometer parameters
        self.nnminus1 = (self.num_qumodes * (self.num_qumodes - 1))
        self.nnminus1div2 = self.nnminus1 // 2
        self.num_interfermoter_params = int(self.nnminus1 ) + max(1, self.num_qumodes - 1)

        # # Initialize parameters without requires_grad for random network
        self.theta_1 = nn.Parameter(
            torch.randn(num_layers, 
                        self.num_interfermoter_params, device=self.device)
             * passive_sd, requires_grad=True,
        )

        self.theta_2 = nn.Parameter(
            torch.randn(num_layers, 
                        self.num_interfermoter_params, 
                        device=self.device)
             *  (0.2 * np.pi), requires_grad=True,
        )

        # self.theta_3 = nn.Parameter(
        #     torch.randn(num_layers, self.num_interfermoter_params, device=self.device)
        #      * passive_sd, requires_grad=True,
        # )

        self.squeezing_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, 
                        device=self.device) * active_sd,
            requires_grad=True,
        )

        self.squeezing_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, 
                        device=self.device)  * passive_sd,
            requires_grad=True,
        )

        self.displacement_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, 
                        device=self.device) * active_sd,
            requires_grad=True,
        )

        self.displacement_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, 
                        device=self.device)  * passive_sd, 
            requires_grad=True,
        )

        self.kerr_params = nn.Parameter(
            torch.randn(num_layers, num_qumodes, 
                        device=self.device) * active_sd,
            requires_grad=True,
        )

        # self.rotation_phi1 = nn.Parameter(
        #     torch.randn(num_layers, num_qumodes, device=self.device)  * passive_sd, 
        #     requires_grad=True,
        # )

        # self.rotation_phi2 = nn.Parameter(
        #     torch.randn(num_layers, num_qumodes, device=self.device)  * passive_sd, 
        #     requires_grad=True,
        # )

        self.activation = nn.Tanh()
        # Create Gaussian kernel for smoothing
        # self.gaussian_kernel = self._create_gaussian_kernel()

        # Create quantum device
        self.dev = qml.device(
            "strawberryfields.fock", wires=num_qumodes, cutoff_dim=cutoff_dim
        )

        # Create quantum node
        self.circuit = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

    # def _create_gaussian_kernel(self):
    #     """Create a 1D Gaussian kernel for smoothing"""
    #     x = torch.linspace(
    #         -self.kernel_size // 2, self.kernel_size // 2, self.kernel_size
    #     )
    #     gaussian = torch.exp(-(x**2) / (2 * self.gaussian_sigma**2))
    #     return gaussian / gaussian.sum()

    # def apply_gaussian_smoothing(self, x):
    #     """Apply Gaussian smoothing to the quantum layer output"""
    #     # Reshape input for 1D convolution
    #     # print(f"size before  x.unsqueeze {x.size()}")  # x.unsqueeze torch.Size([4])
    #     x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    #     # print(f"size after x.unsqueeze {x.size()}")  # x.unsqueeze torch.Size([1, 1, 4])
    #     # Pad the input
    #     pad_size = self.kernel_size // 2
    #     x_padded = F.pad(x, (pad_size, pad_size), mode="replicate")
    #     # print(f"size after F.pad {x_padded.size()}")  # F.pad torch.Size([1, 1, 6])

    #     # Apply Gaussian smoothing using 1D convolution
    #     kernel = self.gaussian_kernel.to(x.device, dtype=x.dtype)
    #     kernel = kernel.view(1, 1, -1)  # Reshape for conv1d
    #     smoothed = F.conv1d(x_padded, kernel)
    #     result = smoothed.squeeze(0).squeeze(0)
    #     # print(f"size after squeeze {result.size()}")  # squeeze torch.Size([4])
    #     return result  # Remove batch and channel dimensions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         return torch.stack([((self.circuit(sample))) for sample in x])


    def _quantum_circuit(self, inputs):
        # Encode input x into quantum statevector
        # normalized_inputs = [input_val / (torch.norm(inputs) + 1e-8) for input_val in inputs]
        for i, input_val in enumerate(inputs):
            qml.Displacement(input_val, 0.0, wires=i)
        # iterative quantum layers
        for layer_idx in range(self.num_layers):
            self.qnn_layer(layer_idx)
            
        return  [
                 qml.expval(qml.QuadOperator(wires=wire, phi=0.0)) for wire in range(self.num_qumodes)
            ]

    def qnn_layer(self, layer_idx):
        """CV quantum neural network layer acting on N modes."""
        qumode_list = list(range(self.num_qumodes))
        
        self.interferometer(self.theta_1[layer_idx])

        # # for i in range(max(1, self.num_qumodes - 1)):
        # for i in range(self.num_qumodes):
        #     qml.Rotation(self.rotation_phi1[layer_idx, i], qumode_list[i])

        for wire in range(self.num_qumodes):
            qml.Squeezing(
                self.squeezing_r[layer_idx, wire]* 0.5,0.0,
                # self.squeezing_phi[layer_idx, wire],
                wires=wire,
            )

        self.interferometer(self.theta_2[layer_idx])

        for wire in range(self.num_qumodes):
            qml.Displacement(
                self.displacement_r[layer_idx, wire],0.0,
                # self.displacement_phi[layer_idx, wire],
                wires=wire,
            )
            qml.Kerr(self.kerr_params[layer_idx, wire]* 0.001, wires=wire)

    def interferometer(self, params):
        """Parameterised interferometer acting on N modes."""
        qumode_list = list(range(self.num_qumodes))

        theta = params[: self.nnminus1div2]
        phi = params[self.nnminus1div2 : (self.nnminus1) ]
        # rphi = params[-self.num_qumodes + 1 :]
        rphi = params[-self.num_qumodes :]

        if self.num_qumodes == 1:
            qml.Rotation(rphi[0] , wires=0)
            return

        n = 0
        for l in range(self.num_qumodes):
            for k, (q1, q2) in enumerate(zip(qumode_list[:-1], qumode_list[1:])):
                if (l + k) % 2 != 1:
                    # print(f"beem spliter is used between qubits {q1} and {q2}")
                    qml.Beamsplitter(theta[n] , 0.0,
                                    #  phi[n], 
                                     wires=[q1, q2])
                    n += 1

        # apply the final local phase shifts to all modes except the last one
        for i in range(max(1, self.num_qumodes - 1)):
            qml.Rotation(rphi[i], qumode_list[i])

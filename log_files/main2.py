import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
## Imports 
import os
import sys
import matplotlib.pyplot as plt
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.poisson_dataset import generate_Poisson_training_dataset
from src.utils.logger import Logging
from src.poisson.hybridpinn import HybridPINN
from src.nn.pde import helmholtz_operator
from src.utils.plot_prediction import plt_prediction
from src.data.helmholtz_dataset import u, f

args = {
    "batch_size": 4,
    "epochs": 2000,
    "optimiser": "sgd",
    "lr": 0.0001,
    "seed": 1,
    "print_every": 10,
    "log_path": "../checkpoints",
    "save_model": False,
    "plot": True,
    "batchnorm": False,
    "quantum": True,
    "q_backend": "qasm_simulator",
    "width": 1,
    "encoding": "vector",
    "q_ansatz": "alternating_layer_tdcnot", #options: "alternating_layer_tdcnot", "abbas" , farhi , sim_circ_13_half, sim_circ_13 , sim_circ_14_half, sim_circ_14 , sim_circ_15 ,sim_circ_19 
    "q_sweeps": 1,
    "activation": "null", #options: "null", "partial_measurement_half" , partial_measurement_x
    "shots": None,
    "layers": 4,
    "feature_map": "zz_feature_map"  # z_feature_map , pauli_feature_map ,custom , Specify the feature map here

}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A1 = 1
A2 = 4
LAMBDA = 1.0

# Training data
num_points = 50

dom_coords = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32).to(DEVICE)

t = (
    torch.linspace(dom_coords[0, 0], dom_coords[1, 0], num_points)
    .to(DEVICE)
    .unsqueeze(1)
)

x = (
    torch.linspace(dom_coords[0, 1], dom_coords[1, 1], num_points)
    .to(DEVICE)
    .unsqueeze(1)
)
t, x = torch.meshgrid(t.squeeze(), x.squeeze())
X_star = torch.hstack((t.flatten().unsqueeze(1), x.flatten().unsqueeze(1))).to(DEVICE)


log_path = args["log_path"]
logger = Logging(log_path)

# Initialize the hybrid model
# Example data (ensure double precision)
# SIZE = 4
num_qubits=8
num_layers=8
batch_size = 32
classic_network = [2 ,  num_qubits]
# dataset = generate_Poisson_training_dataset(size=SIZE)
# training_dataset = dataset["interior"]

# inputs = torch.tensor(training_dataset[0] , dtype=torch.float32, requires_grad=True)
# targets = torch.tensor(training_dataset[1] , dtype=torch.float32)

model = HybridPINN(args , logger ,
                   classic_network=classic_network,
                     num_qubits=num_qubits, num_layers=num_layers,
                     data=X_star ,
                     quantum_reps=2)

# Training loop
epochs = 1000
model.train(epochs , batch_size)

# ###############################################  Test data ################################

# Exact solution
u_star = u(X_star, A1, A2)
f_star = f(X_star, A1, A2, LAMBDA)


# Predictions
u_pred_star, f_pred_star = helmholtz_operator(
    model, X_star[:, 0:1], X_star[:, 1:2]
)

# Relative L2 error
error_u = torch.norm(u_pred_star - u_star, 2) / torch.norm(u_star, 2) * 100
error_f = torch.norm(f_pred_star - f_star, 2) / torch.norm(f_star, 2) * 100
logger.print("Relative L2 error_u: {:.2e}".format(error_u.item()))
logger.print("Relative L2 error_f: {:.2e}".format(error_f.item()))


# Plot predictions
plt_prediction(
    logger,
    X_star.cpu().detach().numpy(),
    u_star.cpu().detach().numpy(),
    u_pred_star.cpu().detach().numpy(),
    f_star.cpu().detach().numpy(),
    f_pred_star.cpu().detach().numpy(),
)

torch.norm(model.params).item()


plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid() 

file_path = os.path.join(model.log_path, "loss_history.pdf")
plt.savefig(file_path  , bbox_inches="tight")
plt.show()

plt.close("all" ,)
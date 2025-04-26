import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import pickle


from src.utils.logger import Logging
from src.nn.pde import diffusion_operator
from src.utils.ContourPlotter import ContourPlotter

from src.data.diffusion_dataset import u, r
import src.trainer.diffusion_train as diffusion_train
from src.nn.DVPDESolver import DVPDESolver
from src.nn.CVPDESolver import CVPDESolver
from src.nn.ClassicalSolver import ClassicalSolver

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "hybrid"
num_qubits = 5
output_dim = 1
input_dim = 3
hidden_dim = 50
num_quantum_layers = 1
cutoff_dim = 20
classic_network = [input_dim, hidden_dim, output_dim]


args = {
    "batch_size": 64,
    "epochs": 20000,
    "lr": 0.005,
    "seed": 1,
    "print_every": 100,
    "log_path": "./checkpoints/diffusion",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    "hidden_dim": hidden_dim,
    "num_quantum_layers": num_quantum_layers,
    "classic_network": classic_network,
    "q_ansatz": "sim_circ_19",  # options: "None" for CV and classical , for DV: "alternating_layer_tdcnot", "abbas" , farhi , sim_circ_13_half, sim_circ_13 , sim_circ_14_half, sim_circ_14 , sim_circ_15 ,sim_circ_19
    "mode": mode,
    "activation": "tanh",  # options: "null", "partial_measurement_half" , partial_measurement_x
    "shots": None,  # Analytical gradients enabled
    "problem": "diffusion",
    "solver": "Classical",  # options : "CV", "Classical", "DV"
    "device": DEVICE,
    "method": "None",
    "cutoff_dim": cutoff_dim,  # num_qubits >= cutoff_dim
    "class": "CVNeuralNetwork1",  # options CVNeuralNetwork1, CVNeuralNetwork2, CVNeuralNetwork3
    "encoding": "None",  # options : "ampiltude" , "angle" for DV , none for others
}


log_path = args["log_path"]
logger = Logging(log_path)
# Initialize the hybrid model
# Example data (ensure double precision)
# SIZE = 4


if args["solver"] == "CV":
    model = CVPDESolver(args, logger, DEVICE)
    model.logger.print("Using CV Solver")
elif args["solver"] == "Classical":
    model = ClassicalSolver(args, logger, DEVICE)
    model.logger.print("Using Classical Solver")
else:
    model = DVPDESolver(args, logger, DEVICE)
    model.logger.print("Using DV Solver")
# Training loop
model.logger.print(f"The settings used:")
for key, value in args.items():
    model.logger.print(f"{key} : {value}")


# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
model.logger.print(f"Total number of parameters: {total_params}")

diffusion_train.train(model)

model.save_state()

model.logger.print("Training completed successfuly!")


# plot loss history

plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

file_path = os.path.join(model.log_path, "loss_history.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.show()

plt.close(
    "all",
)

model.logger.print(f"The last loss is: , {model.loss_history[-1]}")


# Testing

NUM_OF_POINTS = 20

dom_coords = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=DEVICE
)

time_ = (
    torch.linspace(dom_coords[0, 0], dom_coords[1, 0], NUM_OF_POINTS)
    .to(DEVICE)
    .unsqueeze(1)
    .to(torch.float32)
)
xfa = (
    torch.linspace(dom_coords[0, 1], dom_coords[1, 1], NUM_OF_POINTS)
    .to(DEVICE)
    .unsqueeze(1)
    .to(torch.float32)
)

yfa = (
    torch.linspace(dom_coords[0, 2], dom_coords[1, 2], NUM_OF_POINTS)
    .to(DEVICE)
    .unsqueeze(1)
    .to(torch.float32)
)

time_, xfa, yfa = torch.meshgrid(time_.squeeze(), xfa.squeeze(), yfa.squeeze())
X_star = torch.hstack(
    (
        time_.flatten().unsqueeze(1),
        xfa.flatten().unsqueeze(1),
        yfa.flatten().unsqueeze(1),
    )
).to(DEVICE)


u_pred, f_pred = diffusion_operator(
    model, X_star[:, 0:1], X_star[:, 1:2], X_star[:, 2:3]
)
if u_pred.is_cuda:
    u_pred = u_pred.cpu()
    f_pred = f_pred.cpu()


u_pred = u_pred.detach().numpy()
f_pred = f_pred.detach().numpy()

# Exact solution
u_analytic = u(X_star).cpu().detach().numpy()
f_analytic = r(X_star).cpu().detach().numpy()

error_u = (
    np.linalg.norm(u_analytic - u_pred, 2) / np.linalg.norm(u_analytic, 2)
) * 100.0
error_f = (
    np.linalg.norm(f_analytic - f_pred, 2) / np.linalg.norm(f_analytic + 1e-9, 2)
) * 100.0

logger.print("Relative L2 error_u: {:.2e}".format(error_u))
logger.print("Relative L2 error_f: {:.2e}".format(error_f))

tstep = NUM_OF_POINTS
xstep = NUM_OF_POINTS
ystep = NUM_OF_POINTS


X = X_star.cpu().detach().numpy()
exact_velocity = u_analytic
exact_force = f_analytic


xf = xfa.reshape(tstep, xstep, ystep).cpu().detach().numpy()  # .reshape(100,100)[0,:]
yf = yfa.reshape(tstep, xstep, ystep).cpu().detach().numpy()  # .reshape(100,100)[:,0]

exact_velocity = exact_velocity.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
exact_force = exact_force.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]

grbf_velocity = u_pred.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
grbf_force = f_pred.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]


# Visualize results
titles = [
    "exact_u",
    "exact_p",
    "pred_u_classic",
    "pred_p_classic",
    "abs_error_u_classic",
    "abs_error_p_classic",
]

nrows_ncols = (3, 2)
values = [99]
xref = 1
yref = 1
model_dirname = model.log_path
img_width = 10
img_height = 10
ticks = 3
fontsize = 7
labelsize = 7
axes_pad = 0.5

visualization_data = [
    exact_velocity,  # exact_u
    exact_force,  # exact_p
    grbf_velocity,  # u_pred_classic
    grbf_force,  # p_pred_classic
    np.abs(exact_velocity - grbf_velocity),  # error_u_classic
    np.abs(exact_force - grbf_force),  # error_p_classic
]

plotter = ContourPlotter(fontsize=7, labelsize=7, axes_pad=0.5)

plotter.draw_contourf_regular_2D(
    time_[:, 0, 0],
    xf[0, :, 0],
    yf[0, 0, :],
    visualization_data,
    titles=titles,
    nrows_ncols=nrows_ncols,
    time_steps=[10],
    xref=1,
    yref=1,
    model_dirname=model_dirname,
    img_width=10,
    img_height=10,
    ticks=3,
)

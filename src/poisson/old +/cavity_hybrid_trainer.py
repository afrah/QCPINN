import pennylane as qml
from pennylane import numpy as np

import torch
import torch.nn as nn

## Imports
import os
import sys
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import h5py
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "./"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import Logging
from src.poisson.dv_solver import DVPDESolver
from src.data.cavity_dataset import CavityDatasetFromFile
from src.utils.cavity_plot_contour import ContourPlotter
from src.utils.logger import Logging
from src.utils.error_metrics import lp_error
from src.poisson import cavity_train
from src.poisson.cv_solver import CVPDESolver
from src.poisson.classical_solver_new import Classical_Solver


mode = "hybrid"
num_qubits = 2
output_dim = 3
input_dim = 3
hidden_dim = 50
num_quantum_layers = 1
cutoff_dim = 20
classic_network = [input_dim, hidden_dim, output_dim]


DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    "batch_size": 64,
    "epochs": 20000,
    "lr": 0.0001,
    "seed": 1,
    "print_every": 100,
    "log_path": "./checkpoints/cavity",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    "hidden_dim": hidden_dim,
    "num_quantum_layers": num_quantum_layers,
    "classic_network": classic_network,
    "q_ansatz": "GSRandomCVQNN2",  # options: "alternating_layer_tdcnot", "abbas" , farhi , sim_circ_13_half, sim_circ_13 , sim_circ_14_half, sim_circ_14 , sim_circ_15 ,sim_circ_19
    "mode": mode,
    "activation": "tanh",  # options: "null", "partial_measurement_half" , partial_measurement_x
    "shots": None,  # Analytical gradients enabled
    "problem": "cavity",
    "solver": "CV", # options : "CV", "Classical", "DV"
    "method": "None",
    "device": DEVICE,
    "cutoff_dim": cutoff_dim,  # num_qubits >= cutoff_dim
    "class": "GSRandomCVQNN2",  # options DVQuantumLayer, GSRandomCVQNN , CVQuantumLayer, Enhanced_CVQuantumLayer
    "encoding": "GSRandomCVQNN2", #options : ampiltude , angle 
}


log_path = args["log_path"]
logger = Logging(log_path)


data_file = "../data/cavity.mat"

obj = CavityDatasetFromFile(data_file, DEVICE)
train_dataloader = obj.__getitem__()


# model = CavityHybridPINN(args , logger ,
#                      data=train_dataloader )

if args["solver"] == "CV":
    model = CVPDESolver(args, logger, train_dataloader, DEVICE)

elif args["solver"] == "Classical":
    model = Classical_Solver(args, logger, train_dataloader, DEVICE)
else:
    model = DVPDESolver(args, logger, train_dataloader, DEVICE)
# Training loop
logger.print(f"The settings used: {args}")
for key, value in args.items():
    logger.print(f"{key} : {value}")

model.logger.print(f"device: {model.device}")

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
model.logger.print(f"Total number of parameters: {total_params}")

cavity_train.train(model)

model.save_state()


model.logger.print("Training completed successfuly!")


### Prediction

skip = 10
tstep = 101
xstep = 100
ystep = 100

TEST_DATA_PKL = "../data/cavity.mat"  # Assuming this is the correct path
data = h5py.File(TEST_DATA_PKL, "r")  # load dataset from matlab

domain = pd.DataFrame(data["cavity_internal"]).T.to_numpy()

#  t = tf.reshape(tstep,N_data)[:,0].T
time_ = (
    domain[:, 0:1].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
)  # (101, 10, 10)
xfa = domain[:, 1:2].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
yfa = domain[:, 2:3].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
ufa = domain[:, 3:4].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
vfa = domain[:, 4:5].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
pfa = domain[:, 5:6].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)

new_shape = domain[:, 0:1].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].shape


test_torch_data = torch.tensor(
    np.concatenate([time_, xfa, yfa], axis=1), dtype=torch.float32
).to(model.device)
with torch.no_grad():
    predictions = model.forward(test_torch_data)
if predictions.is_cuda:
    predictions = predictions.cpu()
u_pred = predictions[:, 0:1].numpy()
v_pred = predictions[:, 1:2].numpy()
p_pred = predictions[:, 2:3].numpy()


text = "RelL2_"
# logger.print("\n Relative L2 ERROR:")
u_error2 = lp_error(u_pred, ufa, (text + "U%"), logger, 2)
v_error2 = lp_error(v_pred, vfa, (text + "V%"), logger, 2)
p_error2 = lp_error(p_pred, pfa, (text + "P%"), logger, 2)

logger.print("Final loss %e" % (model.loss_history[-1]))

logger.print("******************************\n")

logger.print("file directory:", logger.get_output_dir())

tstep = new_shape[0]
xstep = new_shape[1]
ystep = new_shape[2]

txy = [xfa, yfa, time_]
steps = [tstep, xstep, ystep]

#  t = tf.reshape(tstep,N_data)[:,0].T
tf = time_.reshape(tstep, xstep, ystep)
xf = xfa.reshape(tstep, xstep, ystep)
yf = yfa.reshape(tstep, xstep, ystep)


exact_u = ufa.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
exact_v = vfa.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
exact_p = pfa.reshape(tstep, xstep, ystep)  # [1,:].reshape(100,100)[0,:]

u_pred_tanh = u_pred.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
v_pred_tanh = v_pred.reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
p_pred_tanh = p_pred.reshape(tstep, xstep, ystep)  # [1,:].reshape(100,100)[0,:]


tanh_error_u = np.abs(u_pred_tanh - exact_u)
tanh_error_v = np.abs(v_pred_tanh - exact_v)
tanh_error_p = np.abs(p_pred_tanh - exact_p)

data = [
    u_pred_tanh,
    v_pred_tanh,
    p_pred_tanh,
    exact_u,
    exact_v,
    exact_p,
    tanh_error_u,
    tanh_error_v,
    tanh_error_p,
]
titles = [
        f"Exact\n{'$u(x)$'}",
        f"Classical PINN Prediction\n{'$u(x)$'}", 
        "Absolute Error\nCLassic PINN",
        f"QCPINN Prediction\n(angle_cascade)\n{'$u(x)$'}",
        "Absolute Error\nQCPINN(angle_cascade)",
        f"Exact\n{'$v(x)$'}",
        f"Classical PINN Prediction\n{'$v(x)$'}", 
        "Absolute Error\nCLassic PINN",
        f"QCPINN Prediction\n(angle_cascade)\n{'$v(x)$'}",
        "Absolute Error\nQCPINN(angle_cascade)",
        f"Exact\n{'$p(x)$'}",
        f"Classical PINN Prediction\n{'$p(x)$'}", 
        "Absolute Error\nCLassic PINN",
        f"QCPINN Prediction\n(angle_cascade)\n{'$p(x)$'}",
        "Absolute Error\nQCPINN(angle_cascade)",
    ]


plot_xy = False

time_steps = [99]
xy_labels = [r"$x→$", r"$y→$"]

x = xf[0, :, :][..., None]
y = yf[0, :, :][..., None]
nrows_ncols = (3, 3)
values = [99]
xref = 1
yref = 1
model_dirname = model.log_path
img_width = 20
img_height = 4
ticks = 3
X = np.concatenate([x, y], axis=-1)
plotter = ContourPlotter(fontsize=8, labelsize=7, axes_pad=0.65)



visualization_data = [
        exact_u['exact_u'],                  # exact_u
        exact_v['exact_v'],                  # exact_v
        exact_p['exact_p'],              # v_pred_classic
        u_pred_tanh['u_pred_tanh'],        # error_v_classic
        v_pred_tanh['v_pred_tanh'],          # v_pred_angle_cascade
        p_pred_tanh['p_pred_tanh'],    # error_v_angle_cascade
        tanh_error_u['tanh_error_u'],                  # exact_p
        tanh_error_v['tanh_error_v'],              # p_pred_classic
        tanh_error_p['tanh_error_p'],        # error_p_classic
    ]
plotter.draw_contourf_regular_2D(
    tf['tf'][:, 0, 0],
    xf['xf'][0, 0, :],
    yf['yf'][0, :, 0],
    visualization_data,
    titles=titles,
    nrows_ncols=nrows_ncols,
    time_steps=values,
    xref=xref,
    yref=yref,
    model_dirname=model_dirname,
    img_width=img_width,
    img_height=img_height,
    ticks=ticks,    
)

plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

file_path = os.path.join(model.log_path, "loss_history.pdf")
plt.savefig(file_path, bbox_inches="tight")

plt.close(
    "all",
)

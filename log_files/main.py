## Imports 
import numpy as np


from qiskit import transpile, assemble
from qiskit.visualization import * 
from qiskit_aer import Aer

import matplotlib.pyplot as plt
import sys
import os
from math import pi


PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    

from poisson.old.trainer3 import  Quantum_Model

from src.utils.color import model_color
from src.utils.plot_loss import plot_loss_history

# from src.utils.dataset import generate_dataset
from src.utils.plotting import plot_results, plot_training_dataset
from src.utils.logger import Logging
from src.utils.utilities import feature_scaling

SIZE  = 32
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
    "q_ansatz": "sim_circ_13_half", #options: "alternating_layer_tdcnot", "abbas" , farhi , sim_circ_13_half, sim_circ_13 , sim_circ_14_half, sim_circ_14 , sim_circ_15 ,sim_circ_19 
    "q_sweeps": 1,
    "activation": "null", #options: "null", "partial_measurement_half" , partial_measurement_x
    "shots": 1024,
    "layers": 4,
    "feature_map": "zz_feature_map"  # z_feature_map , pauli_feature_map ,custom , Specify the feature map here

}

log_path = args["log_path"]
logger = Logging(log_path)
print("training configuration values:")
for key, value in args.items():
    logger.print(f"key: {key}, value: {value}")

n_qubits = 2

r = 0.5
model = Quantum_Model( logger , args , n_qubits, r )

file_name = os.path.join(model.log_path, "circuit.pdf")
model.quantum_circuit.circuit.draw(output="mpl"  , filename=file_name)


training_dataset = generate_training_dataset(size=SIZE)

training_file = os.path.join(model.log_path, "training_dataset.pdf")

plot_training_dataset(training_dataset   , training_file)

model.train(training_dataset)


data_list = [
    {
        "data": model.loss_history,
        "color": model_color["bspline"],
        "name": "loss",
        "alpha": 0.9,
        "window": 2,
        "show_avg": False,
        "show_lower": False,
    }
]

plot_loss_history(
    data_list,
    os.path.join(model.log_path, "loss_history2.png"),
    y_max=1,
)

SIZE = 4
testing_dataset = generate_testing_dataset(SIZE)

# Define the problem domain and parameters
Lx, Ly = 1.0, 1.0  # Domain size
nx, ny = SIZE, SIZE  # Number of grid points in x and y directions

# Initialize grid
x_axis = np.linspace(0, Lx, nx)
y_axis = np.linspace(0, Ly, ny)

data = testing_dataset["data"]
u_min = testing_dataset["u_min"]
u_max = testing_dataset["u_max"]

loss_history = []  # Initialize `tl`
# for batch, (x, target) in enumerate(zip(data, u)):
    
u_approximate = model.forward(data[0])  # Forward pass
u_approximate = feature_scaling(u_approximate , data[1] , u_min , u_max) # shift_target_to_output_domain( target, self.output_min, self.output_max               )

for index in range(len(data[0])):
    print(f'index: {index} Prediction: {u_approximate[index]:.3f} \t Target: {data[1][index]:.3f}')
    # print(f'index: {index}  \t Target: {testing_exact[index]:.3f}')


# Prediction logic (compare predictions with targets)
mse = np.mean(np.square(u_approximate -  data[1]))/  np.mean(np.square( data[1]))* 100.0 # Count correct predictions

# Calculate and print accuracy
model.logger.print(f'mse: {mse:.2f}%')

title = ["Exact Solution" , "Predicted Solution" ]
u_approximate = np.array(u_approximate).reshape( data[1].shape)

prediction_file = os.path.join(model.log_path, "prediction.pdf")

plot_results( data[1], u_approximate, x_axis, y_axis, title, prediction_file)
# Plot the loss history

plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

loss_history_file = os.path.join(model.log_path, "loss_history.pdf")
plt.savefig(loss_history_file, format="pdf", dpi=300)  # Save in PDF format with high resolution
plt.close("all" , )  # Close the plot to free memory
import numpy as np
import os
import sys
import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn

from src.utils.plot_loss import plot_loss_history
from src.utils.regular_expression import extract_loss_values_cavity

from src.nn.DVPDESolver import DVPDESolver
from src.nn.CVPDESolver import CVPDESolver

from src.utils.logger import Logging

from src.nn.pde import klein_gordon_operator
from src.utils.plot_model_results import plt_model_results
from src.data.klein_gordon_dataset import u, f
from src.nn.ClassicalSolver import ClassicalSolver

log_path = "testing_checkpoints/klein_gordon"
logger = Logging(log_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test data
# Parameters of equations
alpha = torch.tensor(-1.0, device=DEVICE)
beta = torch.tensor(0.0, device=DEVICE)
gamma = torch.tensor(1.0, device=DEVICE)
k = 3
dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)


# Create mesh grid with float32
number_of_points = 20
t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], number_of_points, dtype=np.float32)[
    :, None
]
x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], number_of_points, dtype=np.float32)[
    :, None
]
t, x = np.meshgrid(t, x)

# Convert to PyTorch tensor with float32
X_star = (
    torch.hstack(
        (torch.from_numpy(t.flatten()[:, None]), torch.from_numpy(x.flatten()[:, None]))
    )
    .to(DEVICE)
    .to(torch.float32)
)

u_star = u(X_star)
f_star = f(X_star, alpha, beta, gamma, k)

model_path_angle_cascade = "./models/2025-02-21_11-44-19-583365"  # angle_cascade

model_path_classical = "./models/2025-02-25_17-01-13-323053"  # classical


# old

# model_path_classical = (
#     "./log_files/checkpoints/klein_gordon/2025-02-21_11-30-37-031082"  # classical
# )

MODEL_DIRS = {
    "classical": ("Classical", model_path_classical),
    "angle_cascade": ("DV", model_path_angle_cascade),
}

data = X_star

results = {}
all_loss_history = {}

for model_name, (solver, model_path) in MODEL_DIRS.items():
    model_path = os.path.join(model_path, "model.pth")
    if solver == "DV":
        state = DVPDESolver.load_state(model_path)
        model = DVPDESolver(state["args"], logger, data, DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
        model.logger.print(f"Using DV Solver")

    elif solver == "Classical":
        state = ClassicalSolver.load_state(model_path)

        if "hidden_network" in state:
            from src.nn.ClassicalSolver2 import ClassicalSolver2

            state = ClassicalSolver2.load_state(model_path)
            model = ClassicalSolver2(state["args"], logger, data, DEVICE)
            model.preprocessor.load_state_dict(state["preprocessor"])
            model.hidden.load_state_dict(state["hidden_network"])
            model.postprocessor.load_state_dict(state["postprocessor"])

        else:
            from src.nn.ClassicalSolver import ClassicalSolver

            model = ClassicalSolver(state["args"], logger, data, DEVICE)
            model.preprocessor.load_state_dict(state["preprocessor"])
            model.postprocessor.load_state_dict(state["postprocessor"])

    elif solver == "CV":
        state = CVPDESolver.load_state(model_path)
        model = CVPDESolver(state["args"], logger, data, DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
        model.logger.print(f"Using CV Solver")
    else:
        raise ValueError(f"Unknown solver {solver}")

    model.logger = logger

    model.logger.print(f"Total number of iterations : {len(state['loss_history'])}")
    model.logger.print(f"The final loss : {state['loss_history'][-1]}")

    model.model_path = logger.get_output_dir()

    u_pred_star, f_pred_star = klein_gordon_operator(
        model, X_star[:, 0:1], X_star[:, 1:2]
    )

    u_pred = u_pred_star.cpu().detach().numpy()
    f_pred = f_pred_star.cpu().detach().numpy()
    u_exact = u_star.cpu().detach().numpy()
    f_exact = f_star.cpu().detach().numpy()
    X = X_star.cpu().detach().numpy()

    error_u = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2) * 100
    error_f = np.linalg.norm(f_pred - f_exact, 2) / np.linalg.norm(f_exact, 2) * 100
    logger.print("Relative L2 error_u: {:.2e}".format(error_u.item()))
    logger.print("Relative L2 error_f: {:.2e}".format(error_f.item()))

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    model.logger.print(f"Total number of parameters: {total_params}")

    results[model_name] = (u_pred, f_pred)
    all_loss_history[model_name] = state["loss_history"]

    del model

plt_model_results(
    logger,
    X,
    u_exact,
    f_exact,
    results,
)


plot_loss_history(
    all_loss_history,
    os.path.join(logger.get_output_dir(), "loss_history_klein_gordon.png"),
    y_max=2000,
    legend=False,
)

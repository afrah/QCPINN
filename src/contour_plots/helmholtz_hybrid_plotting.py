
import numpy as np
import os
import sys
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "./"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.poisson.dv_solver import DVPDESolver
from src.poisson.cv_solver import CVPDESolver

# from src.utils.color import model_color
# from src.utils.plot_loss import plot_loss_history
from src.utils.logger import Logging

from src.nn.pde import helmholtz_operator
from src.utils.plot_model_results import plt_model_results
from src.data.helmholtz_dataset import u, f
# from src.poisson.classical_solver import Classical_Solver
from src.poisson.classical_solver import Classical_Solver

log_path = "testing_checkpoints/helmholtz"
logger = Logging(log_path)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test data
num_points = 80
A1 = 1
A2 = 4
LAMBDA = 1.0

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


# Exact solution
u_star = u(X_star, A1, A2)
f_star = f(X_star, A1, A2, LAMBDA)

model_path_angle_cascade = (
    "./final_models/2025-02-06_19-25-14-069398"  # angle_cascade
)

# model_path_classical = (
#     "./log_files/checkpoints/helmholtz/2025-02-25_16-06-27-412553"  # classical
# )


# #old
model_path_classical = (
    "./final_models/2025-02-09_00-01-28-238904"  # classical
)

MODEL_PATHS = {
    "classical": ("classical", model_path_classical),
    "angle_cascade": ("dv", model_path_angle_cascade),
}

data = X_star

results  = {}
for model_name, (solver, model_path) in MODEL_PATHS.items():
    if solver == "dv":
        state = DVPDESolver.load_state(os.path.join(model_path, "model.pth"))
        model = DVPDESolver(state["args"], logger, data, DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.logger.print(f"Using DV Solver")
    elif solver == "classical":
        state = Classical_Solver.load_state(os.path.join(model_path , "model.pth"))
        model = Classical_Solver(state["args"], logger)    
        model.preprocessor.load_state_dict(state["preprocessor"])
        # model.hidden.load_state_dict(state["hidden_network"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.logger.print(f"Using classical Solver")

    elif solver == "cv":
        state = CVPDESolver.load_state(os.path.join(model_path, "model.pth"))
        model = CVPDESolver(state["args"], logger, data, DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.logger.print(f"Using CV Solver")
    else:
        raise ValueError(f"Unknown solver {solver}")

    model.logger = logger

    model.logger.print(f"Total number of iterations : {len(state['loss_history'])}")
    model.logger.print(f"The final loss : {state['loss_history'][-1]}")
    model.logger.print(f"Total number of parameters : {sum(p.numel() for p in model.parameters())}")

    model.model_path = logger.get_output_dir()

    # Predictions
    u_pred_star, f_pred_star = helmholtz_operator(model, X_star[:, 0:1], X_star[:, 1:2])

        # Relative L2 error
    error_u = torch.norm(u_pred_star - u_star, 2) / torch.norm(u_star, 2) * 100
    error_f = torch.norm(f_pred_star - f_star, 2) / torch.norm(f_star, 2) * 100 
    logger.print("Relative L2 error_u: {:.2e}".format(error_u.item()))
    logger.print("Relative L2 error_f: {:.2e}".format(error_f.item()))


    u_pred = u_pred_star.cpu().detach().numpy()
    f_pred = f_pred_star.cpu().detach().numpy()

    results[model_name] = (u_pred, f_pred)
    del model

# Plot predictions

u_exact = u_star.cpu().detach().numpy()
f_exact = f_star.cpu().detach().numpy()
X = X_star.cpu().detach().numpy()
    
plt_model_results(
    logger,
    X,
    u_exact,
    f_exact,
    results,
)
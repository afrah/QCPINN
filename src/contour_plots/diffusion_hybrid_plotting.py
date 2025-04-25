## Imports
import numpy as np
import os
import sys
import torch

## Imports
import os
import sys

from src.utils.plot_loss import plot_loss_history


PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "./"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.poisson.dv_solver import DVPDESolver
from src.poisson.cv_solver import CVPDESolver

# from src.utils.color import model_color
# from src.utils.plot_loss import plot_loss_history
from src.utils.logger import Logging

from src.nn.pde import diffusion_operator
from src.data.diffusion_dataset import u, r
from src.poisson.classical_solver_new import Classical_Solver
from src.utils.cavity_plot_contour import ContourPlotter

log_path = "testing_checkpoints/diffusion"
logger = Logging(log_path)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test data

NUM_OF_POINTS = 10

dom_coords = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cpu"
)

time_ = (
    torch.linspace(dom_coords[0, 0], dom_coords[1, 0], NUM_OF_POINTS)
    .to("cpu")
    .unsqueeze(1)
    .to(torch.float32)
)
xfa = (
    torch.linspace(dom_coords[0, 1], dom_coords[1, 1], NUM_OF_POINTS)
    .to("cpu")
    .unsqueeze(1)
    .to(torch.float32)
)

yfa = (
    torch.linspace(dom_coords[0, 2], dom_coords[1, 2], NUM_OF_POINTS)
    .to("cpu")
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
).to("cpu")




# Exact solution
u_analytic = u(X_star).cpu().detach().numpy()
f_analytic = r(X_star).cpu().detach().numpy()



model_path_angle_cascade = (
    "./models/2025-02-21_12-00-52-045180"  # angle_cascade
)

model_path_classical = (
    "./models/2025-02-25_17-03-12-608017"  # classical
)


# #old
# model_path_classical = (
#     "./log_files/checkpoints/diffusion/2025-02-21_11-48-39-023832"  # classical
# )


MODEL_PATHS = {
    "classical": ("classical", model_path_classical),
    "angle_cascade": ("dv", model_path_angle_cascade),
}

data = X_star

results  = {}
all_loss_history = {}

for model_name, (solver, model_path) in MODEL_PATHS.items():
    if solver == "dv":
        state = DVPDESolver.load_state(os.path.join(model_path, "model.pth"))
        model = DVPDESolver(state["args"], logger, data, DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
        model.logger.print(f"Using DV Solver")
    elif solver == "classical":
        state = Classical_Solver.load_state(os.path.join(model_path , "model.pth"))
        model = Classical_Solver(state["args"], logger)    
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.hidden.load_state_dict(state["hidden_network"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.logger.print(f"Using classical Solver")

    elif solver == "cv":
        state = CVPDESolver.load_state(os.path.join(model_path, "model.pth"))
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

    # Predictions
    u_pred_star, f_pred_star = diffusion_operator(
        model, X_star[:, 0:1], X_star[:, 1:2], X_star[:, 2:3]
    )

    u_pred = u_pred_star.cpu().detach().numpy()
    f_pred = f_pred_star.cpu().detach().numpy()
    X = X_star.cpu().detach().numpy()

        # Relative L2 error
    error_u = np.linalg.norm(u_pred - u_analytic, 2) / np.linalg.norm(u_analytic, 2) * 100
    error_f = np.linalg.norm(f_pred - f_analytic, 2) / np.linalg.norm(f_analytic, 2) * 100
    logger.print("Relative L2 error_u: {:.2e}".format(error_u.item()))
    logger.print("Relative L2 error_f: {:.2e}".format(error_f.item()))


    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    model.logger.print(f"Total number of parameters: {total_params}")

    log_path = model.log_path
    results[model_name] = (u_pred, f_pred)
    all_loss_history[model_name] =state['loss_history']

    del model

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

classic_velocity = results["classical"][0].reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
classic_force = results["classical"][1].reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]

angle_cascade_velocity = results["angle_cascade"][0].reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
angle_cascade_force = results["angle_cascade"][1].reshape(tstep, xstep, ystep)  # .reshape(100,100)[0,:]
# Visualize results
titles = [
    "Exact solution $u(x)$",
    "PINN prediction $\\hat{u}(x)$",
    "PINN error",
    "QCPINN prediction $\\hat{u}(x)$",
    "QCPINN error",
    "Exact solution $f(x)$",
    "PINN prediction $\\hat{f}(x)$",
    "PINN error",
    "QCPINN prediction $\\hat{f}(x)$",
    "QCPINN error",
]

nrows_ncols = (2, 5)
xref = 1
yref = 1
model_dirname = log_path
img_width = 20
img_height = 4
ticks = 3

visualization_data = [
   exact_velocity,
    classic_velocity,            
    np.abs(exact_velocity - classic_velocity),      
    angle_cascade_velocity,              
    np.abs(exact_velocity - angle_cascade_velocity),  
    exact_force,                  
    classic_force,             
     np.abs(exact_force - classic_force),       
    angle_cascade_force,          
    np.abs(exact_force - angle_cascade_force), 
]

plotter = ContourPlotter(fontsize=8, labelsize=7, axes_pad=0.65)

plotter.draw_contourf_regular_2D(
    time_[:, 0, 0],
    xf[0, :, 0],
    yf[0, 0, :],
    visualization_data,
    titles=titles,
    nrows_ncols=nrows_ncols,
    time_steps=[NUM_OF_POINTS-1],
    xref=xref,
    yref=yref,
    model_dirname=model_dirname,
    img_width=img_width,
    img_height=img_height,
    ticks=ticks,    
)

# Plot loss history
plot_loss_history(
    all_loss_history,
    os.path.join(logger.get_output_dir(), "loss_history_diffusion.png"),
    y_max=10,
    legend=True,
)

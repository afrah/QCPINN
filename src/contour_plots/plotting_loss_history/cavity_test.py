import sys
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import h5py
import pandas as pd

from src.utils.logger import Logging
from src.utils.plot_loss import plot_loss_history
from src.utils.color import model_color
from src.poisson.cv_solver import CVPDESolver
from src.poisson.dv_solver import DVPDESolver
from src.poisson.classical_solver_new import Classical_Solver
from src.utils.error_metrics import lp_error

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "./"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_DATA_PKL = "./data/cavity.mat"
TEST_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "testing_checkpoints/cavity")

def setup_logger():
    """Initialize logging"""
    logger = Logging(TEST_CHECKPOINT_PATH)
    return logger, logger.get_output_dir()

def load_and_process_data(skip=20, tstep=101, xstep=100, ystep=100):
    """Load and preprocess the cavity flow data"""
    data = h5py.File(TEST_DATA_PKL, "r")
    domain = pd.DataFrame(data["cavity_internal"]).T.to_numpy()
    
    # Process each component
    def reshape_data(data_slice):
        return (data_slice.reshape(tstep, xstep, ystep)
                [:, ::skip, ::skip].reshape(-1, 1))
    
    time_ = reshape_data(domain[:, 0:1])
    xfa = reshape_data(domain[:, 1:2])
    yfa = reshape_data(domain[:, 2:3])
    ufa = reshape_data(domain[:, 3:4])
    vfa = reshape_data(domain[:, 4:5])
    pfa = reshape_data(domain[:, 5:6])
    
    return time_, xfa, yfa, ufa, vfa, pfa

def get_model_paths():
    """Define paths for different model checkpoints"""
    base_path = "./log_files/checkpoints/cavity"
    return {
        f"{base_path}/2025-02-12_16-32-09-851527": ("dv", "angle_layered"),
        f"{base_path}/2025-02-06_19-28-34-814985": ("dv", "angle_cascade"),
        f"{base_path}/2025-02-06_19-28-52-910332": ("dv", "angle_cross_mesh"),
        f"{base_path}/2025-02-06_19-27-57-462145": ("dv", "angle_alternate"),
        f"{base_path}/2025-02-12_16-32-09-865339": ("dv", "amp_layered"),
        f"{base_path}/2025-02-06_18-44-40-359259": ("dv", "amp_alternate"),
        f"{base_path}/2025-02-06_18-29-51-200273": ("dv", "amp_cross_mesh"),
        f"{base_path}/2025-02-06_18-41-52-938544": ("dv", "amp_cascade"),
        # f"{base_path}/2025-02-06_22-52-49-345794": ("cv", "cv"),
        # f"{base_path}/2025-02-09_19-13-42-309529": ("cv", "gcv"),
        f"{base_path}/2025-02-25_17-21-36-221407": ("classical", "classical")
    }

    #old classical
        # f"{base_path}/2025-02-25_11-35-11-375027": ("classical", "classical")


def load_model(model_path, solver_type, logger):
    """Load appropriate solver model based on type"""
    model_file = os.path.join(model_path, "model.pth")
    
    if solver_type == "cv":
        state = CVPDESolver.load_state(model_file)
        model = CVPDESolver(state["args"], logger)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
        model.postprocessor.load_state_dict(state["postprocessor"])

    elif solver_type == "classical":
        state = Classical_Solver.load_state(model_file)
        model = Classical_Solver(state["args"], logger)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.hidden.load_state_dict(state["hidden_network"])
        model.postprocessor.load_state_dict(state["postprocessor"])

    else:  # dv
        state = DVPDESolver.load_state(model_file)
        model = DVPDESolver(state["args"], logger)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])

    model.logger.print(f"Model loaded successfully from {model_path}")
    model.logger.print(f"Using {solver_type} Solver")
    model.logger.print(f"Total number of iterations : {len(state['loss_history'])}")
    model.logger.print(f"The final loss : {state['loss_history'][-1]}")

    model.logger = logger
    return model, state

def evaluate_model(model, input_data, ufa, vfa, pfa, logger):
    """Evaluate model performance and compute errors"""
    with torch.no_grad():
        predictions = model.forward(input_data)
    if predictions.is_cuda:
        predictions = predictions.cpu()
        
    u_pred = predictions[:, 0:1].numpy()
    v_pred = predictions[:, 1:2].numpy()
    p_pred = predictions[:, 2:3].numpy()
    
    # Calculate errors
    text = "RelL2_"
    u_error2 = lp_error(u_pred, ufa, (text + "U%"), logger, 2)
    v_error2 = lp_error(v_pred, vfa, (text + "V%"), logger, 2)
    p_error2 = lp_error(p_pred, pfa, (text + "P%"), logger, 2)
    
    return u_error2, v_error2, p_error2

def create_plot_config(all_loss_history):
    """Create plot configuration for all models"""
    PLOT_STYLES = {
        "amp_layered": "-",      # Solid line
        "angle_layered": "-",    # Solid line
        "amp_cascade": "-",    # Solid line
        "amp_alternate": "-",    # Solid line
        "amp_cross_mesh": "-",     # Solid line
        "angle_cross_mesh": "-",   # Solid line
        "angle_alternate": "-",  # Solid line
        "angle_cascade": "-",  # Solid line
        "cv": "-",              # Solid line
        "gcv": "-",             # Solid line
        "classical": "-"        # Solid line
    }
    
    return [
        {
            "data": all_loss_history[model_name],
            "color": model_color[model_name],
            "name": "CV" if model_name == "cv" else model_name,
            "alpha": 1.0,
            "window": 100,
            "show_avg": False,
            "show_lower": False,
            "linestyle": style,
            "linewidth": 3.0 if style == ":" else 3.5
        }
        for model_name, style in PLOT_STYLES.items()
        if model_name in all_loss_history
    ]

def main():
    # Setup
    logger, model_dirname = setup_logger()
    
    # Load and process data
    time_, xfa, yfa, ufa, vfa, pfa = load_and_process_data()
    
    # Prepare input data
    test_data = np.concatenate([time_, xfa, yfa], axis=1)
    
    # Process all models
    all_loss_history = {}
    model_paths = get_model_paths()
    
    for model_path, (solver_type, model_name) in model_paths.items():
        model, state = load_model(model_path, solver_type, logger)
        
        # Log model info
        logger.print("******************************\n")
        logger.print("******************************\n")

        logger.print(f"Method used: {model_name}")
        logger.print(f"Total iterations: {len(state['loss_history'])}")
        logger.print(f"Final loss: {state['loss_history'][-1]}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.print(f"Total number of parameters: {total_params}")

        # Evaluate model
        input_tensor = torch.tensor(test_data, dtype=torch.float32).to(model.device)
        # evaluate_model(model, input_tensor, ufa, vfa, pfa, logger)
        
        logger.print(f"File directory: {model_path}")
        
        all_loss_history[model_name] = state["loss_history"][:12500]
        logger.print("******************************\n")
        logger.print("******************************\n")

    # Create and save plot
    # plot_config = create_plot_config(all_loss_history)
    plot_loss_history(
        all_loss_history,
        os.path.join(logger.get_output_dir(), "loss_history_cavity.png"),
        y_max=7,
        legend=True
    )

if __name__ == "__main__":
    main()
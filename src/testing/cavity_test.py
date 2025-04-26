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
from src.nn.DVPDESolver import DVPDESolver
from src.nn.CVPDESolver import CVPDESolver
from src.nn.ClassicalSolver import ClassicalSolver
from src.utils.error_metrics import lp_error


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        return data_slice.reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)

    time_ = reshape_data(domain[:, 0:1])
    xfa = reshape_data(domain[:, 1:2])
    yfa = reshape_data(domain[:, 2:3])
    ufa = reshape_data(domain[:, 3:4])
    vfa = reshape_data(domain[:, 4:5])
    pfa = reshape_data(domain[:, 5:6])

    return time_, xfa, yfa, ufa, vfa, pfa


def get_model_paths():
    """Define paths for different model checkpoints"""
    base_path = "./models/cavity_amplitude"
    return {
        f"{base_path}/2025-02-12_16-32-09-851527": ("DV", "angle_layered"),
        f"{base_path}/2025-02-06_19-28-34-814985": ("DV", "angle_cascade"),
        f"{base_path}/2025-02-06_19-28-52-910332": ("DV", "angle_cross_mesh"),
        f"{base_path}/2025-02-06_19-27-57-462145": ("DV", "angle_alternate"),
        f"{base_path}/2025-02-12_16-32-09-865339": ("DV", "amp_layered"),
        f"{base_path}/2025-02-06_18-44-40-359259": ("DV", "amp_alternate"),
        f"{base_path}/2025-02-06_18-29-51-200273": ("DV", "amp_cross_mesh"),
        f"{base_path}/2025-02-06_18-41-52-938544": ("DV", "amp_cascade"),
        # f"{base_path}/2025-02-06_22-52-49-345794": ("cv", "cv"),
        # f"{base_path}/2025-02-09_19-13-42-309529": ("cv", "gcv"),
        f"{base_path}/2025-02-25_17-21-36-221407": ("Classical", "classical"),
    }

    # old classical
    # f"{base_path}/2025-02-25_11-35-11-375027": ("classical", "classical")


def load_model(model_dir, solver_type, logger):
    """Load appropriate solver model based on type"""
    model_path = os.path.join(model_dir, "model.pth")

    if solver_type == "CV":
        state = CVPDESolver.load_state(model_path)
        model = CVPDESolver(state["args"], logger)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
        model.postprocessor.load_state_dict(state["postprocessor"])

    elif solver_type == "Classical":
        from src.nn.ClassicalSolver import ClassicalSolver

        state = ClassicalSolver.load_state(model_path)

        if "hidden_network" in state:
            from src.nn.ClassicalSolver2 import ClassicalSolver2

            state = ClassicalSolver2.load_state(model_path)
            model = ClassicalSolver2(state["args"], logger)
            model.preprocessor.load_state_dict(state["preprocessor"])
            model.hidden.load_state_dict(state["hidden_network"])
            model.postprocessor.load_state_dict(state["postprocessor"])

        else:
            from src.nn.ClassicalSolver import ClassicalSolver

            model = ClassicalSolver(state["args"], logger)
            model.preprocessor.load_state_dict(state["preprocessor"])
            model.postprocessor.load_state_dict(state["postprocessor"])

    else:  # DV
        state = DVPDESolver.load_state(model_path)
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
        "amp_layered": "-",
        "angle_layered": "-",
        "amp_cascade": "-",
        "amp_alternate": "-",
        "amp_cross_mesh": "-",
        "angle_cross_mesh": "-",
        "angle_alternate": "-",
        "angle_cascade": "-",
        "cv": "-",
        "gcv": "-",
        "classical": "-",
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
            "linewidth": 3.0 if style == ":" else 3.5,
        }
        for model_name, style in PLOT_STYLES.items()
        if model_name in all_loss_history
    ]


def main():

    logger, model_dirname = setup_logger()

    time_, xfa, yfa, ufa, vfa, pfa = load_and_process_data()

    test_data = np.concatenate([time_, xfa, yfa], axis=1)

    all_loss_history = {}
    model_paths = get_model_paths()

    for model_path, (solver_type, model_name) in model_paths.items():
        model, state = load_model(model_path, solver_type, logger)

        logger.print("******************************\n")
        logger.print("******************************\n")

        logger.print(f"Method used: {model_name}")
        logger.print(f"Total iterations: {len(state['loss_history'])}")
        logger.print(f"Final loss: {state['loss_history'][-1]}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.print(f"Total number of parameters: {total_params}")

        input_tensor = torch.tensor(test_data, dtype=torch.float32).to(model.device)
        # evaluate_model(model, input_tensor, ufa, vfa, pfa, logger)

        logger.print(f"File directory: {model_path}")

        all_loss_history[model_name] = state["loss_history"][:12500]
        logger.print("******************************\n")
        logger.print("******************************\n")

    plot_loss_history(
        all_loss_history,
        os.path.join(logger.get_output_dir(), "loss_history_cavity.png"),
        y_max=7,
        legend=True,
    )


if __name__ == "__main__":
    main()

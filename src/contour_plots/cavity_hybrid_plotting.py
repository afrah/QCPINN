import numpy as np
import os
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "./"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# from src.utils.color import model_color
# from src.utils.plot_loss import plot_loss_history
from src.utils.logger import Logging
from src.utils.cavity_plot_contour import ContourPlotter
from src.poisson.cv_solver import CVPDESolver
from src.poisson.dv_solver import DVPDESolver
from src.utils.error_metrics import lp_error
from src.poisson.classical_solver_new import Classical_Solver



class CavityFlowAnalyzer:
    def __init__(self, logger, device: torch.device):
        self.logger = logger
        self.device = device
        self.results = {}
        
    def load_data(self, data_path: str, tstep: int, xstep: int, ystep: int, skip: int) -> None:
        """Load and preprocess cavity flow data."""
        with h5py.File(data_path, "r") as data:
            domain = pd.DataFrame(data["cavity_internal"]).T.to_numpy()
            
        # Reshape and skip data points
        self.time_ = domain[:, 0:1].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
        self.xfa = domain[:, 1:2].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
        self.yfa = domain[:, 2:3].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
        self.ufa = domain[:, 3:4].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
        self.vfa = domain[:, 4:5].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
        self.pfa = domain[:, 5:6].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].reshape(-1, 1)
        
        self.new_shape = domain[:, 0:1].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].shape
        
    def load_model(self, model_name: str, solver_type: str, model_path: str) -> None:
        """Load model based on solver type."""
        
        if solver_type == "CV":
            state = CVPDESolver.load_state(os.path.join(model_path, "model.pth"))

            model = CVPDESolver(state["args"], self.logger, None, self.device)
            model.preprocessor.load_state_dict(state["preprocessor"])
            model.quantum_layer.load_state_dict(state["quantum_layer"])
            model.postprocessor.load_state_dict(state["postprocessor"])
            model.logger.print("CV Model loaded successfully")
        elif solver_type == "classical":
            state = Classical_Solver.load_state(os.path.join(model_path, "model.pth"))
            model = Classical_Solver(state["args"], self.logger)
            model.preprocessor.load_state_dict(state["preprocessor"])
            model.hidden.load_state_dict(state["hidden_network"])
            model.postprocessor.load_state_dict(state["postprocessor"])
            model.logger.print("classical Model loaded successfully")

        else:  # dv solver
            state = DVPDESolver.load_state(os.path.join(model_path, "model.pth"))
            model = DVPDESolver(state["args"], self.logger, None, self.device)
            model.preprocessor.load_state_dict(state["preprocessor"])
            model.postprocessor.load_state_dict(state["postprocessor"])
            model.quantum_layer.load_state_dict(state["quantum_layer"])
            model.logger.print("DV Model loaded successfully")

        self.logger.print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")
        self.logger.print(f"Final loss: {state['loss_history'][-1]}")
        self.logger.print(f"Total number of iterations: {len(state['loss_history'])}")
        return model, state

    def make_predictions(self, model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions using the loaded model."""
        test_data = torch.tensor(
            np.concatenate([self.time_, self.xfa, self.yfa], axis=1), 
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            predictions = model.forward(test_data)
            if predictions.is_cuda:
                predictions = predictions.cpu()
                
        return (predictions[:, 0:1].numpy(),  # u_pred
                predictions[:, 1:2].numpy(),  # v_pred
                predictions[:, 2:3].numpy())  # p_pred

    def calculate_errors(self, u_pred: np.ndarray, v_pred: np.ndarray, p_pred: np.ndarray) -> Tuple[float, float, float]:
        """Calculate L2 errors for predictions."""
        u_error = lp_error(u_pred, self.ufa, "RelL2_U%", self.logger, 2)
        v_error = lp_error(v_pred, self.vfa, "RelL2_V%", self.logger, 2)
        p_error = lp_error(p_pred, self.pfa, "RelL2_P%", self.logger, 2)
        return u_error, v_error, p_error

    def reshape_results(self) -> Dict[str, np.ndarray]:
        """Reshape all results for visualization."""
        tstep, xstep, ystep = self.new_shape
        
        reshaped_data = {
            'tf': self.time_.reshape(tstep, xstep, ystep),
            'xf': self.xfa.reshape(tstep, xstep, ystep),
            'yf': self.yfa.reshape(tstep, xstep, ystep),
            'exact_u': self.ufa.reshape(tstep, xstep, ystep),
            'exact_v': self.vfa.reshape(tstep, xstep, ystep),
            'exact_p': self.pfa.reshape(tstep, xstep, ystep)
        }
        
        for model_name, preds in self.results.items():
            reshaped_data[f'{model_name}_u'] = preds[0].reshape(tstep, xstep, ystep)
            reshaped_data[f'{model_name}_v'] = preds[1].reshape(tstep, xstep, ystep)
            reshaped_data[f'{model_name}_p'] = preds[2].reshape(tstep, xstep, ystep)
            
            # Calculate errors
            reshaped_data[f'{model_name}_error_u'] = np.abs(reshaped_data[f'{model_name}_u'] - reshaped_data['exact_u'])
            reshaped_data[f'{model_name}_error_v'] = np.abs(reshaped_data[f'{model_name}_v'] - reshaped_data['exact_v'])
            reshaped_data[f'{model_name}_error_p'] = np.abs(reshaped_data[f'{model_name}_p'] - reshaped_data['exact_p'])
            
        return reshaped_data

def main():
    log_path = "testing_checkpoints/cavity"
    logger = Logging(log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = CavityFlowAnalyzer(logger, device)
    
    data_path = "./data/cavity.mat"
    analyzer.load_data(data_path, tstep=101, xstep=100, ystep=100, skip=15)
    
    model_paths = {
    #     "angle_cascade": ("dv", "./log_files/checkpoints/cavity/2025-02-22_14-29-03-610093"),
        "classical": ("classical", "./models/2025-02-25_17-21-36-221407"),

    # old:
        "angle_cascade": ("dv", "./models/2025-02-06_19-28-34-814985"),
        # "classical": ("classical", "./log_files/checkpoints/cavity/2025-02-09_22-46-51-012656")
    }


    for model_name, (solver_type, model_path) in model_paths.items():
        model, state = analyzer.load_model(model_name, solver_type, model_path)
        predictions = analyzer.make_predictions(model)
        analyzer.results[model_name] = predictions
        
        u_error, v_error, p_error = analyzer.calculate_errors(*predictions)
        logger.print(f"{model_name} Errors - U: {u_error:.2e}, V: {v_error:.2e}, P: {p_error:.2e}")
    
    reshaped_data = analyzer.reshape_results()
    titles = [
        "Exact solution $u(x)$",
        "PINN prediction $\\hat{u}(x)$", 
        "PINN error",
        "QCPINN prediction $\\hat{u}(x)$",
        "QCPINN error",
        "Exact solution $v(x)$",
        "PINN prediction $\\hat{v}(x)$", 
        "PINN error",
        "QCPINN prediction $\\hat{v}(x)$",
        "QCPINN error",
        "Exact solution $p(x)$",
        "PINN prediction $\\hat{p}(x)$", 
        "PINN error",
        "QCPINN prediction $\\hat{p}(x)$",
        "QCPINN error",
    ]
    nrows_ncols = (3, 5)
    values = [99]
    xref = 1
    yref = 1
    model_dirname = logger.get_output_dir()
    img_width = 30
    img_height = 6
    ticks = 3

    visualization_data = [
        reshaped_data['exact_u'],                  # exact_u
        reshaped_data['classical_u'],
        reshaped_data['classical_error_u'],
        reshaped_data['angle_cascade_u'],
        reshaped_data['angle_cascade_error_u'],
        reshaped_data['exact_v'],                  # exact_v
        reshaped_data['classical_v'],              # v_pred_classic
        reshaped_data['classical_error_v'],        # error_v_classic
        reshaped_data['angle_cascade_v'],          # v_pred_angle_cascade
        reshaped_data['angle_cascade_error_v'],    # error_v_angle_cascade
        reshaped_data['exact_p'],                  # exact_p
        reshaped_data['classical_p'],              # p_pred_classic
        reshaped_data['classical_error_p'],        # error_p_classic
        reshaped_data['angle_cascade_p'],          # p_pred_angle_cascade
        reshaped_data['angle_cascade_error_p'],    # error_p_angle_cascade
    ]

    plotter = ContourPlotter(fontsize=8, labelsize=7, axes_pad=0.65)

    plotter.draw_contourf_regular_2D(
        reshaped_data['tf'][:, 0, 0],
        reshaped_data['xf'][0, 0, :],
        reshaped_data['yf'][0, :, 0],
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

if __name__ == "__main__":
    main()
import sys
import os
import torch
import numpy as np
from typing import Dict, Tuple, Any


from src.utils.logger import Logging
from src.utils.plot_loss import plot_loss_history
from src.nn.DVPDESolver import DVPDESolver
from src.nn.CVPDESolver import CVPDESolver
from src.nn.pde import helmholtz_operator
from src.data.helmholtz_dataset import u, f


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "testing_checkpoints/helmholtz")


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_POINTS = 20
    A1, A2 = 1, 4
    LAMBDA = 1.0

    BASE_PATH = "./models/helmholtz_amplitude"
    MODEL_PATHS = {
        f"{BASE_PATH}/2025-02-12_16-33-22-249802": ("DV", "angle_layered"),
        f"{BASE_PATH}/2025-02-06_19-25-14-069398": ("DV", "angle_cascade"),
        f"{BASE_PATH}/2025-02-06_19-32-04-408294": ("DV", "angle_cross_mesh"),
        f"{BASE_PATH}/2025-02-06_19-25-14-069609": ("DV", "angle_alternate"),
        f"{BASE_PATH}/2025-02-12_16-33-59-543343": ("DV", "amp_layered"),
        f"{BASE_PATH}/2025-02-06_18-54-03-851796": ("DV", "amp_alternate"),
        f"{BASE_PATH}/2025-02-06_18-52-46-865258": ("DV", "amp_cross_mesh"),
        f"{BASE_PATH}/2025-02-06_18-48-39-607215": ("DV", "amp_cascade"),
        # f"{BASE_PATH}/2025-02-06_18-10-42-119955": ("CV", "cv"),
        # f"{BASE_PATH}/2025-02-09_20-05-25-094982": ("CV", "gcv"),
        f"{BASE_PATH}/2025-02-09_00-01-28-238904": ("Classical", "classical"),
    }

    # old classical
    # f"{BASE_PATH}/2025-02-09_00-01-28-238904": ("classical", "classical")


def setup_logger() -> Tuple[Logging, str]:
    """Initialize logging system"""
    try:
        logger = Logging(TEST_CHECKPOINT_PATH)
        return logger, logger.get_output_dir()
    except Exception as e:
        print(f"Error setting up logger: {e}")
        sys.exit(1)


def create_test_grid() -> torch.Tensor:
    """Create the test grid for computations"""
    try:
        dom_coords = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32).to(
            Config.DEVICE
        )
        t = (
            torch.linspace(dom_coords[0, 0], dom_coords[1, 0], Config.NUM_POINTS)
            .to(Config.DEVICE)
            .unsqueeze(1)
        )
        x = (
            torch.linspace(dom_coords[0, 1], dom_coords[1, 1], Config.NUM_POINTS)
            .to(Config.DEVICE)
            .unsqueeze(1)
        )
        t, x = torch.meshgrid(t.squeeze(), x.squeeze(), indexing="ij")
        return torch.hstack((t.flatten().unsqueeze(1), x.flatten().unsqueeze(1))).to(
            Config.DEVICE
        )
    except Exception as e:
        print(f"Error creating test grid: {e}")
        raise


def get_exact_solutions(X_star: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate exact solutions for comparison"""
    try:
        u_star = u(X_star, Config.A1, Config.A2)
        f_star = f(X_star, Config.A1, Config.A2, Config.LAMBDA)
        return u_star.cpu().detach().numpy(), f_star.cpu().detach().numpy()
    except Exception as e:
        print(f"Error calculating exact solutions: {e}")
        raise


def load_model(model_dir: str, solver_type: str, logger: Logging) -> Tuple[Any, Dict]:
    """Load the appropriate solver model based on type"""
    try:
        model_path = os.path.join(model_dir, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

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

        return model, state
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        raise


def evaluate_model(
    model: Any,
    X_star: torch.Tensor,
    u_exact: np.ndarray,
    f_exact: np.ndarray,
    logger: Logging,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model performance and compute errors"""
    try:
        u_pred_star, f_pred_star = helmholtz_operator(
            model, X_star[:, 0:1], X_star[:, 1:2]
        )

        u_pred = u_pred_star.detach().cpu().numpy()
        f_pred = f_pred_star.detach().cpu().numpy()

        error_u = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2) * 100
        error_f = np.linalg.norm(f_pred - f_exact, 2) / np.linalg.norm(f_exact, 2) * 100

        logger.print(f"Relative L2 error_u: {error_u.item():.2e}")
        logger.print(f"Relative L2 error_f: {error_f.item():.2e}")

        return u_pred, f_pred
    except Exception as e:
        logger.print(f"Error in evaluate_model: {str(e)}")
        raise
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise


def main():
    try:
        logger, model_dirname = setup_logger()
        X_star = create_test_grid()
        u_exact, f_exact = get_exact_solutions(X_star)

        all_loss_history = {}
        for model_path, (solver_type, model_name) in Config.MODEL_PATHS.items():
            try:
                model, state = load_model(model_path, solver_type, logger)
                model.logger = logger
                logger.print("******************************\n")

                logger.print(f"\nProcessing model: {model_name}")
                total_params = sum(p.numel() for p in model.parameters())
                logger.print(f"Total number of parameters: {total_params}")

                logger.print(f"Method used: {model_name}")
                logger.print(f"Total iterations: {len(state['loss_history'])}")
                if state["loss_history"]:
                    logger.print(f"Final loss: {state['loss_history'][-1]}")

                # evaluate_model(model, X_star, u_exact, f_exact, logger)
                all_loss_history[model_name] = state["loss_history"]
                logger.print(f"File directory: {model_path}")

            except Exception as e:
                logger.print(f"Error processing model {model_name}: {e}")
                continue
            finally:
                if "model" in locals():
                    del model
            logger.print("******************************\n")

        plot_loss_history(
            all_loss_history,
            os.path.join(logger.get_output_dir(), "loss_history_helmholtz.png"),
            y_max=10000,
            legend=True,
        )

    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

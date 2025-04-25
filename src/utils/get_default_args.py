import argparse
import datetime


def get_default_args():
    """
    Creates and returns the default arguments for the PyTorch MNIST example.

    Returns:
        argparse.Namespace: Parsed arguments with default values.
    """
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--samples-per-class",
        default=500,
        type=int,
        help="Number of training images per class in the training set (default: 500)",
    )
    parser.add_argument(
        "--optimiser", type=str, default="sgd", help="Optimiser to use (default: SGD)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--submission_time",
        type=str,
        default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        metavar="N",
        help="Timestamp at submission",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--plot", action="store_true", default=True, help="Plot the results of the run"
    )
    parser.add_argument(
        "--batchnorm",
        action="store_true",
        default=False,
        help="If enabled, apply BatchNorm1d to the input of the pre-quantum Sigmoid.",
    )
    parser.add_argument(
        "-q",
        "--quantum",
        dest="quantum",
        default=True,
        help="If enabled, use a minimised version of ResNet-18 with QNet as the final layer",
    )
    parser.add_argument(
        "--q_backend",
        type=str,
        default="qasm_simulator",
        help="Type of backend simulator to run quantum circuits on (default: qasm_simulator)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=8,
        help="Width of the test network (default: 8). If quantum, this is the number of qubits.",
    )
    parser.add_argument(
        "--classes", type=int, default=8, help="Number of MNIST classes."
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="vector",
        help="Data encoding method (default: vector)",
    )
    parser.add_argument(
        "--q_ansatz",
        type=str,
        default="sim_circ_14",
        help="Variational ansatz method (default: abbas)",
    )
    parser.add_argument(
        "--q_sweeps", type=int, default=1, help="Number of ansatz sweeps."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="partial_measurement_half",
        help="Quantum layer activation function type (default: null)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Number of shots for quantum circuit evaluations.",
    )
    parser.add_argument(
        "--layers", type=int, default=2, help="Number of test network layers."
    )

    # Parse arguments with default settings
    args = parser.parse_args(args=[])

    return args

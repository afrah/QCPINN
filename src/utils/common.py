from poisson.old.identity_matrix import identity_matrix_system
import os


def sum_vector_cost_func(vector):
    return sum(vector)


def solve_poisson_equation(size):
    import scipy.sparse.linalg as spla

    # get the system setup
    x_axis, y_axis, L_sys, b = identity_matrix_system(size)
    # solve the linear system L_sys * u = b
    u = spla.spsolve(L_sys, b)

    return x_axis, y_axis, u


def create_output_model_path(RESULT_DIR, args, version=0):
    if args["quantum"]:  # Access 'quantum' using dictionary syntax
        model_path = os.path.join(
            RESULT_DIR,
            "MNIST-quantum_{}-backend_{}-classes_{}-ansatz_{}-netwidth_{}-nlayers_{}-nsweeps_{}"
            "-activation_{}-shots_{}-samples_{}-bsize_{}-optimiser_{}-lr_{}-batchnorm_{}"
            "-tepochs_{}-loginterval_{}_{}".format(
                args["quantum"],
                args["q_backend"],
                args["classes"],
                args["q_ansatz"],
                args["width"],
                args["layers"],
                args["q_sweeps"],
                args["activation"],
                args["shots"],
                args["samples_per_class"],
                args["batch_size"],
                args["optimiser"],
                args["lr"],
                args["batchnorm"],
                args["epochs"],
                args["log_interval"],
                version,
            ),
        )
    else:
        model_path = os.path.join(
            RESULT_DIR,
            "MNIST-quantum_{}-classes_{}-netwidth_{}-nlayers_{}-samples_{}-"
            "bsize_{}-optimiser_{}-lr_{}-batchnorm_{}-tepochs_{}-loginterval_{}_{}".format(
                args["quantum"],
                args["classes"],
                args["width"],
                args["layers"],
                args["samples_per_class"],
                args["batch_size"],
                args["optimiser"],
                args["lr"],
                args["batchnorm"],
                args["epochs"],
                args["log_interval"],
                version,
            ),
        )

    if os.path.exists(model_path + ".npy"):
        return create_output_model_path(RESULT_DIR, args, version=version + 1)
    else:
        return model_path

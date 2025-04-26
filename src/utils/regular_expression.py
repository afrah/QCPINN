import re


def extract_loss_values_cavity(filename):
    iterations = []
    loss_r_values = []
    loss_bc_values = []

    pattern = (
        r"Iteration: (\d+), loss_r = ([\d.]+)e-([\d]+) ,\s+loss_bc = ([\d.]+)e-([\d]+)"
    )

    with open(filename, "r") as file:
        for line in file:
            if "Iteration" in line and "loss_r" in line and "loss_bc" in line:
                match = re.search(pattern, line)
                if match:
                    iteration = int(match.group(1))
                    loss_r = float(match.group(2)) * (10 ** -int(match.group(3)))
                    loss_bc = float(match.group(4)) * (10 ** -int(match.group(5)))

                    iterations.append(iteration)
                    loss_r_values.append(loss_r)
                    loss_bc_values.append(loss_bc)

    return iterations, loss_r_values, loss_bc_values


def extract_loss_values_helmholtz(filename):
    iterations = []
    loss_r_values = []
    loss_bc_values = []

    pattern = (
        r"Iteration: (\d+), loss_r = ([\d.]+)e\+(\d+) ,\s+loss_bc = ([\d.]+)e-([\d]+)"
    )

    with open(filename, "r") as file:
        for line in file:
            if "Iteration" in line and "loss_r" in line and "loss_bc" in line:
                match = re.search(pattern, line)
                if match:
                    iteration = int(match.group(1))
                    loss_r = float(match.group(2)) * (10 ** int(match.group(3)))
                    loss_bc = float(match.group(4)) * (10 ** -int(match.group(5)))

                    iterations.append(iteration)
                    loss_r_values.append(loss_r)
                    loss_bc_values.append(loss_bc)

    return iterations, loss_r_values, loss_bc_values

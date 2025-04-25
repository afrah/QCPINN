import torch

# Parameters of the equation
default_D = 0.01  # Diffusion coefficient
default_v_x = 1.0  # Convection velocity in x direction
default_v_y = 1.0  # Convection velocity in y direction


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None, device="cpu"):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
        self.device = device

    def sample(self, N):
        # Generate random samples within the specified range on the correct device
        rand_vals = torch.rand(N, self.dim, device=self.device)
        x = (
            self.coords[0:1, :]
            + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        )
        y = self.func(x.to(self.device))  # Ensure the function is applied on the device
        return x, y


# Define the exact solution and its derivatives for the convection-diffusion equation
def u(txy):
    time_ = txy[:, 0:1]
    x = txy[:, 1:2]
    y = txy[:, 2:3]
    return torch.exp(-100 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) * torch.exp(-time_)


def u_t(txy):
    return -u(txy)


def u_x(txy):
    return -200 * (txy[:, 1:2] - 0.5) * u(txy)


def u_y(txy):
    return -200 * (txy[:, 2:3] - 0.5) * u(txy)


def u_xx(txy):
    return (20000 * (txy[:, 1:2] - 0.5) ** 2 - 100) * u(txy)


def u_yy(txy):
    return (20000 * (txy[:, 2:3] - 0.5) ** 2 - 100) * u(txy)


def r(txy, Diffusion=default_D, v_x=default_v_x, v_y=default_v_y):
    return (
        u_t(txy) + v_x * u_x(txy) + v_y * u_y(txy) - Diffusion * (u_xx(txy) + u_yy(txy))
    )


def generate_training_dataset(device):

    # Domain boundaries
    ics_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float32, device=device
    )
    bc1_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32, device=device
    )
    bc2_coords = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device
    )
    dom_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device
    )

    # Create initial conditions samplers
    ics_sampler = Sampler(3, ics_coords, u, name="Initial Condition", device=device)

    # Create boundary conditions samplers
    bc1 = Sampler(3, bc1_coords, u, name="Dirichlet BC1", device=device)
    bc2 = Sampler(3, bc2_coords, u, name="Dirichlet BC2", device=device)
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(3, dom_coords, r, name="Forcing", device=device)

    return [ics_sampler, bcs_sampler, res_sampler]


# # Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = generate_training_dataset(device)

# # Sampling example
# ics_sampler, bcs_sampler, coll_sampler, res_sampler = dataset
# x, y = bcs_sampler[0].sample(10)
# print(x)
# print(y)
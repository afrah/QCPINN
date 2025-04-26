import torch

A1 = 1
A2 = 4
LAMBDA = 1.0


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None, device="cpu"):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
        self.device = device

    def sample(self, N: int):
        # Generate random values in [0, 1) with shape (N, self.dim) and send to device
        rand_vals = torch.rand(N, self.dim, device=self.device)

        # Calculate the range for sampling: self.coords[1:2, :] - self.coords[0:1, :]
        # Using broadcasting to create the samples within the range
        x = (
            self.coords[0:1, :]
            + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        )

        # Apply the function to the sampled points
        y = self.func(x)

        return x, y


def u(x, a_1, a_2):
    return torch.sin(a_1 * torch.pi * x[:, 0:1]) * torch.sin(a_2 * torch.pi * x[:, 1:2])


def u_xx(x, a_1, a_2):
    return (
        -((a_1 * torch.pi) ** 2)
        * torch.sin(a_1 * torch.pi * x[:, 0:1])
        * torch.sin(a_2 * torch.pi * x[:, 1:2])
    )


def u_yy(x, a_1, a_2):
    return (
        -((a_2 * torch.pi) ** 2)
        * torch.sin(a_1 * torch.pi * x[:, 0:1])
        * torch.sin(a_2 * torch.pi * x[:, 1:2])
    )


# Forcing function
def f(x, a_1, a_2, lam):
    return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * u(x, a_1, a_2)


def generate_training_dataset(device):
    # Parameter
    bc1_coords = torch.tensor([[-1.0, -1.0], [1.0, -1.0]], dtype=torch.float32).to(
        device
    )
    bc2_coords = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float32).to(device)
    bc3_coords = torch.tensor([[1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32).to(device)
    bc4_coords = torch.tensor([[-1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32).to(
        device
    )
    dom_coords = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32).to(
        device
    )

    # Create boundary condition samplers
    bc1 = Sampler(
        2, bc1_coords, lambda x: u(x, A1, A2), name="Dirichlet BC1", device=device
    )
    bc2 = Sampler(
        2, bc2_coords, lambda x: u(x, A1, A2), name="Dirichlet BC2", device=device
    )
    bc3 = Sampler(
        2, bc3_coords, lambda x: u(x, A1, A2), name="Dirichlet BC3", device=device
    )
    bc4 = Sampler(
        2, bc4_coords, lambda x: u(x, A1, A2), name="Dirichlet BC4", device=device
    )

    bcs_sampler = [bc1, bc2, bc3, bc4]
    res_sampler = Sampler(
        2, dom_coords, lambda x: f(x, A1, A2, LAMBDA), name="Forcing", device=device
    )

    return [bcs_sampler, res_sampler]

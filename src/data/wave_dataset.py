import numpy as np
import torch


class Sampler:
    def __init__(self, dim, coords, func, device="cpu"):
        self.dim = dim
        self.coords = torch.tensor(
            coords, dtype=torch.float32, device=device
        )  # Convert coords to float32 tensor
        self.func = func
        self.device = device

    def sample(self, N):
        rand_vals = torch.rand(N, self.dim, dtype=torch.float32, device=self.device)
        x = (
            self.coords[0:1, :]
            + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        )
        y = self.func(x)
        return x, y


def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:, 0:1]
    x = x[:, 1:2]
    return torch.sin(
        torch.tensor(np.pi, dtype=torch.float32, device=x.device) * x
    ) * torch.cos(
        c * torch.tensor(np.pi, dtype=torch.float32, device=x.device) * t
    ) + a * torch.sin(
        2 * c * torch.tensor(np.pi, dtype=torch.float32, device=x.device) * x
    ) * torch.cos(4 * c * torch.tensor(np.pi, dtype=torch.float32, device=x.device) * t)


def u_t(x, a, c):
    t = x[:, 0:1]
    x = x[:, 1:2]
    pi = torch.tensor(np.pi, dtype=torch.float32, device=x.device)
    u_t = -c * pi * torch.sin(pi * x) * torch.sin(
        c * pi * t
    ) - a * 4 * c * pi * torch.sin(2 * c * pi * x) * torch.sin(4 * c * pi * t)
    return u_t


def u_tt(x, a, c):
    t = x[:, 0:1]
    x = x[:, 1:2]
    pi = torch.tensor(np.pi, dtype=torch.float32, device=x.device)
    u_tt = -((c * pi) ** 2) * torch.sin(pi * x) * torch.cos(c * pi * t) - a * (
        4 * c * pi
    ) ** 2 * torch.sin(2 * c * pi * x) * torch.cos(4 * c * pi * t)
    return u_tt


def u_xx(x, a, c):
    t = x[:, 0:1]
    x = x[:, 1:2]
    pi = torch.tensor(np.pi, dtype=torch.float32, device=x.device)
    u_xx = -(pi**2) * torch.sin(pi * x) * torch.cos(c * pi * t) - a * (
        2 * c * pi
    ) ** 2 * torch.sin(2 * c * pi * x) * torch.cos(4 * c * pi * t)
    return u_xx


def r(x, a, c):
    return u_tt(x, a, c) - c**2 * u_xx(x, a, c)


def generate_training_dataset(device):
    a = torch.tensor(0.5, dtype=torch.float32, device=device)
    c = torch.tensor(2.0, dtype=torch.float32, device=device)

    ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bc1_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    bc2_coords = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    ics_sampler = Sampler(2, ics_coords, lambda x: u(x, a, c), device=device)

    bc1 = Sampler(2, bc1_coords, lambda x: u(x, a, c), device=device)
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, a, c), device=device)
    bcs_sampler = [bc1, bc2]

    res_sampler = Sampler(2, dom_coords, lambda x: r(x, a, c), device=device)
    coll_sampler = Sampler(2, dom_coords, lambda x: u(x, a, c), device=device)

    return [ics_sampler, bcs_sampler, res_sampler]

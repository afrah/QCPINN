import torch
import numpy as np


def u(x):
    """
    :param x: x = (t, x)
    """
    return (
        x[:, 1:2] * torch.cos(5 * torch.pi * x[:, 0:1]) + (x[:, 0:1] * x[:, 1:2]) ** 3
    )


def u_tt(x):
    return (
        -25 * torch.pi**2 * x[:, 1:2] * torch.cos(5 * torch.pi * x[:, 0:1])
        + 6 * x[:, 0:1] * x[:, 1:2] ** 3
    )


def u_xx(x):
    return (
        torch.zeros((x.shape[0], 1), device=x.device) + 6 * x[:, 1:2] * x[:, 0:1] ** 3
    )


def f(x, alpha, beta, gamma, k):
    return u_tt(x) + alpha * u_xx(x) + beta * u(x) + gamma * u(x) ** k


def operator(u, t, x, alpha, beta, gamma, k, sigma_t=1.0, sigma_x=1.0):
    u_t = (
        torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        / sigma_t
    )
    u_x = (
        torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        / sigma_x
    )
    u_tt = (
        torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True
        )[0]
        / sigma_t
    )
    u_xx = (
        torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        / sigma_x
    )
    residual = u_tt + alpha * u_xx + beta * u + gamma * u**k
    return residual


class Sampler:
    def __init__(self, dim, coords, func, device="cpu", name=None):
        self.dim = dim
        self.coords = torch.tensor(coords, dtype=torch.float32, device=device)
        self.func = func
        self.name = name
        self.device = device

    def sample(self, N):
        rand_vals = torch.rand(N, self.dim, device=self.device)
        x = (
            self.coords[0:1, :]
            + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        )
        y = self.func(x)
        return x, y


def generate_training_dataset(device="cpu"):
    # Parameters of equations
    alpha = torch.tensor(-1.0, device=device)
    beta = torch.tensor(0.0, device=device)
    gamma = torch.tensor(1.0, device=device)
    k = 3

    # Domain boundaries
    ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bc1_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    bc2_coords = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    # Create initial conditions samplers
    ics_sampler = Sampler(2, ics_coords, lambda x: u(x), device=device)

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x), device=device)
    bc2 = Sampler(2, bc2_coords, lambda x: u(x), device=device)
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(
        2,
        dom_coords,
        lambda x: f(x, alpha, beta, gamma, k),
        device=device,
        name="Forcing",
    )

    return [ics_sampler, bcs_sampler, res_sampler]

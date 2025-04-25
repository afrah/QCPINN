import re
import torch


def navier_stokes_2D_operator(model, t, x, y, min_x=0, max_x=1):

    mu = 0.00345
    DENSITY = 1056.0

    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    uvp = model(torch.concatenate((t, x, y), 1))

    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    continuity = u_x + v_y
    f_u = u_t + (u * u_x + v * u_y) + 1.0 / DENSITY * p_x - mu * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + 1.0 / DENSITY * p_y - mu * (v_xx + v_yy)

    return [continuity, f_u, f_v]


# Operator to compute residuals
def klein_gordon_operator(fluid_model, t, x, x_min=0.0, x_max=1.0):

    # Parameters of equations
    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k = 3

    t.requires_grad = True
    x.requires_grad = True

    u = fluid_model(torch.concatenate((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_tt + alpha * u_xx + beta * u + gamma * u**k
    return u, residual


# Operator to compute residuals
def wave_operator(model, t, x, sigma_t=1.0, sigma_x=1.0):

    c = 2
    t.requires_grad = True
    x.requires_grad = True

    u = model(torch.concatenate((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_tt - c**2 * u_xx
    return u, residual


# Operator to compute residuals for the 2D convection-diffusion equation
def diffusion_operator(
    model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, D=0.01, v_x=1.0, v_y=1.0
):

    # Enable gradient tracking
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    # Forward pass through the model
    u = model(torch.cat((t, x, y), 1))

    # Compute derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] / sigma_x
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0] / sigma_y

    u_xx = (
        torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        / sigma_x
    )
    u_yy = (
        torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        / sigma_y
    )

    # Convection-diffusion equation residual
    residual = u_t + v_x * u_x + v_y * u_y - D * (u_xx + u_yy)

    return u, residual


# Operator for Helmholtz equation
def helmholtz_operator(
    fluid_model,
    x1,
    x2,
):
    LAMBDA = 1.0

    x1.requires_grad_(True)
    x2.requires_grad_(True)
    u = fluid_model(torch.concatenate((x1, x2), 1))
    # Compute gradients with respect to x1 and x2

    # print(f"{u=}")
    # print(f"{x1=}")
    # print(f"{x2=}")

    u_x1 = torch.autograd.grad(
        u, x1, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_x2 = torch.autograd.grad(
        u, x2, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_xx1 = torch.autograd.grad(
        u_x1, x1, grad_outputs=torch.ones_like(u_x1), create_graph=True
    )[0]

    u_xx2 = torch.autograd.grad(
        u_x2, x2, grad_outputs=torch.ones_like(u_x2), create_graph=True
    )[0]

    residual = u_xx1 + u_xx2 + LAMBDA * u

    # print(
    #     f"{u.shape=}, {u_x1.shape=}, {u_x2.shape=}, {u_xx1.shape=}, {u_xx2.shape=}, {residual.shape=}"
    # )
    return [u, residual]

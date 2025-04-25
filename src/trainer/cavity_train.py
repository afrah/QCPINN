import random
import time
import torch
import random
import pennylane as qml
import torch
import torch.nn as nn
import os

from src.nn.pde import navier_stokes_2D_operator


def get_random_minibatch(dataset_length, batch_size):
    batch_indices = random.sample(range(dataset_length), batch_size)
    return batch_indices


def compute_losses(model):
    # print("model data" , model)
    # Access the randomly selected minibatch for each tensor
    batch_indices = get_random_minibatch(
        model.data[0]["txy_domain"].shape[0], model.batch_size
    )
    txy_domain = model.data[0]["txy_domain"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["txy_sensors"].shape[0], model.batch_size
    )
    txy_sensors = model.data[0]["txy_sensors"][batch_indices, :]
    uvp_sensors = model.data[1]["uvp_sensors"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["txy_left"].shape[0], model.batch_size
    )
    txy_left = model.data[0]["txy_left"][batch_indices, :]
    uvp_left = model.data[1]["uvp_left"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["txy_right"].shape[0], model.batch_size
    )
    txy_right = model.data[0]["txy_right"][batch_indices, :]
    uvp_right = model.data[1]["uvp_right"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["txy_bottom"].shape[0], model.batch_size
    )
    txy_bottom = model.data[0]["txy_bottom"][batch_indices, :]
    uvp_bottom = model.data[1]["uvp_bottom"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["txy_up"].shape[0], model.batch_size
    )
    txy_up = model.data[0]["txy_up"][batch_indices, :]
    uvp_up = model.data[1]["uvp_up"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["txy_initial"].shape[0], model.batch_size
    )
    txy_initial = model.data[0]["txy_initial"][batch_indices, :]
    uvp_initial = model.data[1]["uvp_initial"][batch_indices, :]

    t_r, x_r, y_r = txy_domain[:, 0:1], txy_domain[:, 1:2], txy_domain[:, 2:3]

    [continuity, f_u, f_v] = navier_stokes_2D_operator(model, t_r, x_r, y_r)

    lphy = torch.mean(continuity**2 + f_u**2 + f_v**2)

    pred_left = model.forward(
        txy_left,
    )
    lleft = model.loss_fn(pred_left[:, 0], uvp_left[:, 0]) + model.loss_fn(
        pred_left[:, 1], uvp_left[:, 1]
    )

    pred_right = model.forward(
        txy_right,
    )
    lright = model.loss_fn(pred_right[:, 0], uvp_right[:, 0]) + model.loss_fn(
        pred_right[:, 1], uvp_right[:, 1]
    )

    pred_bottom = model.forward(
        txy_bottom,
    )
    lbottom = (model.loss_fn(pred_bottom[:, 0], uvp_bottom[:, 0])) + (
        model.loss_fn(pred_bottom[:, 1], uvp_bottom[:, 1])
    )

    pred_sensors = model.forward(
        txy_sensors,
    )
    lsensors = (
        model.loss_fn(pred_sensors[:, 0], uvp_sensors[:, 0])
        + model.loss_fn(pred_sensors[:, 1], uvp_sensors[:, 1])
        + model.loss_fn(
            pred_sensors[:, 2], uvp_sensors[:, 2]
        )  # adding presssure is essential
    )
    pred_up = model.forward(
        txy_up,
    )
    lup = (model.loss_fn(pred_up[:, 0], uvp_up[:, 0])) + (
        model.loss_fn(pred_up[:, 1], uvp_up[:, 1])
    )

    pred_initial = model.forward(
        txy_initial,
    )
    linitial = (
        model.loss_fn(pred_initial[:, 0], uvp_initial[:, 0])
        + model.loss_fn(pred_initial[:, 1], uvp_initial[:, 1])
        + model.loss_fn(
            pred_initial[:, 2], uvp_initial[:, 2]
        )  # adding presssure is essential
    )

    pred_sensors = model.forward(
        txy_sensors,
    )
    lsensors = (
        model.loss_fn(pred_sensors[:, 0], uvp_sensors[:, 0])
        + model.loss_fn(pred_sensors[:, 1], uvp_sensors[:, 1])
        + model.loss_fn(
            pred_sensors[:, 2], uvp_sensors[:, 2]
        )  # adding presssure is essential
    )
    return {
        "lleft": lleft,
        "lright": lright,
        "lbottom": lbottom,
        "lup": lup,
        "linitial": linitial + lsensors,
        "lphy": lphy,
    }


def train(model):

    for it in range(model.epochs+1):
        time_start = time.time()
        if model.optimizer is not None:
            model.optimizer.zero_grad()
        bclosses = compute_losses(model)

        loss_bc = (
            bclosses["lleft"]
            + bclosses["lright"]
            + bclosses["lbottom"]
            + bclosses["lup"]
        )
        loss_res = bclosses["lphy"]

        loss_factors = {
            "lleft": 2.0,
            "lright": 2.0,
            "lbottom": 2.0,
            "lup": 2.0,
            "linitial": 2.0,
            "lphy": 0.1,
        }
        loss = sum(factor * bclosses[key] for key, factor in loss_factors.items())
        time_end = time.time()
        time_taken = time_end - time_start
        if it % model.args["print_every"] == 0:
            model.logger.print(
                "Iteration: %d, loss_r = %.1e ,  loss_bc = %.1e,  lr = %0.1e, time_taken = %.1e"
                % (
                    it,
                    loss_res.item(),
                    loss_bc.item(),
                    model.optimizer.param_groups[0]["lr"],
                    time_taken,
                )
            )
            # print(f"quantum parameters: {model.params.shape}")
            # print(f"quantum parameters: {model.params.shape}")
            # print(f"quantum parameters: {model.params}")

            model.save_state()
        loss.backward(retain_graph=True)
        if model.optimizer is not None:
            model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.step(loss)  # Step the learning rate scheduler
        model.loss_history.append(loss.item())

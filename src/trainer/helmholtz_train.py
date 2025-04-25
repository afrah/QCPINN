import time
import torch
from src.data.helmholtz_dataset import generate_training_dataset
from src.nn.pde import helmholtz_operator


def fetch_minibatch(sampler, N):
    X, Y = sampler.sample(N)
    return X, Y


def train(model):
    [bcs_sampler, res_sampler] = generate_training_dataset(model.device)

    def objective_fn(it):
        time_start = time.time()
        if model.optimizer is not None:
            model.optimizer.zero_grad()
        # Fetch boundary mini-batches
        X_bc1_batch, u_bc1_batch = fetch_minibatch(bcs_sampler[0], model.batch_size)
        X_bc2_batch, u_bc2_batch = fetch_minibatch(bcs_sampler[1], model.batch_size)
        X_bc3_batch, u_bc3_batch = fetch_minibatch(bcs_sampler[2], model.batch_size)
        X_bc4_batch, u_bc4_batch = fetch_minibatch(bcs_sampler[3], model.batch_size)
        # Fetch residual mini-batch
        X_res_batch, f_res_batch = fetch_minibatch(res_sampler, model.batch_size)
        u_bc1_pred = model.forward(X_bc1_batch)
        u_bc2_pred = model.forward(X_bc2_batch)
        u_bc3_pred = model.forward(X_bc3_batch)
        u_bc4_pred = model.forward(X_bc4_batch)
        # print(f"u_bc4_pred: {u_bc4_pred.shape} , u_bc4_batch: {u_bc4_batch.shape}")

        x1_r, x2_r = X_res_batch[:, 0:1], X_res_batch[:, 1:2]
        [_, r_pred] = helmholtz_operator(model, x1_r, x2_r)
        # print(f"r_pred: {r_pred.shape} , f_res_batch: {f_res_batch.shape}")

        # if model.args["solver"] == "CV" :

        #     # Normalize physics residual
        #     r_pred = r_pred / (torch.norm(r_pred) + 1e-8)
        #     u_bc1_pred = u_bc1_pred / (torch.norm(u_bc1_pred) + 1e-8)
        #     u_bc2_pred = u_bc2_pred / (torch.norm(u_bc2_pred) + 1e-8)
        #     u_bc3_pred = u_bc3_pred / (torch.norm(u_bc3_pred) + 1e-8)
        #     u_bc4_pred = u_bc4_pred / (torch.norm(u_bc4_pred) + 1e-8)
        #     f_res_batch = f_res_batch / (torch.norm(f_res_batch) + 1e-8)

        loss_r = model.loss_fn(r_pred, f_res_batch)
        loss_bc1 = model.loss_fn(u_bc1_pred, u_bc1_batch)
        loss_bc2 = model.loss_fn(u_bc2_pred, u_bc2_batch)
        loss_bc3 = model.loss_fn(u_bc3_pred, u_bc3_batch)
        loss_bc4 = model.loss_fn(u_bc4_pred, u_bc4_batch)

        loss_bc = loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
        if model.args["solver"] == "CV":
            loss = 0.1 * loss_r + 1 * (loss_bc)
        else:
            loss = loss_r + 10.0 * (loss_bc)
        time_end = time.time()
        time_taken = time_end - time_start
        if it % model.args["print_every"] == 0:
            model.logger.print(
                "Iteration: %d, loss_r = %.1e ,  loss_bc = %.1e,  lr = %0.1e, time_taken = %.1e"
                % (
                    it,
                    loss_r.item(),
                    loss_bc.item(),
                    model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0,
                    time_taken,
                )
            )
            # print(f"quantum parameters: {model.params.shape}")
            # print(f"quantum parameters: {model.params.shape}")
            # print(f"quantum parameters: {model.params}")

            model.save_state()
        return loss

    for it in range(model.epochs + 1):
        loss = objective_fn(it)
        # print(f"{loss.item()=}")
        loss.backward(retain_graph=True)
        if model.args["solver"] == "CV":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        if model.optimizer is not None:
            model.optimizer.step()

        if model.scheduler is not None:
            model.scheduler.step(loss)  # Step the learning rate scheduler

        model.loss_history.append(loss.item())

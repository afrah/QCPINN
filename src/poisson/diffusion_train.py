import time
import torch

from src.data.diffusion_dataset import generate_training_dataset
from src.nn.pde import diffusion_operator



def fetch_minibatch(sampler, N):
    X, Y = sampler.sample(N)
    return X, Y


def train(model, nIter=10000, batch_size=128, log_NTK=False, update_lam=False):


    [ics_sampler , bcs_sampler, res_sampler] = generate_training_dataset(model.device)

    def objective_fn(it):
        start_time =  time.time()
        if model.optimizer is not None:
            model.optimizer.zero_grad()
        # Fetch boundary mini-batches , 
        X_ics_batch, u_ics_batch = fetch_minibatch(ics_sampler, batch_size // 3)
        X_bcs_batch, u_bcs_batch = fetch_minibatch(bcs_sampler[0], batch_size // 3)
        
        # Fetch residual mini-batch
        X_res_batch, r_res_batch = fetch_minibatch(res_sampler, batch_size)


        X_ics_batch.requires_grad_(True)
        t_ics = X_ics_batch[:, 0:1]  # temporal component
        t_ics.requires_grad_(True)
        u_bc1_pred = model.forward(X_bcs_batch)
        u_ics_pred = model.forward(X_ics_batch)
        

        t_r, x_r , y_r = X_res_batch[:, 0:1], X_res_batch[:, 1:2], X_res_batch[:, 2:3]
        [_, r_pred] = diffusion_operator(model, t_r, x_r , y_r)

        # Compute the loss

        loss_r = model.loss_fn(r_pred , r_res_batch)

        loss_bc1 = model.loss_fn(u_bc1_pred, u_bcs_batch )
        loss_ics = model.loss_fn(u_ics_pred, u_ics_batch)


        loss = 2.0 * (loss_r ) +   4.0 *  loss_bc1 +    2.0 * loss_ics 

        elapsed = time.time() - start_time

        # Print
        if it % model.args["print_every"] == 0:
          
            model.logger.print('It: %d, Loss: %.2e, Loss_res: %.2e,  Loss_bcs: %.2e, loss_ics: %.2e, lr: %.2e, Time: %.2e'
                                %(it, loss.item(), loss_r.item(), loss_bc1.item(), loss_ics.item(), 
                                  model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0,
                                  elapsed))
            
            # Compute and Print adaptive weights during training
                # Compute the adaptive constant
            model.save_state()
        return loss

    for it in range(model.epochs+1):
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

import time
import torch

from src.data.wave_dataset import generate_training_dataset
from src.nn.pde import wave_operator



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
        X_bc1_batch, _ = fetch_minibatch(bcs_sampler[0], batch_size // 3)
        X_bc2_batch, _ = fetch_minibatch(bcs_sampler[1], batch_size // 3)
        
        # Fetch residual mini-batch
        X_res_batch, _ = fetch_minibatch(res_sampler, batch_size)


        X_ics_batch.requires_grad_(True)
        t_ics = X_ics_batch[:, 0:1]  # temporal component
        t_ics.requires_grad_(True)
        u_bc1_pred = model.forward(X_bc1_batch)
        u_bc2_pred = model.forward(X_bc2_batch)
        u_ics_pred = model.forward(X_ics_batch)
        

        # Compute gradients with respect to time
        u_t = torch.autograd.grad(
            u_ics_pred, 
            X_ics_batch,  # Changed from X_ics_batch[0] to t_ics
            grad_outputs=torch.ones_like(u_ics_pred),
            create_graph=True,
        )[0]
        
        x1_r, x2_r = X_res_batch[:, 0:1], X_res_batch[:, 1:2]
        [_, r_pred] = wave_operator(model, x1_r, x2_r)

        # Compute the loss

        loss_r = model.loss_fn(r_pred , torch.zeros_like(r_pred))

        loss_bc1 = model.loss_fn(u_bc1_pred, torch.zeros_like(u_bc1_pred))
        loss_bc2 = model.loss_fn(u_bc2_pred, torch.zeros_like(u_bc2_pred))
        loss_ics = model.loss_fn(u_ics_pred, u_ics_batch)


        loss_u_t = model.loss_fn(u_t[: , 0], torch.zeros_like(u_t[: , 0]))

        loss_bc= loss_bc1 + loss_bc2 +  loss_ics
        loss = 0.1 * (loss_r +  loss_u_t )+  10 *  loss_bc 

        elapsed = time.time() - start_time

        # Print
        if it % model.args["print_every"] == 0:
          
            model.logger.print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e, lr: %.3e, Time: %.2e'
                                %(it, loss.item(), loss_r.item(), loss_bc.item(), loss_u_t.item(), 
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
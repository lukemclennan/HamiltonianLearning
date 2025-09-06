from trainers.gda_trainer import GDATrainer
from utils.tensorboard_logger_mixin import TensorboardLoggerMixin
from torchdiffeq import odeint
from collections import defaultdict
import torch
import copy
import os


class AveragedModel(torch.nn.Module):
    def __init__(self, model, avg_fn=None):
        """
        Implements Polyak averaging for model parameters.
        
        Args:
            model: Base model to average
            avg_fn: Function to use for averaging (default: exponential moving average)
        """
        super(AveragedModel, self).__init__()
        self.model = model
        self.averaged_model = copy.deepcopy(model)
        
        if avg_fn is None:
            # Default to exponential moving average
            self.avg_fn = lambda averaged_param, param, step_num: \
                0.9 * averaged_param + 0.1 * param
        else:
            self.avg_fn = avg_fn
            
        self.step_num = 0
    
    def forward(self, *args, **kwargs):
        return self.averaged_model(*args, **kwargs)
    
    def update_parameters(self):
        """Update the parameters of the averaged model."""
        for p_avg, p in zip(self.averaged_model.parameters(), self.model.parameters()):
            p_avg.data = self.avg_fn(p_avg.data, p.data, self.step_num)
        self.step_num += 1


class SGDATrainer(GDATrainer, TensorboardLoggerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device = self.device if hasattr(self, 'device') else 'cpu'
        self.lambdas = torch.ones(4, device=device, requires_grad=True)

    
    
    def train(self, **kwargs):
        averaged_model = AveragedModel(self.model)
        for epoch in range(1, self.epochs + 1):
            if self.is_distributed:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.data_loader):
                inputs, targets = self.unpack_batch(batch)
                # Enforce unpack_batch make gpu forwarding
                # inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.predict(inputs)
                loss, neg_loglike, KL_x0, KL_w = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # update lambda
                # Update averaged model
                averaged_model.update_parameters()
                
                # Update lambda parameters in opposite direction (ascent)
                # Recreate computation graph for lambda updates
                (batch_x0, batch_t) = inputs
                s_batch_x0 = self.model.sampling_x0(batch_x0)
                self.model.sampling_epsilon_f()
                pred_x = odeint(self.model, s_batch_x0, batch_t, method='fehlberg2', atol=1e-4, rtol=1e-4)
                
                liouville_loss = self.cons_vol_loss(pred_x)
                energy_loss = self.compute_energy_conservation_loss(self.model, pred_x)
                hamiltonians_00 = self.model.sample_hamiltonian(torch.zeros(batch_x0.shape[-1]))
                hamiltonians_00_penalty = torch.abs(hamiltonians_00)
                hamiltonians_0 = self.model.sample_hamiltonian(pred_x[0])
                penalty = torch.relu(-hamiltonians_0).mean()
                
                # Update lambdas using gradient ascent (maximize constraint violations)
                lambda_loss = -(lambdas[0]*liouville_loss + lambdas[1]*penalty + 
                            lambdas[2]*hamiltonians_00_penalty + lambdas[3]*energy_loss)
                lambda_loss.backward()
                lambda_optim.step()
                lambda_optim.zero_grad()


                loss = loss / self.batch_x0.shape[0] / self.batch_t.shape[0]
                epoch_loss += loss.item()
                if batch_idx % self.log_interval == 0 and self.rank == 0:
                    print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}, λs = {[round(l.item(), 4) for l in self.lambdas]}")

            if self.rank == 0:
                if epoch % kwargs['ckpt_interval'] == 0:
                    print(f"Epoch {epoch} complete. Avg loss: {epoch_loss / len(self.data_loader):.4f}")
                    self.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lambdas': self.lambdas
                    }, filename=os.path.join(self.save_dir, f'gda_checkpoint_epoch{epoch}.pt'))
                
                # Validation
                if epoch % self.log_interval == 0:
                    train_loss = epoch_loss / len(self.data_loader)
                    val_loss, val_KL_x0, val_KL_w, val_nll = self.validate()

                    # Record stats
                    self.stats['train_loss'].append(train_loss)
                    self.stats['val_loss'].append(val_loss)
                    self.stats['train_kl_x0'].append(KL_x0.item())
                    self.stats['train_kl_w'].append(KL_w.item())
                    self.stats['train_neg_loglike'].append(neg_loglike.item())
                    self.stats['val_kl_x0'].append(val_KL_x0)
                    self.stats['val_kl_w'].append(val_KL_w)
                    self.stats['val_neg_loglike'].append(val_nll)

                    # TensorBoard logging
                    metrics = {
                        "Loss/Train": train_loss,
                        "Loss/Val": val_loss,
                        "KL/train_x0": KL_x0.item(),
                        "KL/train_w": KL_w.item(),
                        "KL/val_x0": val_KL_x0,
                        "KL/val_w": val_KL_w,
                        "NegLogLike/Train": neg_loglike.item(),
                        "NegLogLike/Val": val_nll,
                    }
                    self.log_scalars(metrics, epoch)

                    # Save best checkpoint
                    if val_loss < self.min_val_loss or epoch == self.epochs:
                        self.min_val_loss = val_loss
                        self.best_step = epoch
                        self.best_train_loss = train_loss
                        self.best_model = copy.deepcopy(self.model)
                        self.save_checkpoint({
                            'step': epoch,
                            'state_dict': self.model.state_dict(),
                            'optim_dict': self.optimizer.state_dict(),
                            'stats': self.stats,
                            'best_train_loss': self.best_train_loss,
                            'min_val_loss': self.min_val_loss,
                            'best_step': self.best_step,
                            'model_hyperparameters': {
                                'input_dim': self.model.d,
                                'basis': self.model.num_basis,
                                'friction': bool(self.model.eta.item()),
                                'K': self.model.K,
                                'lambdas': self.lambdas.detach().cpu().tolist(),
                            }
                        }, filename=os.path.join(self.save_dir, f"gda_best_epoch{epoch}.pth.tar"))

        # Close TensorBoard writer
        self.close_writer()

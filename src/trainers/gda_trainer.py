from trainers.base_trainer import BaseTrainer
from utils.tensorboard_logger_mixin import TensorboardLoggerMixin
from torchdiffeq import odeint
from collections import defaultdict
import torch
import copy
import os

class GDATrainer(BaseTrainer, TensorboardLoggerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device = self.device if hasattr(self, 'device') else 'cpu'
        self.lambdas = torch.zeros(4, device=device, requires_grad=False)

        self.min_val_loss = float("inf")
        self.best_model = None
        self.best_step = 0
        self.best_train_loss = None
        self.stats = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_kl_x0': [],
            'train_kl_w': [],
            'train_neg_loglike': [],
            'val_kl_x0': [],
            'val_kl_w': [],
            'val_neg_loglike': [],
        }
        log_name = kwargs.get('save_dir', "logs/gda")
        self.checkpoint_prefix = "gda"
        self.init_writer(name=log_name)

    def compute_loss(self, outputs, targets):
        # targets: (batch_step, N, 1, d)
        pred_x = outputs
        batch_ys = targets
        model = self.model
        # Step 1: Base loss components
        neg_loglike = model.neg_loglike(batch_ys, pred_x)
        KL_x0 = model.KL_x0(self.batch_x0.squeeze(1))
        KL_w = model.KL_w()
        elbo_loss = neg_loglike + KL_w + KL_x0

        # Step 2: Extra losses
        liouville_loss = self.cons_vol_loss(pred_x)
        hamiltonians_0 = model.sample_hamiltonian(pred_x[0])
        energy_loss = 0
        for i in range(pred_x.shape[0] - 1):
            energy_loss += torch.mean((model.sample_hamiltonian(pred_x[i + 1]) - hamiltonians_0)**2)

        # Hamiltonian(0,0) penalty
        hamiltonians_00 = model.sample_hamiltonian(torch.tensor([[0.0, 0.0]], device=self.device))
        hamiltonians_00_penalty = hamiltonians_00
        penalty = torch.relu(-hamiltonians_0).mean()

        loss = elbo_loss \
               + self.lambdas[0] * liouville_loss \
               + self.lambdas[1] * penalty \
               + self.lambdas[2] * hamiltonians_00_penalty \
               + self.lambdas[3] * energy_loss

        self._gda_duals = (liouville_loss, penalty, hamiltonians_00_penalty[0][0], energy_loss)
        
        return loss, neg_loglike, KL_x0, KL_w

    def unpack_batch(self, batch):
        self.batch_x0, self.batch_t, self.batch_ys = batch
        self.batch_x0 = self.batch_x0.to(self.device)
        #TODO: Broadcasting time slice
        self.batch_t = self.batch_t[0].to(self.device)
        self.batch_ys = self.batch_ys.to(self.device).permute(1, 0, 2, 3)
        return (self.batch_x0, self.batch_t), self.batch_ys


    def cons_vol_loss(self, pred_x):
        q1_bounds = torch.rand(2).sort()[0].to(pred_x.device) - 0.5
        p1_bounds = torch.rand(2).sort()[0].to(pred_x.device) * 0.2 - 0.1
        in_domain = torch.cat([
            pred_x[:, :, :, 0:1] > q1_bounds[0],
            pred_x[:, :, :, 0:1] < q1_bounds[1],
            pred_x[:, :, :, 1:2] > p1_bounds[0],
            pred_x[:, :, :, 1:2] < p1_bounds[1]
        ], dim=-1)
        in_domain = torch.all(in_domain, dim=-1).float()
        in_domain = torch.mean(in_domain, dim=1)
        return torch.mean((in_domain - in_domain[0])**2)

    def save_checkpoint(self, state, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)

    def predict(self, inputs):
        # inputs = (x0, t) — unpack here
        x0, t = inputs
        s_x0 = self.model.sampling_x0(x0)
        self.model.sampling_epsilon_f()
        return odeint(self.model, s_x0, t, method='fehlberg2', atol=1e-4, rtol=1e-4)
    
    
    def get_model_hyperparameters(self):
        # subclasses can override if they have extra fields
        return {
            'input_dim': self.model.d,
            'basis':     self.model.num_basis,
            'friction':  bool(self.model.eta.item()),
            'K':         self.model.K,
            'lambdas':   self.lambdas.detach().cpu().tolist(),
        }

    def _log_and_checkpoint(self,
                            epoch: int,
                            epoch_loss: float,
                            KL_x0, KL_w, neg_loglike, 
                            **validate_kwargs):
        # 1) compute train & val
        train_loss = epoch_loss / len(self.data_loader)
        val_loss, val_KL_x0, val_KL_w, val_nll = self.validate(**validate_kwargs)

        # 2) record stats
        self.stats['epoch'].append(epoch)
        self.stats['train_loss'].append(train_loss)
        self.stats['val_loss'].append(val_loss)
        self.stats['train_kl_x0'].append(KL_x0.item())
        self.stats['train_kl_w'].append(KL_w.item())
        self.stats['train_neg_loglike'].append(neg_loglike.item())
        self.stats['val_kl_x0'].append(val_KL_x0)
        self.stats['val_kl_w'].append(val_KL_w)
        self.stats['val_neg_loglike'].append(val_nll)

        # 3) TensorBoard
        metrics = {
            "Loss/Train":      train_loss,
            "Loss/Val":        val_loss,
            "KL/train_x0":     KL_x0.item(),
            "KL/train_w":      KL_w.item(),
            "KL/val_x0":       val_KL_x0,
            "KL/val_w":        val_KL_w,
            "NegLogLike/Train":neg_loglike.item(),
            "NegLogLike/Val":  val_nll,
        }
        self.log_scalars(metrics, epoch)

        # 4) best‐model checkpoint
        if (val_loss < self.min_val_loss) or (epoch % self.ckpt_interval == 0):
            best = val_loss < self.min_val_loss
            interval = epoch % self.ckpt_interval == 0
            
            self.min_val_loss    = val_loss
            self.best_step       = epoch
            self.best_train_loss = train_loss
            # deep‐copy so you don’t accidentally keep graph refs
            self.best_model      = copy.deepcopy(self.model)
            ckpt = {
                'step': epoch,
                'state_dict':    self.model.state_dict(),
                'optim_dict':    self.optimizer.state_dict(),
                'stats':         self.stats,
                'best_train_loss': self.best_train_loss,
                'min_val_loss':  self.min_val_loss,
                'best_step':     self.best_step,
                'model_hyperparameters': self.get_model_hyperparameters(),
            }
            if best:
                filename = os.path.join(self.save_dir, f"{self.checkpoint_prefix}_best_epoch.pth.tar")
                self.save_checkpoint(ckpt, filename)
            if interval:
                filename = os.path.join(self.save_dir, f"{self.checkpoint_prefix}_epoch{epoch}.pth.tar")
                self.save_checkpoint(ckpt, filename)
    
    def train(self, **kwargs):
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
                # GDA Lagrange multipliers update (duals)
                with torch.no_grad():
                    lr = self.optimizer.param_groups[0]['lr']
                    duals = self._gda_duals
                    for i in range(4):
                        self.lambdas[i] += lr * duals[i]

                loss = loss / self.batch_x0.shape[0] / self.batch_t.shape[0]
                epoch_loss += loss.item()
                if batch_idx % self.log_interval == 0 and self.rank == 0:
                    print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}, λs = {[round(l.item(), 4) for l in self.lambdas]}")

            if self.rank == 0:
                # Validation
                if epoch % self.log_interval == 0:
                    self._log_and_checkpoint(epoch, loss, KL_x0=KL_x0, KL_w=KL_w, neg_loglike=neg_loglike)
        # Close TensorBoard writer
        self.close_writer()
    
    def validate(self):
        self.model.eval()
        if self.val_loader is None:
            raise ValueError("No validation dataset")
        val_losses, klx0s, klws, nlls = [], [], [], []
        for val_batch in self.val_loader:
            (batch_x0, batch_t), batch_ys = self.unpack_batch(val_batch)
            with torch.no_grad():
                pred_val = self.predict((batch_x0, batch_t))
                val_nll = self.model.neg_loglike(batch_ys, pred_val).item()
                val_KL_x0 = self.model.KL_x0(batch_x0.squeeze()).item()
                val_KL_w = self.model.KL_w().item()
                val_loss = val_nll + val_KL_x0 + val_KL_w
                val_losses.append(val_loss / batch_t.shape[0] / batch_x0.shape[0])
                klx0s.append(val_KL_x0)
                klws.append(val_KL_w)
                nlls.append(val_nll)
            
        self.model.train()
        
        val_loss = sum(val_losses) / len(val_losses)  
        val_KL_x0 = sum(klx0s) / len(klx0s)
        val_KL_w = sum(klws) / len(klws)
        val_nll = sum(nlls) / len(nlls)

        return val_loss, val_KL_x0, val_KL_w, val_nll

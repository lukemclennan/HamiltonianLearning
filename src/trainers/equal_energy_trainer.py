from trainers.gda_energy_trainer import GDAEnergyTrainer
from utils.tensorboard_logger_mixin import TensorboardLoggerMixin
from torchdiffeq import odeint
from collections import defaultdict
import torch
import copy
import os

class EqualEnergyTrainer(GDAEnergyTrainer, TensorboardLoggerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device = self.device if hasattr(self, 'device') else 'cpu'

        # equal weights
        self.lambdas = torch.ones(4, device=device, requires_grad=False)

        self.checkpoint_prefix = "equal_energy"

    
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

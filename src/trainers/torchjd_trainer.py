from trainers.gdad_trainer import GDADTrainer
from utils.tensorboard_logger_mixin import TensorboardLoggerMixin
from torchdiffeq import odeint
from collections import defaultdict
import torch
import copy
import os

import torchjd

def print_weights(_, __, weights: torch.Tensor) -> None:
    """Prints the extracted weights."""
    print(f"Weights: {weights}")

def print_gd_similarity(_, inputs: tuple[torch.Tensor, ...], aggregation: torch.Tensor) -> None:
    """Prints the cosine similarity between the aggregation and the average gradient."""
    matrix = inputs[0]
    gd_output = matrix.mean(dim=0)
    similarity = torch.nn.functional.cosine_similarity(aggregation, gd_output, dim=0)
    print(f"Cosine similarity: {similarity.item():.4f}")

class TorchJDTrainer(GDADTrainer, TensorboardLoggerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device = self.device if hasattr(self, 'device') else 'cpu'

        # equal weights
        self.lambdas = torch.ones(4, device=device, requires_grad=False)

        self.checkpoint_prefix = "torchjd"

        self.aggregator = torchjd.aggregation.CAGrad(c=0.5)
        self.aggregator.weighting.register_forward_hook(print_weights)
        self.aggregator.register_forward_hook(print_gd_similarity)

    def compute_loss(self, outputs, targets):
        # targets: (batch_step, N, 1, d)
        pred_x, pred_x_c = outputs
        batch_ys = targets
        model = self.model
        # Step 1: Base loss components
        neg_loglike = model.neg_loglike(batch_ys, pred_x)
        KL_x0 = model.KL_x0(self.batch_x0.squeeze(1))
        KL_w = model.KL_w()
        elbo_loss = neg_loglike + KL_w + KL_x0

        # Step 2: Extra losses
        liouville_loss = self.cons_vol_loss(pred_x_c)
        hamiltonians_0 = model.sample_hamiltonian(pred_x_c[0])
        energy_loss = 0
        for i in range(pred_x.shape[0] - 1):
            energy_loss += torch.mean((torch.relu(model.sample_hamiltonian(pred_x_c[i + 1]) - hamiltonians_0))**2)

        # Hamiltonian(0,0) penalty
        hamiltonians_00 = model.sample_hamiltonian(torch.tensor([[0.0, 0.0]], device=self.device))
        hamiltonians_00_penalty = hamiltonians_00
        penalty = torch.relu(-hamiltonians_0).mean()

        loss = elbo_loss \
               + self.lambdas[0] * liouville_loss \
               + self.lambdas[1] * penalty \
               + self.lambdas[2] * hamiltonians_00_penalty \
               + self.lambdas[3] * energy_loss

        # losses = (elbo_loss, liouville_loss, penalty, hamiltonians_00_penalty[0][0], energy_loss)

        return loss, neg_loglike, KL_x0, KL_w, liouville_loss, penalty, hamiltonians_00_penalty[0][0], energy_loss
    
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
                loss, neg_loglike, KL_x0, KL_w, liouville_loss, penalty, hamiltonians_00_penalty, energy_loss = self.compute_loss(outputs, targets)

                losses = [neg_loglike+KL_x0+KL_w, liouville_loss, penalty, hamiltonians_00_penalty, energy_loss]

                torchjd.backward(losses, self.aggregator)
                self.optimizer.step()

                loss = loss / self.batch_x0.shape[0] / self.batch_t.shape[0]
                epoch_loss += loss.item()
                if batch_idx % self.log_interval == 0 and self.rank == 0:
                    print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}")

            if self.rank == 0:
                # Validation
                if epoch % self.log_interval == 0:
                    self._log_and_checkpoint(epoch, loss, KL_x0=KL_x0, KL_w=KL_w, neg_loglike=neg_loglike)

        # Close TensorBoard writer
        self.close_writer()

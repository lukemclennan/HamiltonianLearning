from trainers.gda_trainer import GDATrainer
from utils.tensorboard_logger_mixin import TensorboardLoggerMixin
from torchdiffeq import odeint
from collections import defaultdict
import torch
import copy
import os

class NoRegTrainer(GDATrainer, TensorboardLoggerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device = self.device if hasattr(self, 'device') else 'cpu'
        self.checkpoint_prefix = "noreg"
        
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
        loss = elbo_loss

        return loss, neg_loglike, KL_x0, KL_w
    
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

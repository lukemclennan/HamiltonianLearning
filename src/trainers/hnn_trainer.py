from trainers.base_trainer import BaseTrainer
from utils.tensorboard_logger_mixin import TensorboardLoggerMixin
from torchdiffeq import odeint
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os

class HNNTrainer(BaseTrainer, TensorboardLoggerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device = self.device if hasattr(self, 'device') else 'cpu'

        self.min_val_loss = float("inf")
        self.best_model = None
        self.best_step = 0
        self.best_train_loss = None
        self.stats = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
        }
        log_name = kwargs.get('save_dir', "logs/hnn")
        self.checkpoint_prefix = "hnn"
        self.init_writer(name=log_name)

    def predict(self, inputs):
        return self.model(inputs[0], inputs[1])

    def compute_loss(self, outputs, targets):
        return (outputs-targets).pow(2).mean()

    def unpack_batch(self, batch):
        self.batch_x0, self.batch_t, self.batch_ys = batch
        self.batch_x0 = self.batch_x0.to(self.device)
        self.batch_t = self.batch_t[0].to(self.device).requires_grad_(True)
        self.batch_ys = self.batch_ys.to(self.device).permute(1, 0, 2, 3).requires_grad_(True)
        # print(self.batch_x0.shape, self.batch_t.shape, self.batch_ys.shape)
        self.batch_dys = torch.tensor(np.gradient(self.batch_ys.cpu().detach().numpy(), self.batch_t.cpu().detach().numpy(), axis=0), device=self.device, requires_grad=True)
        return (self.batch_t, self.batch_ys), self.batch_dys
    
    def save_checkpoint(self, state, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
    
    def get_model_hyperparameters(self):
        # subclasses can override if they have extra fields
        return {
            'input_dim':  self.model.in_dim,
            'hid_dim':    self.model.hid_dim,
            'hid_layers': self.model.hid_layers
        }

    def _log_and_checkpoint(self,
                            epoch: int,
                            epoch_loss: float,
                            **validate_kwargs):
        # 1) compute train & val
        train_loss = epoch_loss / len(self.data_loader)
        val_loss = self.validate(**validate_kwargs)

        # 2) record stats
        self.stats['epoch'].append(epoch)
        self.stats['train_loss'].append(train_loss)
        self.stats['val_loss'].append(val_loss)

        # 3) TensorBoard
        metrics = {
            "Loss/Train":      train_loss,
            "Loss/Val":        val_loss,
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
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if batch_idx % self.log_interval == 0 and self.rank == 0:
                    print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}")

            if self.rank == 0:
                # Validation
                if epoch % self.log_interval == 0:
                    self._log_and_checkpoint(epoch, loss)
            
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # Close TensorBoard writer
        self.close_writer()
    
    def validate(self):
        self.model.eval()
        if self.val_loader is None:
            raise ValueError("No validation dataset")
        val_losses = []
        for val_batch in self.val_loader:
            inputs, targets = self.unpack_batch(val_batch)
            outputs = self.predict(inputs)
            val_loss = self.compute_loss(outputs, targets)
            val_losses.append(val_loss.item())
            
        self.model.train()
        
        val_loss = sum(val_losses) / len(val_losses)

        return val_loss

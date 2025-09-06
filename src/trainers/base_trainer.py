import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

class BaseTrainer:
    def __init__(self, model, dataset, optimizer, val_dataset=None, epochs=100, batch_size=32, log_interval=10, ckpt_interval=1000, save_dir="checkpoints", device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.ckpt_interval = ckpt_interval
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.val_dataset = val_dataset
        # wrap val_dataset in a DataLoader if provided
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.val_loader = None

        # Prepare data
        if self.is_distributed:
            self.train_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
            self.data_loader = DataLoader(dataset, batch_size=batch_size, sampler=self.train_sampler)
        else:
            self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare model
        self.model = model.to(self.device)
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device.index])
        
        self.optimizer = optimizer

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
                print(f"Epoch {epoch} complete. Avg loss: {epoch_loss / len(self.data_loader):.4f}")
                self.save_checkpoint(epoch)

    def predict(self, inputs):
        return self.model(inputs)  # for SDEs / ODEs, this needs to be overloaded

    def compute_loss(self, outputs, targets):
        # Override in subclass
        raise NotImplementedError

    def unpack_batch(self, batch):
        # Override in subclass if dataset returns more than (x, y)
        return batch

    def save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_epoch_{epoch}.pt"))
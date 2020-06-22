from catalyst.dl import Runner
import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        x, y, path = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.batch_metrics = {'loss': loss}
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = self._batch2device(batch, self.device)
        if len(batch) == 2:
            x, _ = batch
        elif len(batch) == 3:
            x, y, _ = batch
        else:
            raise RuntimeError
        pred = self.model(x)
        return pred

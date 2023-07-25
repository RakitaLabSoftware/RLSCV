import torch.nn as nn
import lightning as pl
import torch


class InferenceRunner(pl.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def from_config(cls, cfg) -> Self:
        return cls()

    @torch.no_grad()
    def forward(self, data) -> Any:
        return self.model.forward()

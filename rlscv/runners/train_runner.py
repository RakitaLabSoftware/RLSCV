from typing import Any, Self
import lightning as pl
import torch.nn as nn


class TrainRunner(pl.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @classmethod
    def from_config(cls, cfg) -> Self:
        return cls()

    def forward(self, data) -> Any:
        return self.model.forward()

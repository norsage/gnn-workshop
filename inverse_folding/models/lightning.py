from typing import Any, Callable, Iterator

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from torch_geometric.data import Data
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy


def create_metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        metrics=[
            # AUROC(task="multiclass", num_classes=20),
            Accuracy(task="multiclass", num_classes=20),
        ],
        prefix=prefix,
    )


class BaseModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Lit(L.LightningModule):
    def __init__(
        self,
        model: BaseModel,
        optimizer: Callable[[Iterator[torch.nn.Parameter]], Optimizer],
    ):
        super().__init__()
        self.model = model
        self._optimizer = optimizer
        self.train_metrics = create_metrics("train_")
        self.val_metrics = create_metrics("val_")

    def training_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        return self._step(batch, is_train=True)

    def validation_step(self, batch: tuple[Data, Data], batch_idx: int) -> STEP_OUTPUT:
        return self._step(batch, is_train=False)

    def configure_optimizers(self) -> Any:
        return self._optimizer(self.model.parameters())

    def _step(self, batch: Data, is_train: bool) -> dict[str, Any]:
        logits = self.model.forward(x=batch.x, pos=batch.pos, batch=batch.batch)
        labels: torch.Tensor = batch.y
        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
        )

        batch_size = batch.batch.unique().numel()

        metrics = self.train_metrics if is_train else self.val_metrics
        prefix = "train" if is_train else "val"
        metrics(logits, labels)

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log_dict(
            metrics,  # type: ignore[arg-type]
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return {
            "loss": loss,
            "logits": logits,
            "batch": batch.batch,
        }

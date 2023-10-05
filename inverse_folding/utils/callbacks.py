from typing import Any, Literal, cast

import plotly.graph_objects as go
from aim import Figure, Text
from aim.pytorch_lightning import AimLogger
from inverse_folding.utils.constants import INDEX_TO_LETTER
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from numpy.typing import NDArray
from torch import Tensor
from torchmetrics.classification.confusion_matrix import ConfusionMatrix


class ConfusionMatrixPlotCallback(Callback):
    def __init__(
        self,
        class_labels: list[str],
        normalize: Literal["true", "pred", "all", "none"] | None = None,
        colorscale: str = "blugrn",
    ) -> None:
        super().__init__()
        self.class_labels = list(class_labels)
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=len(class_labels), normalize=normalize
        )
        self.colorscale = colorscale

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        y = batch.y
        assert isinstance(outputs, dict)
        y_hat = outputs["logits"]
        self.confusion_matrix.update(y_hat.argmax(dim=-1), y)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        fig_ = plot_confusion_matrix(
            self.confusion_matrix.compute().numpy(),  # type: ignore[func-returns-value]
            labels=self.class_labels,
            title="ConfusionMatrix",
            colorscale=self.colorscale,
        )

        aim_figure = Figure(fig_)
        logger = cast(AimLogger, trainer.logger)
        logger.experiment.track(aim_figure, name="confusion-matrix", step=trainer.current_epoch)
        self.confusion_matrix.reset()


class PrintSequences(Callback):
    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples
        self._aligned_sequences: list[str] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(outputs, dict)
        target = batch.y
        preds = cast(Tensor, outputs["logits"]).argmax(dim=-1)
        batch = outputs["batch"]

        for i in outputs["batch"].unique():
            true_sequence = "".join(
                (INDEX_TO_LETTER[k.item()] for k in target[outputs["batch"] == i])
            )
            predicted_sequence = "".join(
                (INDEX_TO_LETTER[k.item()] for k in preds[outputs["batch"] == i])
            )
            match_string = "".join(
                ("|" if a == b else " " for a, b in zip(true_sequence, predicted_sequence))
            )
            self._aligned_sequences.append(
                true_sequence + "\n" + match_string + "\n" + predicted_sequence
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the val epoch ends."""
        aim_text = Text("\n\n".join(self._aligned_sequences[: self.n_samples]))
        logger = cast(AimLogger, trainer.logger)
        logger.experiment.track(aim_text, name="true-vs-predicted", step=trainer.global_step)
        self._aligned_sequences = []


def plot_confusion_matrix(cm: NDArray, labels: list[str], title: str, colorscale: str) -> go.Figure:
    data = go.Heatmap(z=cm, y=labels, x=labels, colorscale=colorscale)
    value_annotation = "{:d}" if (cm > 1).any() else "{:.2f}"

    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "white"},
                    "text": value_annotation.format(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False,
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted class"},
        "yaxis": {"title": "Real class"},
        "annotations": annotations,
        "width": 900,
        "height": 650,
    }
    fig = go.Figure(data=data, layout=layout)
    return fig

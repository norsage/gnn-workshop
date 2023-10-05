from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


@dataclass
class DataModuleConfig:
    datadir: str
    train_csv: str
    val_csv: str
    test_csv: str
    batch_size: int
    # Callable[[Path, Path], Dataset]
    dataset_fn: Any


class DataModule(LightningDataModule):
    datadir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    batch_size: int
    dataset_fn: Callable[[Path, Path], Dataset]

    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(
        self,
        datadir: str,
        train_csv: str,
        val_csv: str,
        batch_size: int,
        dataset_fn: Callable[[Path, Path], Dataset],
    ) -> None:
        super().__init__()
        self.datadir = Path(datadir)
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.batch_size = batch_size
        self.dataset_fn = dataset_fn

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.dataset_fn(self.train_csv, self.datadir)
            self.val_dataset = self.dataset_fn(self.val_csv, self.datadir)
        elif stage == "validate":
            self.val_dataset = self.dataset_fn(self.val_csv, self.datadir)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

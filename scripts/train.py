from dataclasses import dataclass
from typing import Any, cast

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from inverse_folding.data.datamodule import DataModule, DataModuleConfig
from inverse_folding.models.lightning import Lit
from inverse_folding.utils.hydra_utils import (
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf


@dataclass
class TrainConfig:
    seed: int | None
    datamodule: DataModuleConfig
    lightning: Any
    trainer: Any
    callbacks: list[Any] | None
    logger: Any | None


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=TrainConfig)


@hydra.main(version_base=None, config_path="../config", config_name="train.yaml")
def train(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.seed:
        L.seed_everything(cfg.seed, workers=True)

    lit: Lit = hydra.utils.instantiate(cfg.lightning)
    callbacks: list[Callback] = instantiate_callbacks(cast(DictConfig, cfg.callbacks))
    logger: list[Logger] = instantiate_loggers(cast(DictConfig, cfg.logger))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    datamodule: DataModule = hydra.utils.instantiate(cfg.datamodule)

    log_hyperparameters(
        {
            "cfg": cfg,
            "datamodule": datamodule,
            "lightning": lit,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }
    )
    trainer.fit(
        model=lit,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train()

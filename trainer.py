# -*- coding: utf-8 -*-
""" 
This is the basic trainer implemented on pytorch_ligtning and hydra
"""
from omegaconf.dictconfig import DictConfig
from dataloader import DownscalingDataLoader
from model_wrapper import ModelWrapper

import torch
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from datetime import date

from utils.dataset_splits import kfold_split

import logging, coloredlogs

logger = logging.getLogger(__name__)


@hydra.main(config_path="CONFIGS", config_name="config")
def main(cfg: DictConfig):
    coloredlogs.install(level="INFO", logger=logger)

    start_date = date.fromisoformat(cfg.DATAS.start_date)
    end_date = date.fromisoformat(cfg.DATAS.end_date)
    cfg.num_years = end_date.year - start_date.year + 1
    cfg.start_year = start_date.year
    cfg.end_year = end_date.year
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    logger.info(f"Configs: \n{OmegaConf.to_yaml(cfg, resolve=True)}")
    num_dsets = cfg.num_years
    k_folds = cfg.num_folds
    splits = kfold_split(num_dsets, k_folds)

    total_metric = 0
    num_runned_folds = 0
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        dm = DownscalingDataLoader(
            cfg,
            train_idx,
            val_idx,
            test_idx,
        )

        model = ModelWrapper(cfg, dm.in_channels, dm.in_height, dm.in_width,
                             dm.out_features)

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), f"{cfg.model_save_dir}"),
            every_n_epochs=1,
            save_top_k=1,
            # save_last=True,
            monitor=cfg.monitor,
            # filename="checkpoint_{epoch}",
            filename=f"best_{fold_idx}",
            # auto_insert_metric_name=True,
        )
        early_stop_callback = EarlyStopping(monitor=cfg.monitor,
                                            min_delta=0.00,
                                            patience=10)
        tlogger = TensorBoardLogger(save_dir=os.getcwd(),
                                    version=".",
                                    name=f"fold_{fold_idx}",
                                    default_hp_metric=True)

        trainer = pl.Trainer(
            gpus=[cfg.gpu],
            max_epochs=cfg.max_epochs,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=tlogger)  # gpus=[1],
        trainer.fit(model, datamodule=dm)
        metric = trainer.test(ckpt_path="best", datamodule=dm)
        metric = metric[0]["hp_metric"]

        model_meta = {
            "x_processor": dm.x_processor,
            "y_processor": dm.y_processor,
            "grid_mask": dm.grid_mask,
            "out_features": dm.out_features,
            "lat": dm.lat,
            "lon": dm.lon,
            "config": cfg,
        }
        torch.save(model_meta, f"{cfg.model_save_dir}/metadata_{fold_idx}.pth")
        total_metric += metric
        num_runned_folds += 1

    logger.info("Done!")
    return total_metric / num_runned_folds


if __name__ == "__main__":
    main()
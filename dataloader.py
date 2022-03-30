# -*- coding: utf-8 -*-
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Optional
import xarray as xr
import numpy as np
from prettytable import PrettyTable

from selector.predictor_selector import PredictorSelector

from utils.preprocessing import SequentialProcessor

import logging

logger = logging.getLogger(__name__)


class DownscalingDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        selector: PredictorSelector,
        train_yrs_idx: list,
        val_yrs_idx: list,
        test_yrs_idx: list,
    ):
        super().__init__()
        in_noaug_filepath = cfg.in_noaug_filepath.format(region=cfg.region)
        in_aug_filepath = cfg.in_aug_filepath.format(region=cfg.region)
        out_noaug_filepath = cfg.out_noaug_filepath.format(region=cfg.region)
        out_aug_filepath = cfg.out_aug_filepath.format(region=cfg.region)
        out_mask_filepath = cfg.out_mask_filepath.format(region=cfg.region)

        # Data loading
        ## for input
        predictors_sub = selector.get_predictors_sub(cfg.bistr)
        x_noaug = xr.open_dataset(in_noaug_filepath)[predictors_sub]
        logger.info(
            f"Selected input factors: {len(predictors_sub)}-{predictors_sub}")
        logger.info(f"Shape of x_noaug: {x_noaug.to_array().shape}")
        ## for output
        y_noaug = xr.open_dataarray(out_noaug_filepath)
        logger.info(f"Shape of y_noaug: {y_noaug.shape}")

        # Non-aug dataset split
        years_all = np.arange(cfg.start_year, cfg.end_year + 1)
        train_years = years_all[train_yrs_idx]
        val_years = years_all[val_yrs_idx]
        test_years = years_all[test_yrs_idx]
        train_noaug_idx = [
            yr in train_years for yr in y_noaug.time.dt.year.data
        ]
        val_noaug_idx = [yr in val_years for yr in y_noaug.time.dt.year.data]
        test_noaug_idx = [yr in test_years for yr in y_noaug.time.dt.year.data]
        # do some logs
        ptb = PrettyTable()
        ptb.field_names = ["Dataset", "Years"]
        ptb.add_rows([["train", train_years], ["Validation", val_years],
                      ["test", test_years]])
        logger.info(f"Datasets split:\n {ptb}")
        logger.info(
            f"Split of no-aug dataset: #train - {sum(train_noaug_idx)}, #val - {sum(val_noaug_idx)}, #test - {sum(test_noaug_idx)}"
        )

        # process x and y
        ## grid mask
        grid_mask = np.load(out_mask_filepath)
        y_noaug.values[:, ~grid_mask] = np.nan  # REVIEW need or not
        # processors of x and y
        x_processor = SequentialProcessor(cfg.x_processor_names)
        logger.info(f"x processor names: {cfg.x_processor_names}")
        y_processor = SequentialProcessor(cfg.y_processor_names)
        logger.info(f"y processor names: {cfg.y_processor_names}")
        ## fit processors of x and y
        x_processor.fit(x_noaug.isel(time=train_noaug_idx))
        y_processor.fit(y_noaug.isel(time=train_noaug_idx))
        ## peform process on x-noaug and y-noaug
        x_noaug_prcsd = x_processor.process(x_noaug).to_array().transpose(
            "time", "variable", "lat", "lon")
        y_noaug_prcsd = y_processor.process(y_noaug)

        if cfg.aug:
            logger.info("Use augmented dataset")
            # load augmented data
            ## for input
            x = xr.open_dataset(in_aug_filepath)[predictors_sub]
            logger.info(f"Shape of x_aug: {x.to_array().shape}")
            ## for output
            y = xr.open_dataarray(out_aug_filepath)
            logger.info(f"Shape of y_aug: {y.shape}")
            assert (x.time == y.time).all(), "Time of x and y should match"

            # find the split index
            train_aug_idx = [yr in train_years for yr in x.time.dt.year.data]
            logger.info(
                f"Number of augmented training samples: {len(train_aug_idx)}")

            # process x and y
            self.x_train = x_processor.process(
                x.isel(time=train_aug_idx)).to_array().transpose(
                    "time", "variable", "lat", "lon")
            self.y_train = y_processor.process(y.isel(time=train_aug_idx))
        else:
            logger.info("Use non-augmented dataset")
            assert (x_noaug.time == y_noaug.time
                    ).all(), "Time of x_noaug and y_noaug should match"
            # for train
            self.x_train = x_noaug_prcsd.isel(time=train_noaug_idx)
            self.y_train = y_noaug_prcsd.isel(time=train_noaug_idx)

        # val and test data
        if cfg.val_aug and cfg.aug:
            raise NotImplementedError()
        else:
            self.x_val = x_noaug_prcsd.isel(time=val_noaug_idx)
            self.y_val = y_noaug_prcsd.isel(time=val_noaug_idx)

        if cfg.test_aug and cfg.aug:
            raise NotImplementedError()
        else:
            self.x_test = x_noaug_prcsd.isel(time=test_noaug_idx)
            self.y_test = y_noaug_prcsd.isel(time=test_noaug_idx)

        # fillna
        self.y_test.fillna(0.0)
        self.y_train.fillna(0.0)
        self.y_val.fillna(0.0)

        # do some logs about the dataset input and output
        ptb = PrettyTable()
        ptb.field_names = ["Dataset", "Input Shape", "Output Shape"]
        ptb.add_rows([["train", self.x_train.shape, self.y_train.shape],
                      ["Validation", self.x_val.shape, self.y_val.shape],
                      ["test", self.x_test.shape, self.y_test.shape]])
        logger.info(f"Datasets shapes :\n {ptb}")

        # for later use
        self.cfg = cfg
        self.out_features = grid_mask.sum()
        # metadata for the dataloader
        self.x_processor = x_processor
        self.y_processor = y_processor
        self.grid_mask = grid_mask
        self.lat = y_noaug.lat.values
        self.lon = y_noaug.lon.values
        self.in_height, self.in_width = self.x_test.shape[-2:]
        self.out_height, self.out_width = self.y_test.shape[-2:]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit" or stage is None:
            self.train_dset = TensorDataset(
                torch.from_numpy(self.x_train.values.astype(np.float32)),
                torch.from_numpy(self.y_train.values[:, self.grid_mask].astype(
                    np.float32)))

            self.val_dset = TensorDataset(
                torch.from_numpy(self.x_val.values.astype(np.float32)),
                torch.from_numpy(self.y_val.values[:, self.grid_mask].astype(
                    np.float32)))
        if stage == "test" or stage is None:
            self.test_dset = TensorDataset(
                torch.from_numpy(self.x_test.values.astype(np.float32)),
                torch.from_numpy(self.y_test.values[:, self.grid_mask].astype(
                    np.float32)))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dset,
                          batch_size=self.cfg.batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=self.cfg.shuffle)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dset,
                          batch_size=self.cfg.batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dset,
                          batch_size=self.cfg.batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=False)


if __name__ == "__main__":
    import coloredlogs
    # from mr_update_config import update_cfg
    coloredlogs.install(level="INFO")

    from utils.dataset_splits import kfold_split
    from omegaconf.omegaconf import OmegaConf
    # 1. cfg
    cfg = OmegaConf.load("CONFIGS/config.yaml")
    cfg.region = "SC"

    logger.info(f"Configs: \n {OmegaConf.to_yaml(cfg)}")
    # 2. train, val and test dataset spllits

    splits = kfold_split(cfg.num_years, cfg.num_folds)
    train_index, val_index, test_index = splits[0]  # input
    logger.info(
        f"Initial splits: #train - {len(train_index)}, #val - {len(val_index)}, #test - {test_index}"
    )
    selector = PredictorSelector()
    dm = DownscalingDataLoader(cfg, selector, train_index, val_index,
                               test_index)
    dm.setup("fit")
    for x, y in dm.train_dataloader():
        print(x.shape, y.shape)
    # dm.setup(stage=None)

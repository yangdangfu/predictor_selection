# -*- coding: utf-8 -*-
""" This module will wrap the CNN architectures in architectures.general_cnn.py into a unified reusable model """

from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
from architectures import general_cnn
import torch.nn as nn


class ModelWrapper(pl.LightningModule):
    def __init__(self, cfg: DictConfig, in_channels: int, in_height: int,
                 in_width: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        if isinstance(cfg.seed, int):
            torch.manual_seed(cfg.seed)
        self.cfg = cfg

        grid_mask = np.load(cfg.out_mask_filepath.format(region=cfg.region))
        out_features = grid_mask.sum()
        self.model = general_cnn.instantiate_model(cfg.model,
                                                   in_channels,
                                                   in_height,
                                                   in_width,
                                                   out_features,
                                                   padding=cfg.padding)

        # loss function
        self.loss_func = nn.MSELoss()
        self.out_features = out_features

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_func(output, y)
        self.log("loss_train", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_func(output, y)
        self.log("loss_val", loss, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_func(output, y)
        self.log("hp_metric", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        if self.cfg.optim_type == "sgd":
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.cfg.lr,
                                  weight_decay=self.cfg.weight_decay,
                                  momentum=self.cfg.momentum)
        elif self.cfg.optim_type == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError()
        return optimizer


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import torch
    cfg = OmegaConf.load("CONFIGS/config.yaml")

    bs = 32  # batch size
    c, h, w = 20, 6, 8  # channels; in width; in height, out width, out height
    x = torch.rand(size=(bs, c, h, w))
    for model_cls_name in [
            "CNN_LM", "CNN_FC", "CNN1", "CNN10", "CNN_PR", "CNNdense"
    ]:
        cfg.model = model_cls_name
        cnn = ModelWrapper(cfg, c, h, w)
        num_params = [torch.numel(param) for param in cnn.parameters()]
        print(model_cls_name, sum(num_params), num_params)

        out = cnn(x)
        print(out.shape)

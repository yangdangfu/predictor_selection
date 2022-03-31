# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import xarray as xr
import numpy as np
import torch
from typing import Union
from captum.attr import GuidedBackprop
import torch.nn as nn


class AggModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)
        return output.sum(axis=1)


def calc_weights_gbp(model: Union[pl.LightningDataModule, list],
                     metadata: dict, raw_input: xr.Dataset):
    x_processor = metadata["x_processor"]
    x_prcsd = x_processor.process(raw_input).to_array().transpose(
        "time", "variable", "lat", "lon")
    # do forward pass
    model = AggModel(model)
    model.eval()
    x_prcsd = torch.from_numpy(x_prcsd.values.astype(np.float32))
    x_prcsd.requires_grad = True

    guided_grads = GuidedBackprop(model)
    guided_grads = guided_grads.attribute(x_prcsd)
    index = np.abs(guided_grads).sum(axis=(2, 3))
    time_index = raw_input.time
    # do reverse-preprocessing
    weights = xr.DataArray(data=index,
                           dims=["time", "variables"],
                           coords=dict(time=time_index,
                                       variables=list(raw_input.data_vars)))
    return weights

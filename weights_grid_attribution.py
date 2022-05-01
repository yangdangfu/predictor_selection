# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import xarray as xr
import numpy as np
import torch
from typing import Union
from captum.attr import GuidedBackprop
import torch.nn as nn


class GridModel(nn.Module):
    def __init__(self, model, grid_idx):
        super().__init__()
        self.model = model
        self.grid_idx = grid_idx

    def forward(self, input):
        output = self.model(input)
        return output[:, self.grid_idx]


def calc_weights_gbp(model: Union[pl.LightningDataModule, list],
                     metadata: dict, raw_input: xr.Dataset):
    x_processor = metadata["x_processor"]
    x_prcsd = x_processor.process(raw_input).to_array().transpose(
        "time", "variable", "lat", "lon")
    time_index = raw_input.time
    num_grid = metadata["grid_mask"].sum()

    grads_all = list()
    for grid_idx in range(num_grid):
        x = torch.from_numpy(x_prcsd.values.astype(np.float32))
        x.requires_grad = True
        agg_model = GridModel(model, grid_idx)
        agg_model.eval()
        # do forward pass
        guided_grads = GuidedBackprop(agg_model)
        guided_grads = guided_grads.attribute(x)

        grads_all.append(guided_grads.numpy())

    grads_all = np.stack(grads_all, axis=2)

    weights = xr.DataArray(data=grads_all,
                           dims=["time", "variables", "grid", "lat", "lon"],
                           coords=dict(time=time_index,
                                       variables=list(raw_input.data_vars),
                                       grid=list(range(num_grid)),
                                       lat=list(raw_input.lat.values),
                                       lon=list(raw_input.lon.values)))

    return weights

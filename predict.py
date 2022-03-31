# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import xarray as xr
import numpy as np
import torch
from typing import Union


def predit(model: Union[pl.LightningDataModule, list], metadata: dict,
           raw_input: xr.DataArray):
    """ Given a model and the conterpart metadata, the function will calculate the model output and do reverse preprocess to obtain and return the predicted grids precipitation
    """
    lat = metadata["lat"]
    lon = metadata["lon"]

    x_processor = metadata["x_processor"]
    y_processor = metadata["y_processor"]
    grid_mask = metadata["grid_mask"]
    x_prcsd = x_processor.process(raw_input).to_array().transpose(
        "time", "variable", "lat", "lon")
    # do forward pass
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_prcsd.values.astype(
            np.float32))).numpy().squeeze()

    time_index = raw_input.time

    pred_grid = np.empty((pred.shape[0], lat.shape[0], lon.shape[0]),
                         np.float32)
    pred_grid[:, grid_mask] = pred
    pred_grid[:, ~grid_mask] = np.nan
    # do reverse-preprocessing
    y_pred = xr.DataArray(data=pred_grid,
                          dims=["time", "lat", "lon"],
                          coords=dict(time=time_index, lat=lat, lon=lon))
    y_pred = y_processor.reverse(y_pred)
    return y_pred

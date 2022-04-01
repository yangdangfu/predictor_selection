# -*- coding: utf-8 -*-
""" 
Predict results with standardized inputs (not raw inputs) with sklearn machine learning models
"""
import numpy as np
import xarray as xr


def predict_from_std(models, metadata, std_inputs):
    pred_list = list()
    for mdl, std_input in zip(models, std_inputs):
        pred = mdl.predict(std_input.values.reshape((std_input.shape[0], -1)))
        pred_list.append(pred)
    pred = np.stack(pred_list, axis=1)

    y_processor = metadata["y_processor"]
    grid_mask = metadata["grid_mask"]
    lat = metadata["lat"]
    lon = metadata["lon"]

    assert pred.shape[1] == (grid_mask).sum()

    time_index = std_inputs[0].time
    time_len = len(time_index)
    y_pred = np.empty(shape=(time_len, lat.shape[0], lon.shape[0]))
    y_pred[:] = np.nan
    np.place(y_pred, np.tile(grid_mask, (time_len, 1, 1)), pred)
    # do reverse-preprocessing
    y_pred = xr.DataArray(data=y_pred,
                          dims=["time", "lat", "lon"],
                          coords=dict(time=time_index, lat=lat, lon=lon))
    y_pred = y_processor.reverse(y_pred)
    return y_pred

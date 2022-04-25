# -*- coding: utf-8 -*-
"""
Usage: 
    python main_draw_cc.py
"""
import logging
from os import path
from typing import Literal
from omegaconf import OmegaConf
import xarray as xr
from utils.get_rundirs import get_rundirs
from weights_grid_attribution import GridModel, calc_weights_gbp
import os
from selector.predictor_selector import PredictorSelector
from model_wrapper import ModelWrapper
import numpy as np
from utils.dataset_splits import kfold_split
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from data_utils.map_data_utils import region_geometry
import cmaps
import imageio
from utils.preprocessing import StandardisationByMonth
import xskillscore as xs
from main_draw_grads import add_basemap, draw_weights

logger = logging.getLogger(path.basename(__file__))


def compute_cc(region: str) -> xr.DataArray:
    cfg = OmegaConf.load("CONFIGS/config.yaml")
    in_noaug_filepath = cfg.in_noaug_filepath.format(region=region)
    out_noaug_filepath = cfg.out_noaug_filepath.format(region=region)
    out_mask_filepath = cfg.out_mask_filepath.format(region=region)
    # load data during 1981 to 2010
    predictors_sub = OmegaConf.to_container(cfg.candidate_predictors)
    x_noaug = xr.open_dataset(in_noaug_filepath)[predictors_sub].sel(
        time=slice("1981-01-01", "2010-12-31"))
    y_noaug = xr.open_dataarray(out_noaug_filepath).sel(
        time=slice("1981-01-01", "2010-12-31"))
    # perform preprocessing
    # y_noaug = y_noaug.sum(["lat", "lon"])
    x_preprocessor = StandardisationByMonth()
    x_preprocessor.fit(x_noaug)
    x_noaug = x_preprocessor.process(x_noaug)
    y_preprocessor = StandardisationByMonth()
    y_preprocessor.fit(y_noaug)
    y_noaug = y_preprocessor.process(y_noaug)
    # compute correlation coefficients
    grid_mask = np.load(out_mask_filepath)
    num_grid = grid_mask.sum()

    grid_idx = 0
    cc_all = list()
    for lat_idx in range(grid_mask.shape[0]):
        for lon_idx in range(grid_mask.shape[1]):
            if grid_mask[lat_idx, lon_idx]:
                cc = xs.pearson_r(x_noaug,
                                  y_noaug.isel(lat=lat_idx, lon=lon_idx),
                                  dim="time")
                grid_idx += 1
                cc_all.append(cc)

    weights_cc = xr.concat(cc_all, dim="grid")
    weights_cc["grid"] = np.arange(num_grid)
    weights_cc = weights_cc.to_array("variables").transpose(
        "variables", "grid", "lat", "lon")

    return weights_cc

if __name__ == "__main__":
    region_ = "SC"

    weights_cc = compute_cc(region_)
    draw_weights(weights_cc, region_, "cc", "IMAGES/CORR")

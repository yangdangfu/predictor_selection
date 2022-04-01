# -*- coding: utf-8 -*-
"""
Fitting, evaluation and other processes for linear regression
"""
from asyncio.log import logger
from sklearn import linear_model
from omegaconf import OmegaConf
from selector.predictor_selector import PredictorSelector
import xarray as xr
from scipy.spatial import cKDTree
import numpy as np
from utils.dataset_splits import kfold_split
from utils.preprocessing import SequentialProcessor
from predict_ml import predict_from_std
import os
from os import path
import pickle
from score import get_score_spatial, get_score_temporal
import logging, coloredlogs

import fire

logger = logging.getLogger(path.basename(__file__))


def ml_pred_eval(re_pred: bool, re_score: bool, region: str, bistr: str):
    mmodel_name = "LM"  # give a short name to the model
    num_neighbor = 2

    cfg = OmegaConf.load("CONFIGS/config.yaml")
    start_year = cfg.start_year
    end_year = cfg.end_year
    selector = PredictorSelector(cfg.candidate_predictors)
    input_factors_sub = selector.get_predictors_sub(bistr)
    num_folds = cfg.num_folds
    num_years = cfg.num_years
    x_processor_names = cfg.x_processor_names
    y_processor_names = cfg.y_processor_names

    in_noaug_filepath = cfg.in_noaug_filepath.format(region=region)
    out_noaug_filepath = cfg.out_noaug_filepath.format(region=region)
    out_mask_filepath = cfg.out_mask_filepath.format(region=region)
    x_noaug = xr.open_dataset(in_noaug_filepath)[input_factors_sub[:]]
    y_noaug = xr.open_dataarray(out_noaug_filepath)
    y_mask = np.load(out_mask_filepath)

    xlat, xlon = x_noaug.lat.values, x_noaug.lon.values
    ylat, ylon = y_noaug.lat.values, y_noaug.lon.values

    #* create kd-tree and search the nearest grid(s)
    lat_kd = cKDTree(data=xlat[:, np.newaxis])
    lon_kd = cKDTree(data=xlon[:, np.newaxis])
    _, near_lat_idx = lat_kd.query(ylat[:, np.newaxis], k=num_neighbor)
    _, near_lon_idx = lon_kd.query(ylon[:, np.newaxis], k=num_neighbor)

    DSET_TYPES = ["train", "val", "test"]
    save_dir = F"outputs_{region}/{mmodel_name}{num_neighbor*num_neighbor}_{bistr}/prediction"
    y_obs_fpath = {
        dset: path.join(save_dir, f"y_obs_{dset}.nc")
        for dset in DSET_TYPES
    }
    y_pred_fpath = {
        dset: path.join(save_dir, f"y_pred_{dset}.nc")
        for dset in DSET_TYPES
    }
    score_spatial_fpath = {
        dset: path.join(save_dir, f"score_spatial_{dset}.pkl")
        for dset in DSET_TYPES
    }
    score_temporal_fpath = {
        dset: path.join(save_dir, f"score_temporal_{dset}.pkl")
        for dset in DSET_TYPES
    }
    # variables to save results of all folds
    y_obs_all, y_pred_all = {dset: list()
                             for dset in DSET_TYPES
                             }, {dset: list()
                                 for dset in DSET_TYPES}

    need_pred, need_score = re_pred, re_score
    if not path.exists(save_dir):
        need_pred, need_score = True, True
        os.makedirs(save_dir, exist_ok=False)
    else:
        for dset in DSET_TYPES:
            if not (path.exists(y_obs_fpath[dset])
                    and path.exists(y_pred_fpath[dset])):
                need_pred = True
            if not (path.exists(score_spatial_fpath[dset])
                    and path.exists(score_temporal_fpath[dset])):
                need_score = True
    if not (need_pred or need_score):
        logger.warning(f"Eval aready done. Skip the eval in {save_dir}")
        return

    if need_pred:
        # dataset split by year
        splits = kfold_split(num_years, num_folds)
        years_all = np.arange(start_year, end_year + 1)
        for split in splits:
            raw_inputs, raw_outputs = dict.fromkeys(DSET_TYPES), dict.fromkeys(
                DSET_TYPES)
            for dset_idx, dset in enumerate(DSET_TYPES):
                yrs_idx = split[dset_idx]
                years = years_all[yrs_idx]
                time_index = [yr in years for yr in y_noaug.time.dt.year.data]
                raw_inputs[dset] = x_noaug.isel(time=time_index)
                raw_outputs[dset] = y_noaug.isel(time=time_index)

            # preprocessing
            x_processor = SequentialProcessor(x_processor_names)
            y_processor = SequentialProcessor(y_processor_names)
            ## fit processors of x and y
            x_processor.fit(raw_inputs["train"])
            y_processor.fit(raw_outputs["train"])
            ## peform process on x-noaug and y-noaug
            std_inputs, std_outputs = dict.fromkeys(DSET_TYPES), dict.fromkeys(
                DSET_TYPES)
            for dset in DSET_TYPES:
                std_inputs[dset] = x_processor.process(
                    raw_inputs[dset]).to_array().transpose(
                        "time", "variable", "lat", "lon")
                std_outputs[dset] = y_processor.process(raw_outputs[dset])

            metadata = {
                "x_processor": x_processor,
                "y_processor": y_processor,
                "grid_mask": y_mask,
                "lat": ylat,
                "lon": ylon,
            }

            #
            models = list()
            std_grids_inputs = {dset: list() for dset in DSET_TYPES}
            for lat_idx, lat in enumerate(ylat):
                for lon_idx, lon in enumerate(ylon):
                    # if the target grid is a nan
                    if not y_mask[lat_idx, lon_idx]:
                        continue
                    # find indices of x based on y using already done nearest search
                    xlat_idx = near_lat_idx[lat_idx]
                    xlon_idx = near_lon_idx[lon_idx]
                    std_x = {
                        dset: std_inputs["train"].isel(lat=xlat_idx,
                                                       lon=xlon_idx)
                        for dset in DSET_TYPES
                    }
                    # select data for training
                    x_train = std_x["train"]
                    y_train = std_outputs["train"].sel(lat=lat, lon=lon)
                    # fit model
                    reg = linear_model.LinearRegression()
                    reg.fit(x_train.values.reshape((x_train.shape[0], -1)),
                            y_train.values)
                    models.append(reg)

                    # prepare data for prediction
                    for dset in DSET_TYPES:
                        std_grids_inputs[dset].append(std_x[dset])

            for dset in DSET_TYPES:
                y_pred = predict_from_std(models, metadata,
                                          std_grids_inputs[dset])
                y_pred_all[dset].append(y_pred)

        # combine all the test&val prediction together
        for dset in DSET_TYPES:
            y_pred_all[dset] = xr.concat(y_pred_all[dset],
                                         dim='time').sortby("time").transpose(
                                             "time", "lat", "lon")
            if dset == "train":
                y_pred_all[dset] = y_pred_all[dset].groupby("time").mean()

            y_obs_all[dset] = y_noaug.sel(time=y_pred_all[dset].time)
            y_pred_all[dset].to_netcdf(y_pred_fpath[dset])
            y_obs_all[dset].to_netcdf(y_obs_fpath[dset])
    else:
        for dset in DSET_TYPES:
            y_pred_all[dset] = xr.open_dataarray(y_pred_fpath[dset])
            y_obs_all[dset] = xr.open_dataarray(y_obs_fpath[dset])

    # score
    # climatology for compute scores
    if need_score:
        y_clim = (y_noaug.sel(time=slice("1981-01-01", "2010-12-31")).resample(
            time="MS").mean())
        y_clim_std = y_clim.groupby("time.month").std()
        y_clim_mean = y_clim.groupby("time.month").mean()
        for dset in DSET_TYPES:
            s_spatial = get_score_spatial(y_pred_all[dset], y_obs_all[dset],
                                          y_clim_mean, y_clim_std)
            s_temporal = get_score_temporal(y_pred_all[dset], y_obs_all[dset],
                                            y_clim_mean, y_clim_std)
            # save
            with open(score_spatial_fpath[dset], "wb") as f:
                pickle.dump(s_spatial, f)

            with open(score_temporal_fpath[dset], "wb") as f:
                pickle.dump(s_temporal, f)


if __name__ == "__main__":
    coloredlogs.install(level="INFO", logger=logger)
    # ml_pred_eval(False, True, "SC", "11111111111111111111")
    fire.Fire(ml_pred_eval)
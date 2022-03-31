# -*- coding: utf-8 -*-
""" 
Usages: 
    python main_eval.py --bistr 00000100010101010101 --region SC
"""
import os
from prettytable import PrettyTable
import torch
from score import get_score_spatial, get_score_temporal
from utils.get_rundirs import get_rundirs
import logging, coloredlogs
from omegaconf import OmegaConf
from model_wrapper import ModelWrapper
from selector.predictor_selector import PredictorSelector
import xarray as xr
from utils.dataset_splits import kfold_split
import numpy as np
from predict import predit
import pickle
from weights_attribution import calc_weights_gbp
import fire

logger = logging.getLogger(os.path.basename(__file__))


def eval_single(rundir: str):
    cfg = OmegaConf.load(os.path.join(rundir, "config.yaml"))
    # all predictor, evaluation and other results will be save into the `prediction` sub directory
    save_dir = os.path.join(rundir, "prediction")
    os.makedirs(save_dir, exist_ok=True)

    # TODO: check already-done prediction, evaluation, and other results, skip?

    # load input data
    selector = PredictorSelector(cfg.candidate_predictors)
    predictors_sub = selector.get_predictors_sub(cfg.bistr)
    x_noaug = xr.open_dataset(
        cfg.in_noaug_filepath.format(region=cfg.region))[predictors_sub]
    logger.info(
        f"Selected input factors: {len(predictors_sub)}-{predictors_sub}")
    logger.info(f"Shape of x_noaug: {x_noaug.to_array().shape}")
    # load y_obs for comparison
    y_noaug = xr.open_dataarray(
        cfg.out_noaug_filepath.format(region=cfg.region))
    # nan_mask = ~np.load(cfg.out_mask_filepath.format(region=region))
    logger.info(f"Shape of y_noaug: {y_noaug.shape}")

    # climatology for compute scores
    y_clim = (y_noaug.sel(time=slice("1981-01-01", "2010-12-31")).resample(
        time="MS").mean())
    y_clim_std = y_clim.groupby("time.month").std()
    y_clim_mean = y_clim.groupby("time.month").mean()

    # dataset split by year
    num_folds = cfg.num_folds
    num_dsets = cfg.num_years
    split_list = kfold_split(num_dsets, num_folds)
    years_all = np.arange(cfg.start_year, cfg.end_year + 1)

    models_dir = os.path.join(rundir, cfg.model_save_dir)

    DSET_TYPES = ["train", "val", "test"]
    # do prediction, extract observation
    y_obs_all, y_pred_all = {dset: list()
                             for dset in DSET_TYPES
                             }, {dset: list()
                                 for dset in DSET_TYPES}
    weights_all = {dset: list() for dset in DSET_TYPES}
    for fold_idx in range(len(split_list)):
        model = ModelWrapper.load_from_checkpoint(
            os.path.join(models_dir, f"best_{fold_idx}.ckpt"))
        metadata = torch.load(
            os.path.join(models_dir, f"metadata_{fold_idx}.pth"))

        for dset_idx, dset in enumerate(DSET_TYPES):
            yrs_idx = split_list[fold_idx][dset_idx]
            years = years_all[yrs_idx]
            time_index = [yr in years for yr in y_noaug.time.dt.year.data]
            raw_input = x_noaug.isel(time=time_index)
            # predict
            y_pred = predit(model, metadata, raw_input)
            y_obs = y_noaug.isel(time=time_index)
            y_obs_all[dset].append(y_obs)
            y_pred_all[dset].append(y_pred)
            # attribute weight
            weights = calc_weights_gbp(model, metadata, raw_input)
            weights_all[dset].append(weights)

    # concate and save the results
    for dset in DSET_TYPES:
        y_obs_all[dset] = xr.concat(y_obs_all[dset],
                                    dim='time').sortby("time").transpose(
                                        "time", "lat", "lon")
        y_pred_all[dset] = xr.concat(y_pred_all[dset],
                                     dim='time').sortby("time").transpose(
                                         "time", "lat", "lon")
        weights_all[dset] = xr.concat(weights_all[dset],
                                      dim='time').sortby("time").transpose(
                                          "time", "variables")
        # NOTE: for training dataset, there are overlaps of data among different folds
        if dset == "train":
            y_obs_all[dset] = y_obs_all[dset].groupby(
                "time").mean()  # mean operation has no influence to obs
            y_pred_all[dset] = y_pred_all[dset].groupby("time").mean()
            weights_all[dset] = weights_all[dset].groupby("time").mean()
        # save
        y_obs_all[dset].to_netcdf(os.path.join(save_dir, f"y_obs_{dset}.nc"))
        y_pred_all[dset].to_netcdf(os.path.join(save_dir, f"y_pred_{dset}.nc"))
        weights_all[dset].to_netcdf(
            os.path.join(save_dir, f"weights_{dset}.nc"))

    # calc & save scores
    for dset in DSET_TYPES:
        s_spatial = get_score_spatial(y_pred, y_obs, y_clim_mean, y_clim_std)
        s_temporal = get_score_temporal(y_pred, y_obs, y_clim_mean, y_clim_std)
        spatial_fpath = os.path.join(save_dir, f"score_spatial_{dset}.pkl")
        temporal_fpath = os.path.join(save_dir, f"score_temporal_{dset}.pkl")
        # save
        with open(spatial_fpath, "wb") as f:
            pickle.dump(s_spatial, f)

        with open(temporal_fpath, "wb") as f:
            pickle.dump(s_temporal, f)


def eval(**kwargs):
    """eval --> eval_single"""
    # get run dirs
    run_dirs = get_rundirs(root_dir=f"outputs_{kwargs['region']}",
                           req_kwargs_dict=kwargs,
                           opt_kwargs_dict=None)

    # do some pretty print about the run dirs to be evaluated
    tb = PrettyTable()
    tb.field_names = ["Index", "Run directory"]
    tb.add_rows([[idx, run] for idx, run in enumerate(run_dirs)])
    logger.info(f"All runs dirs to perform prediction:\n {tb}")

    # run and save predictions
    for rundir in run_dirs:
        eval_single(rundir)


if __name__ == "__main__":
    coloredlogs.install(level="INFO", logger=logger)
    fire.Fire(eval)
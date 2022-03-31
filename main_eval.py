# -*- coding: utf-8 -*-
""" 
Usages: 
    python main_eval.py --re_pred False --re_weight False --re_score False --bistr 00000100010101010101 --region SC
"""
import os
from os import path
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

logger = logging.getLogger(path.basename(__file__))


def eval_single(rundir: str,
                re_pred: bool = False,
                re_weight: bool = False,
                re_score: bool = False):
    cfg = OmegaConf.load(path.join(rundir, "config.yaml"))
    # all predictor, evaluation and other results will be save into the `prediction` sub directory
    save_dir = path.join(rundir, "prediction")
    DSET_TYPES = ["train", "val", "test"]
    # TODO: check already-done prediction, evaluation, and other results, skip?
    y_obs_fpath = {
        dset: path.join(save_dir, f"y_obs_{dset}.nc")
        for dset in DSET_TYPES
    }
    y_pred_fpath = {
        dset: path.join(save_dir, f"y_pred_{dset}.nc")
        for dset in DSET_TYPES
    }
    weights_fpath = {
        dset: path.join(save_dir, f"weights_{dset}.nc")
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

    need_pred, need_weight, need_score = re_pred, re_weight, re_score
    if not path.exists(save_dir):
        need_pred, need_weight, need_score = True, True, True
        os.makedirs(save_dir, exist_ok=False)
    else:
        for dset in DSET_TYPES:
            if not (path.exists(y_obs_fpath[dset])
                    and path.exists(y_pred_fpath[dset])):
                need_pred = True
            if not path.exists(weights_fpath[dset]):
                need_weight = True
            if not (path.exists(score_spatial_fpath[dset])
                    and path.exists(score_temporal_fpath[dset])):
                need_score = True
    if not (need_pred or need_weight or need_score):
        logger.warning(f"Eval aready done. Skip the eval in {rundir}")
        return

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

    models_dir = path.join(rundir, cfg.model_save_dir)
    # do prediction, extract observation
    y_obs_all, y_pred_all = {dset: list()
                             for dset in DSET_TYPES
                             }, {dset: list()
                                 for dset in DSET_TYPES}
    weights_all = {dset: list() for dset in DSET_TYPES}
    for fold_idx in range(len(split_list)):
        model = ModelWrapper.load_from_checkpoint(
            path.join(models_dir, f"best_{fold_idx}.ckpt"))
        metadata = torch.load(path.join(models_dir,
                                        f"metadata_{fold_idx}.pth"))

        for dset_idx, dset in enumerate(DSET_TYPES):
            yrs_idx = split_list[fold_idx][dset_idx]
            years = years_all[yrs_idx]
            time_index = [yr in years for yr in y_noaug.time.dt.year.data]
            raw_input = x_noaug.isel(time=time_index)
            # predict
            if need_pred:
                y_pred = predit(model, metadata, raw_input)
                y_obs = y_noaug.isel(time=time_index)
                y_obs_all[dset].append(y_obs)
                y_pred_all[dset].append(y_pred)
            # attribute weight
            if need_weight:
                weights = calc_weights_gbp(model, metadata, raw_input)
                weights_all[dset].append(weights)

    # concate and save the results
    for dset in DSET_TYPES:
        if need_pred:
            y_obs_all[dset] = xr.concat(y_obs_all[dset],
                                        dim='time').sortby("time").transpose(
                                            "time", "lat", "lon")
            y_pred_all[dset] = xr.concat(y_pred_all[dset],
                                         dim='time').sortby("time").transpose(
                                             "time", "lat", "lon")
        elif need_score:  # load data from file for score
            y_obs_all[dset] = xr.open_dataarray(y_obs_fpath[dset])
            y_pred_all[dset] = xr.open_dataarray(y_pred_fpath[dset])

        if need_weight:
            weights_all[dset] = xr.concat(weights_all[dset],
                                          dim='time').sortby("time").transpose(
                                              "time", "variables")
        # NOTE: for training dataset, there are overlaps of data among different folds
        if dset == "train":
            if need_pred:
                y_obs_all[dset] = y_obs_all[dset].groupby(
                    "time").mean()  # mean operation has no influence to obs
                y_pred_all[dset] = y_pred_all[dset].groupby("time").mean()
            if need_weight:
                weights_all[dset] = weights_all[dset].groupby("time").mean()
        # save
        if need_pred:
            y_obs_all[dset].to_netcdf(y_obs_fpath[dset])
            y_pred_all[dset].to_netcdf(y_pred_fpath[dset])
        if need_weight:
            weights_all[dset].to_netcdf(weights_fpath[dset])

    # calc & save scores
    if need_score:
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


def eval(re_pred: bool, re_weight: bool, re_score: bool, **kwargs):
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
        eval_single(rundir, re_pred, re_weight, re_score)


if __name__ == "__main__":
    # eval(False, False, True, region="SC")
    coloredlogs.install(level="INFO", logger=logger)
    fire.Fire(eval)
# -*- coding: utf-8 -*-
import xarray as xr
import xskillscore as xs
import numpy as np 


def get_score_spatial(
    pred: xr.DataArray,
    target: xr.DataArray,
    clim_mean: xr.DataArray,
    clim_std: xr.DataArray,
) -> dict:
    """Score rmse, nrmse, cc, acc, ncc"""
    pred_anom = pred.groupby("time.month") - clim_mean
    target_anom = target.groupby("time.month") - clim_mean
    pred_norm = pred_anom.groupby("time.month") / clim_std
    target_norm = target_anom.groupby("time.month") / clim_std

    rmse = xs.rmse(pred, target, dim=["lat", "lon"], skipna=True)
    # armse = xs.rmse(pred_anom, target_anom, dim=["lat", "lon"], skipna=True) # euqal to rmse
    nrmse = xs.rmse(pred_norm, target_norm, dim=["lat", "lon"], skipna=True)

    cc = xs.pearson_r(pred, target, dim=["lat", "lon"], skipna=True)
    acc = xs.pearson_r(pred_anom, target_anom, dim=["lat", "lon"], skipna=True)
    ncc = xs.pearson_r(pred_norm, target_norm, dim=["lat", "lon"], skipna=True)

    return dict(rmse=rmse, nrmse=nrmse, cc=cc, acc=acc, ncc=ncc)


def get_score_temporal(
    pred: xr.DataArray,
    target: xr.DataArray,
    clim_mean: xr.DataArray,
    clim_std: xr.DataArray,
) -> dict:
    """ Score tcc, atcc, ntcc and sef"""
    pred_anom = pred.groupby("time.month") - clim_mean
    target_anom = target.groupby("time.month") - clim_mean
    pred_norm = pred_anom.groupby("time.month") / clim_std
    target_norm = target_anom.groupby("time.month") / clim_std

    tcc = xs.pearson_r(pred, target, dim="time")
    ntcc = xs.pearson_r(pred_norm, target_norm, dim="time")
    atcc = xs.pearson_r(pred_anom, target_anom, dim="time")
    # SEF
    diff = pred - target
    diff.values = np.abs(diff.values)
    diff = clim_std - diff.groupby("time.month")
    sef = (diff > 0).where(diff.notnull()).sum(dim="time", skipna=False) / len(
        diff.time)  #((diff >= 0).sum() / (~diff.isnull()).sum()).item()

    return dict(tcc=tcc, ntcc=ntcc, atcc=atcc, sef=sef)
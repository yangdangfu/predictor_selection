# -*- coding: utf-8 -*-
# %%
import logging
from os import path
from typing import Literal
from omegaconf import OmegaConf
from sklearn import datasets
import xarray as xr
from utils.get_rundirs import get_rundirs
from weights_grid_attribution import calc_weights_gbp
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
from main_select import run_select

logger = logging.getLogger(path.basename(__file__))


def _eval_single(rundir: str, dset: str):
    cfg = OmegaConf.load(path.join(rundir, "config.yaml"))
    DSET_TYPES = ["train", "val", "test"]
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

    # dataset split by year
    num_folds = cfg.num_folds
    num_dsets = cfg.num_years
    split_list = kfold_split(num_dsets, num_folds)
    years_all = np.arange(cfg.start_year, cfg.end_year + 1)

    models_dir = path.join(rundir, cfg.model_save_dir)
    # do prediction, extract observation
    weights_all = list()  # {dset: list() for dset in DSET_TYPES}
    for fold_idx in range(len(split_list)):
        model = ModelWrapper.load_from_checkpoint(
            path.join(models_dir, f"best_{fold_idx}.ckpt"))
        metadata = torch.load(path.join(models_dir,
                                        f"metadata_{fold_idx}.pth"))

        dset_idx = DSET_TYPES.index(dset)
        yrs_idx = split_list[fold_idx][dset_idx]
        years = years_all[yrs_idx]
        time_index = [yr in years for yr in y_noaug.time.dt.year.data]
        raw_input = x_noaug.isel(time=time_index)
        # attribute weight
        weights = calc_weights_gbp(model, metadata, raw_input)
        weights_all.append(weights)

    # concate and save the results
    weights_all = xr.concat(weights_all, dim='time').sortby("time").transpose(
        "time", "variables", "grid", "lat", "lon")
    # note: for training dataset, there are overlaps of data among different folds
    if dset == "train":
        weights_all = weights_all.groupby("time").mean()
    return weights_all


def agg_weights(rundirs: list, dset: Literal["train", "val", "test"] = "val"):
    """Aggreate weights from multple runs"""
    weights_all = list()
    for rdir in rundirs:
        weights = _eval_single(rdir, dset)
        weights_all.append(weights)
    weights_avg = sum(weights_all) / len(
        weights_all)  # xr.DataArray, dim: time x predictors
    return weights_avg


# %%
def add_basemap(region: str, ax: plt.Axes, grid_idx: int):
    cfg = OmegaConf.load("CONFIGS/config.yaml")
    region_info = OmegaConf.to_container(cfg.REGIONS[region].shape_range)
    lat_range = OmegaConf.to_container(cfg.REGIONS[region].lat_range)
    lon_range = OmegaConf.to_container(cfg.REGIONS[region].lon_range)

    ax.set_extent((lon_range[0], lon_range[1], lat_range[0], lat_range[1]),
                  crs=ccrs.PlateCarree())

    if region_info is not None:
        gdf = region_geometry(**region_info)
        ax.add_geometries(gdf.boundary,
                          crs=ccrs.PlateCarree(),
                          facecolor="none",
                          edgecolor="k",
                          linewidth=1.0)

    # add a scatter point on image

    gm_fpath = f"DATA/precip_{region}_grid_mask.npy"
    y_fpath = f"DATA/precip_{region}_out_noaug.nc"
    x_fpath = f"DATA/precip_{region}_in_noaug.nc"
    msk = np.load(gm_fpath)
    x = xr.open_dataset(x_fpath)
    y = xr.open_dataarray(y_fpath)
    # plot input grids
    x_lons, x_lats = x.lon.values, x.lat.values
    x_lons, x_lats = np.meshgrid(x_lons, x_lats)
    ax.scatter(x_lons, x_lats, s=16, c="red")

    y_lons, y_lats = y.lon.values, y.lat.values
    idx = 0
    for lat_idx, lat in enumerate(y_lats):
        for lon_idx, lon in enumerate(y_lons):
            if not msk[lat_idx, lon_idx]:
                continue
            if idx == grid_idx:
                ax.scatter(lon, lat, s=36, c="c")
                # ax.text(lon, lat, f"{idx}", fontsize="small")
            else:
                ax.scatter(lon, lat, s=4, c="k")
                # ax.text(lon, lat, f"{idx}", fontsize="small")
            idx += 1


def draw_weights(weights_t_avg: xr.DataArray, region: str, bistr: str,
                 save_root: str):
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    lat, lon = weights_t_avg.lat.values, weights_t_avg.lon.values
    contour_kw_def = {
        "levels": np.linspace(-1, 1, 21),
        "cmap": cmaps.BlueRed,
        "vmin": -1,
        "vmax": 1,
    }

    # draw grads/cc of all predictors for each grid (grid-wise plots)
    for grid_idx in weights_t_avg.grid.values.tolist():
        if not grid_idx == 76:
            continue
        w_grid = weights_t_avg.sel(grid=grid_idx)
        vmax = np.fabs(w_grid).max().item()
        w_grid = w_grid / vmax
        contour_kw_def.update(vmin=-1, vmax=1)
        for var in weights_t_avg.variables.values.tolist():
            print(f"{bistr} {grid_idx} {var}")
            w = w_grid.sel(variables=var)
            fig, ax = plt.subplots(
                subplot_kw={"projection": ccrs.PlateCarree()},
                figsize=(6, 6),
                frameon=True,
                tight_layout=True,
                dpi=72)
            cf = ax.contourf(lon,
                             lat,
                             w.values.squeeze(),
                             transform=ccrs.PlateCarree(),
                             **contour_kw_def)
            add_basemap(region=region, ax=ax, grid_idx=grid_idx)
            # ax.set_title(f"{var}")
            fig.savefig(f"{save_root}/g{grid_idx}_{var}_{bistr}.png",
                        bbox_inches="tight")
            plt.close()
            # break

    # draw grads/cc of all grids for each predictor (predictor-wise), RESULTS will be a gif
    for var in weights_t_avg.variables.values.tolist():
        w_var = weights_t_avg.sel(variables=var)
        vmax = np.fabs(w_var).max().item()
        w_var = w_var / vmax
        contour_kw_def.update(vmin=-1, vmax=1)
        for grid_idx in weights_t_avg.grid.values.tolist():
            print(f"{bistr} {var} {grid_idx}")
            w = w_var.sel(grid=grid_idx)
            fig, ax = plt.subplots(
                subplot_kw={"projection": ccrs.PlateCarree()},
                figsize=(6, 6),
                frameon=True,
                tight_layout=True,
                dpi=72)
            cf = ax.contourf(lon,
                             lat,
                             w.values.squeeze(),
                             transform=ccrs.PlateCarree(),
                             **contour_kw_def)
            img_fname = f"{save_root}/{var}_g{grid_idx}_{bistr}.png"
            add_basemap(region=region, ax=ax, grid_idx=grid_idx)
            fig.savefig(img_fname, bbox_inches="tight")
            plt.close()

        # Build GIF
        with imageio.get_writer(f'{save_root}/{var}_{bistr}.gif',
                                mode='I') as writer:
            img_fnames = [
                f"{save_root}/{var}_g{grid_idx}_{bistr}.png"
                for grid_idx in weights_t_avg.grid.values.tolist()
            ]
            for filename in img_fnames:
                image = imageio.imread(filename)
                writer.append_data(image)

            for filename in img_fnames:
                os.remove(filename)


# %%
if __name__ == "__main__":
    region_ = "SC"
    model_ = "CNN10"
    bistr_ = "11111111111111111111"
    rundirs = get_rundirs(f"outputs_{region_}", {
        "region": region_,
        "model": model_,
        "bistr": bistr_
    })
    if not os.path.exists("weights_t_avg.nc"):
        weights_t_avg = agg_weights(rundirs).mean("time")
        weights_t_avg.to_netcdf("weights_t_avg.nc")
    else:
        weights_t_avg = xr.open_dataarray("weights_t_avg.nc")
    draw_weights(weights_t_avg, region_, bistr_, "IMAGES/GRADS")

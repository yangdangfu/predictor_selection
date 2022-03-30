# -*- coding: utf-8 -*-
"""
To speed up the succeeding training, evaluation and other processes, this script aim to prepare the data of predictors and predictand in the region of intrest, including both augmented and non-augmented data.

A region can be specified by two ways (one of or both): 
1. latitude and longitude rect ranges
2. boundary of conuntries, provinces/states

Examples: 
-  Yangtze: 
    lat_range: [27.5, 35] 
    lon_range: [110, 122.5] 
    shape_range: null
-  SC:
    lat_range: [15, 27.5] 
    lon_range: [102.5, 120] 
    shape_range: 
      region_level: PROVINCE 
      region_names: ["Guangdong", "Guangxi", "Hainan"]
"""
import os
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from datetime import date
from data_utils.ncep_data_utils import read_daily_cpc, read_daily_ncep, read_monthly_cpc, read_monthly_ncep, read_rolled_cpc, read_rolled_ncep
from data_utils.map_data_utils import region_geometry, region_mask
import fire


def load_daily_data(data_info: DictConfig, region_info: DictConfig):
    """Return a 3-tuple daily data out_da, grid_mask, in_ds
    """
    out_da = read_daily_cpc(
        factor=data_info.factor,
        start_date=date.fromisoformat(data_info.start_date),
        end_date=date.fromisoformat(data_info.end_date),
        lat_range=region_info.lat_range,
        lon_range=region_info.lon_range,
    )[data_info.factor]
    # ANCHOR output grids
    # first mask: grids that maybe not on the land
    grid_mask = ~(out_da.isnull().values.sum(axis=0) > 0)
    # second mask: grids that is not within the region (polygon)
    if region_info.shape_range is not None:
        lats = out_da.lat.values
        lons = out_da.lon.values
        gdf = region_geometry(region_info.shape_range.region_level,
                              region_info.shape_range.region_names)
        bbox = gdf.total_bounds  # bbox of the region [minx, miny, max, maxy]
        assert (
            bbox[0] > lons.min() and bbox[1] > lats.min()
            and bbox[2] < lons.max() and bbox[3] < lats.max()
        ), f"bbox of region boundary {bbox} exceeds the lat-lon ranges {(lons.min(), lats.min(), lons.max(), lats.max())}"
        rgn_mask = region_mask(lats, lons, gdf.geometry)
        grid_mask = rgn_mask & grid_mask
    out_da.values[:, ~grid_mask] = np.nan

    # ANCHOR inputs
    in_ds = read_daily_ncep(factors=data_info.input_factors,
                            start_date=date.fromisoformat(
                                data_info.start_date),
                            end_date=date.fromisoformat(data_info.end_date),
                            lat_range=region_info.lat_range,
                            lon_range=region_info.lon_range,
                            source=data_info.input_source)
    in_ds = in_ds.sel(time=out_da.time)

    return out_da, grid_mask, in_ds


def load_monthly_data(data_info: DictConfig, region_info: DictConfig):
    """ Return a 5-tuple out_noaug_da, out_aug_da, grid_mask, in_noaug_ds, in_aug_ds """
    out_aug_da = read_rolled_cpc(
        factor=data_info.factor,
        num_days=30,
        start_date=date.fromisoformat(data_info.start_date),
        end_date=date.fromisoformat(data_info.end_date),
        lat_range=region_info.lat_range,
        lon_range=region_info.lon_range,
    )[data_info.factor]
    out_noaug_da = read_monthly_cpc(
        factor=data_info.factor,
        start_date=date.fromisoformat(data_info.start_date),
        end_date=date.fromisoformat(data_info.end_date),
        lat_range=region_info.lat_range,
        lon_range=region_info.lon_range,
    )[data_info.factor]
    # ANCHOR output grids
    # first mask: grids that maybe not on the land
    land_mask_noaug = (out_noaug_da.isnull().values.sum(axis=0) > 0)
    land_mask_aug = (out_aug_da.isnull().values.sum(axis=0) > 0)
    grid_mask = ~(land_mask_aug | land_mask_noaug)
    # second mask: grids that is not within the region (polygon)
    if region_info.shape_range is not None:
        lats = out_noaug_da.lat.values
        lons = out_noaug_da.lon.values
        gdf = region_geometry(region_info.shape_range.region_level,
                              region_info.shape_range.region_names)
        bbox = gdf.total_bounds  # bbox of the region [minx, miny, max, maxy]
        assert (
            bbox[0] > lons.min() and bbox[1] > lats.min()
            and bbox[2] < lons.max() and bbox[3] < lats.max()
        ), f"bbox of region boundary {bbox} exceeds the lat-lon ranges {(lons.min(), lats.min(), lons.max(), lats.max())}"
        rgn_mask = region_mask(lats, lons, gdf.geometry)
        grid_mask = rgn_mask & grid_mask
    out_noaug_da.values[:, ~grid_mask] = np.nan
    out_aug_da.values[:, ~grid_mask] = np.nan

    # ANCHOR inputs
    in_aug_ds = read_rolled_ncep(
        factors=data_info.input_factors,
        num_days=30,
        start_date=date.fromisoformat(data_info.start_date),
        end_date=date.fromisoformat(data_info.end_date),
        lat_range=region_info.lat_range,
        lon_range=region_info.lon_range,
        source=data_info.input_source)
    in_noaug_ds = read_monthly_ncep(
        factors=data_info.input_factors,
        start_date=date.fromisoformat(data_info.start_date),
        end_date=date.fromisoformat(data_info.end_date),
        lat_range=region_info.lat_range,
        lon_range=region_info.lon_range,
        source=data_info.input_source)
    in_noaug_ds = in_noaug_ds.sel(time=out_noaug_da.time)
    in_aug_ds = in_aug_ds.sel(time=out_aug_da.time)

    return out_noaug_da, out_aug_da, grid_mask, in_noaug_ds, in_aug_ds


def prepare_data(region: str):
    """load and save data for given region

    Args:
        region (str): short name of the region, see CONFIGS/config.yaml --> REGIONS
    """
    cfg_ = OmegaConf.load("CONFIGS/config.yaml")
    os.makedirs(cfg_.data_path, exist_ok=True)
    region_info = cfg_.REGIONS[region]
    data_info = cfg_.DATAS

    # daily data
    out_da, grid_mask, in_ds = load_daily_data(data_info, region_info)
    out_daily_filepath = cfg_.out_daily_filepath.format(region=region)
    out_mask_daily_filepath = cfg_.out_mask_daily_filepath.format(
        region=region)
    in_daily_filepath = cfg_.in_daily_filepath.format(region=region)
    out_da.to_netcdf(out_daily_filepath)
    np.save(out_mask_daily_filepath, grid_mask)
    in_ds.to_netcdf(in_daily_filepath)
    # monthly data
    out_noaug_da, out_aug_da, grid_mask, in_noaug_ds, in_aug_ds = load_monthly_data(
        data_info, region_info)
    out_noaug_filepath = cfg_.out_noaug_filepath.format(region=region)
    out_aug_filepath = cfg_.out_aug_filepath.format(region=region)
    out_mask_filepath = cfg_.out_mask_filepath.format(region=region)
    in_aug_filepath = cfg_.in_aug_filepath.format(region=region)
    in_noaug_filepath = cfg_.in_noaug_filepath.format(region=region)
    out_noaug_da.to_netcdf(out_noaug_filepath)
    out_aug_da.to_netcdf(out_aug_filepath)
    np.save(out_mask_filepath, grid_mask)
    in_noaug_ds.to_netcdf(in_noaug_filepath)
    in_aug_ds.to_netcdf(in_aug_filepath)

if __name__ == "__main__":
    fire.Fire(prepare_data)
    # Usage
    # python main_data_preparation.py region=SC
    # python main_data_preparation.py region=Yangtze
    # ...
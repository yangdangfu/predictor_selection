# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from data_utils.map_data_utils import region_geometry
import cmaps
import cartopy.crs as ccrs

import numpy as np
import xarray as xr


def draw_dist(ax: plt.Axes,
              data: xr.DataArray,
              vmin: float,
              vmax: float,
              levels: int,
              cmap=cmaps.GMT_seis_r,
              region_info: dict = None,
              colors: list = None,
              extend=None):
    # geoax: geoaxes.GeoAxes = plt.subplot(ax, projection=ccrs.PlateCarree())
    geoax = ax

    contour_kw_def = {
        "levels": np.linspace(vmin, vmax, levels),
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "extend": extend,
        "colors": colors
    }

    # if data.min().item() <= vmin:
    #     contour_kw_def.update(extend="min")
    # if data.max().item() >= vmax:
    #     if contour_kw_def["extend"] == "min":
    #         contour_kw_def.update(extend="both")
    #     else:
    #         contour_kw_def.update(extend="max")

    lon, lat = data.lon.values, data.lat.values
    cf = geoax.contourf(lon,
                        lat,
                        data.values.squeeze(),
                        transform=ccrs.PlateCarree(),
                        **contour_kw_def)
    if region_info is not None:
        gdf = region_geometry(**region_info)
        geoax.add_geometries(gdf.boundary,
                             crs=ccrs.PlateCarree(),
                             facecolor="none",
                             edgecolor="dimgray",
                             linewidth=0.25)
    return cf
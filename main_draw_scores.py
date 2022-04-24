# -*- coding: utf-8 -*-
""" 
Usage: 
    # draw results of GBP-based predictor elimination
    python main_draw_scores.py --cc False --reverse_sel False --dset test --region SC --model CNN10 --multirun 10
    python main_draw_scores.py --cc False --reverse_sel True --dset test --region SC --model CNN10 --multirun 10
    # draw results of correlation-analysis-based predictor elimination
    python main_draw_scores.py --cc True --reverse_sel False --dset test --region SC --model CNN10 --multirun 10
The results are saved in folder ./IMAGES
"""

from os import path
import os

from omegaconf import OmegaConf
from utils.get_rundirs import get_rundirs
from main_select import run_select
from main_select_cc import run_select as run_select_cc
from plots.agg_scores import agg_score_spatial, agg_score_temporal
from plots.draw_boxplot import draw_boxplot
from plots.draw_dist import draw_dist
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cmaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fire


def calc_means(scores: list):
    """Compute means of scores"""
    score_types = scores[0].keys()

    scores_mean = dict.fromkeys(score_types)
    for s_type in score_types:
        scores_mean[s_type] = np.array(
            [values[s_type].mean().item() for values in scores])
    return scores_mean


# ANCHOR line plots of means of cc, atcc, rmse
def line_plots(scores_mean_cnn: dict,
               scores_mean_ml: dict,
               reverse_sel: bool, 
               fname_suffix: str):
    # plot style
    markersize = 100

    score_types = scores_mean_cnn.keys()
    score_types = [x for x in score_types if x in ["rmse", "atcc", "cc"]]
    for s_type in score_types:
        means = scores_mean_cnn[s_type]
        num = len(means)
        fig, ax = plt.subplots(figsize=(9.6, 6.4))
        ax.plot(means, marker=".", c="tab:orange", label="CNN")
        ax.axhline(y=means[0], c="tab:orange", ls=":", label="CNN reference")
        if not reverse_sel:  # ugly code
            # better score scatters
            if "rmse" in s_type:
                better_score = [(i, x) for i, x in enumerate(means)
                                if x <= means[0]]
                least = better_score[-1]
                best = (means.tolist().index(min(means)), min(means))
            else:
                better_score = [(i, x) for i, x in enumerate(means)
                                if x >= means[0]]
                least = better_score[-1]
                best = (means.tolist().index(max(means)), max(means))
            # Better
            ax.scatter(
                [better_score[i][0] for i in range(1, len(better_score))],
                [better_score[i][1] for i in range(1, len(better_score))],
                c="tab:green",
                label="Better")
            # BEST and LEST
            ax.scatter(least[0],
                       least[1],
                       s=markersize,
                       marker="v",
                       c="tab:green",
                       label="BEST")
            ax.scatter(best[0],
                       best[1],
                       s=markersize,
                       marker="^",
                       c="tab:green",
                       label="LEAST")

        if scores_mean_ml is not None:
            ax.plot(scores_mean_ml[s_type],
                    marker=".",
                    c="tab:blue",
                    label="LR")
            ax.axhline(y=scores_mean_ml[s_type][0],
                       c="tab:blue",
                       ls=":",
                       label="LR reference")

        # labels, ticks, legend, etc.
        ax.set_xticks(list(range(num)))
        ax.set_xticklabels([str(num - i) for i in range(num)])
        ax.set_xlabel("Number of predictors")
        ax.set_ylabel(s_type.upper())
        ax.legend()

        # save file
        fname = f"IMAGES/line_{s_type}_{fname_suffix}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=320)


def draw(cc: bool, reverse_sel: bool, dset: str, multirun: int, region: str,
         model: str):
    """draw results 

    Args:
        cc (bool): use correlation-analysis-based method or not
        reverse_sel (bool): reverse elimination or not
        dset (str): draw results on "val" or "test" dataset
        multirun (int): number of runs
        region (str): region name
        model (str): model name
    """
    # Obtain the binary strings of selected predictors
    if not cc:
        bi_strings = run_select(reverse_sel=reverse_sel,
                                multirun=multirun,
                                re_pred=False,
                                re_weight=False,
                                re_score=False,
                                skip_runs=True,
                                region=region,
                                model=model)
    else:
        bi_strings = run_select_cc(reverse_sel=reverse_sel,
                                   multirun=multirun,
                                   re_pred=False,
                                   re_weight=False,
                                   re_score=False,
                                   skip_runs=True,
                                   region=region,
                                   model=model)

    fname_dict = dict()
    fname_dict.update(region=region,
                      model=model,
                      reverse_sel=reverse_sel,
                      dset=dset)
    fname_suffix = ""
    if fname_dict is not None:
        for key, val in fname_dict.items():
            fname_suffix += f"_{key}-{val}"
    if cc:
        # Add _CORR to the end of filenames for correlation-analysis-based plots
        fname_suffix += "_CORR" 

    os.makedirs("./IMAGES", exist_ok=True)
    # Collect (averaged) scores
    # all scores are of type dict of dict of xarray
    # keys: bistr -> score_type --> xr.DataArray
    scores_spatial_cnn = list()
    scores_spatial_ml = list()
    scores_temporal_cnn = list()
    scores_temporal_ml = list()
    for bistr in bi_strings:
        run_kwargs = dict(bistr=bistr, region=region, model=model)
        # run dirs
        rootdir = f"outputs_{region}"
        rundirs_cnn = get_rundirs(rootdir,
                                  run_kwargs)  # multiple-run dirs of CNN

        # load scores
        scores_spatial_cnn.append(agg_score_spatial(rundirs_cnn, dset))
        scores_temporal_cnn.append(agg_score_temporal(rundirs_cnn, dset))
        rundir_ml = path.join(rootdir, f"LM4_{bistr}")  # single-run dir of LM
        scores_spatial_ml.append(agg_score_spatial(rundir_ml, dset))
        scores_temporal_ml.append(agg_score_temporal(rundir_ml, dset))

    mean_spatial_cnn = calc_means(scores_spatial_cnn)
    mean_temporal_cnn = calc_means(scores_temporal_cnn)
    mean_spatial_ml = calc_means(scores_spatial_ml)
    mean_temporal_ml = calc_means(scores_temporal_ml)

    # line plots
    line_plots(mean_spatial_cnn, mean_spatial_ml, reverse_sel, fname_suffix)  #
    line_plots(mean_temporal_cnn, mean_temporal_ml, reverse_sel, fname_suffix)  #

    # no box and dist plot for reverse experiments
    if reverse_sel or cc:
        return

    # collect reference, best, and least models
    # find BEST and LEAST model based on RMSE
    s_type = "rmse"
    means = mean_spatial_cnn[s_type]
    better_score = [(i, x) for i, x in enumerate(means) if x <= means[0]]
    # indices is a tuple of the indices to reference, best, and LEAST models, resp.
    indices = (0, means.tolist().index(min(means)), better_score[-1][0])
    scores_spatial = [scores_spatial_cnn[i] for i in indices]
    scores_temporal = [scores_temporal_cnn[i] for i in indices]

    labels = ["Reference", "BEST", "LEAST"]
    colors = ["#faa", "lightgreen", "lightblue"]
    # box plot
    for s_type in ["rmse", "cc"]:
        scores = [s[s_type] for s in scores_spatial]
        fig, ax = plt.subplots(figsize=(9.6, 6.4))
        datas = [s.values for s in scores]
        draw_boxplot(ax, datas, labels, colors, quantile_ref=True)
        fig.savefig(f"IMAGES/boxplot_{s_type}_{fname_suffix}.png",
                    bbox_inches="tight",
                    dpi=320)

    # box plot and dist plot
    for s_type in ["atcc"]:
        scores = [s[s_type] for s in scores_temporal]
        fig, ax = plt.subplots(figsize=(9.6, 6.4))
        datas = [s.values.flatten() for s in scores]
        datas = [d[~np.isnan(d)] for d in datas]
        draw_boxplot(ax, datas, labels, colors, quantile_ref=True)
        fig.savefig(f"IMAGES/boxplot_{s_type}_{fname_suffix}.png",
                    bbox_inches="tight",
                    dpi=320)

        # dist plot
        cfg = OmegaConf.load("CONFIGS/config.yaml")
        region_info = cfg.REGIONS[region].shape_range
        # region_info = dict(region_level="PROVINCE",
        #                    region_names=["Guangdong", "Guangxi", "Hainan"])
        fig, axes = plt.subplots(nrows=1,
                                 ncols=3,
                                 subplot_kw={"projection": ccrs.PlateCarree()},
                                 figsize=(12, 3.6),
                                 frameon=True)
        box_fig, box_ax = plt.subplots(figsize=(4.8, 6.4))
        # for score in [
        #         score_temporal_ref, score_temporal_least, score_temporal_best
        # ]:
        cf = draw_dist(ax=axes[0],
                       data=scores[0],
                       cmap=cmaps.BlAqGrYeOrReVi200,
                       vmin=0,
                       vmax=1,
                       levels=11,
                       region_info=region_info,
                       extend="both")
        # Colorbar
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("bottom",
                                  size="3.3%",
                                  pad=0.05,
                                  axes_class=plt.Axes)
        fig.colorbar(cf,
                     cax=cax,
                     orientation="horizontal",
                     ticks=np.linspace(0.0, 1.0, 6))

        bias_best = scores[1] - scores[0]
        cf = draw_dist(ax=axes[1],
                       data=bias_best,
                       cmap=cmaps.BlueRed,
                       vmin=-0.05,
                       vmax=0.05,
                       levels=11,
                       region_info=region_info,
                       extend="both")
        divider = make_axes_locatable(axes[1])
        # Colorbar
        cax = divider.append_axes("bottom",
                                  size="3.3%",
                                  pad=0.05,
                                  axes_class=plt.Axes)
        fig.colorbar(cf,
                     cax=cax,
                     orientation="horizontal",
                     ticks=np.linspace(-0.05, 0.05, 6))

        bias_least = scores[2] - scores[0]
        cf = draw_dist(ax=axes[2],
                       data=bias_least,
                       cmap=cmaps.BlueRed,
                       vmin=-0.05,
                       vmax=0.05,
                       levels=11,
                       region_info=region_info,
                       extend="both")
        # Colorbar
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("bottom",
                                  size="3.3%",
                                  pad=0.05,
                                  axes_class=plt.Axes)

        # fig.add_axes(cax)
        fig.colorbar(cf,
                     cax=cax,
                     orientation="horizontal",
                     ticks=np.linspace(-0.05, 0.05, 6))

        # Boxplots
        draw_boxplot(box_ax,
                     datas=[
                         bias_best.values[~bias_best.isnull().values],
                         bias_least.values[~bias_best.isnull().values]
                     ],
                     colors=["#faa", "lightgreen"],
                     labels=["BEST", "LEAST"])

        box_fig.savefig(f"IMAGES/boxplot_atcc_bias_{fname_suffix}.png",
                        bbox_inches="tight",
                        dpi=320)
        fig.savefig(f"IMAGES/dist_atcc_{fname_suffix}.png",
                    bbox_inches="tight",
                    dpi=320)


# def main(**run_kwargs):
#     for reverse_sel_ in [True, False]:
#         draw(reverse_sel_, "test", **run_kwargs)

if __name__ == "__main__":
    # run_kwargs_ = {"region": "SC", "model": "CNN10"}
    # # reverse_sel_ = False
    # multirun_ = 10
    # for reverse_sel_ in [True, False]:
    #     main(reverse_sel_, multirun_, "test", **run_kwargs_)
    fire.Fire(draw)
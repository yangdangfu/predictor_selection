# -*- coding: utf-8 -*-
""" 
Usages:
    python main_selct.py --reverse_sel False --multirun 10 --region SC --model CNN10
"""
import subprocess
from typing import Literal
import fire

from omegaconf import OmegaConf
from selector.predictor_selector import PredictorSelector
from utils.get_rundirs import get_rundirs
import logging, coloredlogs
import numpy as np
import os
import xarray as xr
from utils.cli_utils import arguments_check

logger = logging.Logger(__name__)


def get_weights(rundirs, mode: Literal["val", "test"] = "val"):
    """Run run the predictor of given runs """

    # compute weights
    weights_all = list()
    for rdir in rundirs:
        wgts_fpath = os.path.join(rdir, "prediction", f"weights_{mode}.nc")
        # if not os.path.exists(wgts_fpath):
        #     continue
        weights = xr.open_dataarray(wgts_fpath)
        weights_all.append(weights)
    weights_avg = sum(weights_all) / len(
        weights_all)  # xr.DataArray, dim: time x predictors

    weights = weights_avg.sum("time").values  # np.ndarray, dim: predictors
    weights /= weights.sum()
    predictors = list(weights_avg.variables.values)
    return weights.tolist(), predictors


def run_select(reverse_sel: bool, multirun: int, **run_kwargs):
    REQUIRED_KEYS = ["region", "model"]
    arguments_check(run_kwargs.keys(), REQUIRED_KEYS)

    cfg = OmegaConf.load("CONFIGS/config.yaml")
    predictors_all = OmegaConf.to_container(cfg.candidate_predictors)
    num_predictors = len(predictors_all)
    selector = PredictorSelector(predictors_all)
    run_kwargs.update(bistr=selector.get_binary_strings())  # add a bistr arg

    for i in range(num_predictors):

        root_dir = f"outputs_{run_kwargs['region']}"
        rundirs = get_rundirs(root_dir, run_kwargs)

        if len(rundirs) <= multirun:
            cmd_sufix = " "
            for key, val in run_kwargs.items():
                cmd_sufix += f"--{key} {val} "
            # run
            run_cmd_cnn = f"python main_multirun.py --num {multirun}" + cmd_sufix  # --weight_decay {run_kwargs['weight_decay']}
            logger.info(f"Execute run command {run_cmd_cnn} ......")
            subprocess.run(run_cmd_cnn.split(" "))
            logger.info(f"Execute run command {run_cmd_cnn} donw")

        # for mmodel_name in ["LM", "RIDGE", "SVM"][0:1]:
        #     run_cmd_ml = f"python run.py ml {run_kwargs['region']} {mmodel_name} {run_kwargs['bistr']}"
        #     logger.info(f"Execute run command {run_cmd_ml} ......")
        #     subprocess.run(run_cmd_ml.split(" "))
        #     logger.info(f"Execute run command {run_cmd_ml} donw")

        assert len(rundirs) >= multirun, "Not enough runs!"

        print(rundirs)

        logger.info(f"Run predictor selection ......")
        # Compute the weights of each predictors
        weights, predictors = get_weights(rundirs)
        predictor_removed = selector.run_select(weights, predictors,
                                                reverse_sel)

        run_kwargs.update(bistr=selector.get_binary_strings())

        logger.info(
            f"Removed predictor: {predictor_removed}, Binary strings: {selector.get_binary_strings()}"
        )
        if i==2:
            return 


if __name__ == "__main__":
    # reverse_select_ = False
    # multirun_ = 2
    # run_kwargs_ = dict(region="SC", model="CNN10")
    # run_select(reverse_select_, multirun_, **run_kwargs_)
    coloredlogs.install(level="INFO", logger=logger)
    fire.Fire(run_select)

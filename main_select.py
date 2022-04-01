# -*- coding: utf-8 -*-
""" 
Usages:
- for predictor selection
    python main_select.py --reverse_sel False --multirun 10 --re_pred False --re_weight False --re_score False --region SC --model CNN10
- for reverse predictor selection
    python main_select.py --reverse_sel True --multirun 10 --re_pred False --re_weight False --re_score False --region SC --model CNN10
"""
import subprocess
from typing import Literal
import fire

from omegaconf import OmegaConf
from selector.predictor_selector import PredictorSelector
from utils.get_rundirs import get_rundirs
import logging, coloredlogs
import os
import xarray as xr
from utils.cli_utils import arguments_check

logger = logging.getLogger(os.path.basename(__file__))


def get_weights(rundirs: list, dset: Literal["train", "val", "test"] = "val"):
    """ get average weights in given run directories """

    # compute weights
    weights_all = list()
    for rdir in rundirs:
        wgts_fpath = os.path.join(rdir, "prediction", f"weights_{dset}.nc")
        weights = xr.open_dataarray(wgts_fpath)
        weights_all.append(weights)
    weights_avg = sum(weights_all) / len(
        weights_all)  # xr.DataArray, dim: time x predictors

    weights = weights_avg.sum("time").values  # np.ndarray, dim: predictors
    weights /= weights.sum()
    predictors = list(weights_avg.variables.values)
    return weights.tolist(), predictors


def run_select(reverse_sel: bool,
               multirun: int,
               re_pred: bool,
               re_weight: bool,
               re_score: bool,
               skip_runs=False,
               **run_kwargs):
    """ 
    Note: set `skip_runs` to `True` if you wanna to skip both CNN and LR runs. The situation is mainly used for post-procedures, like plot, etc 
    """
    REQUIRED_KEYS = ["region", "model"]
    arguments_check(run_kwargs.keys(), REQUIRED_KEYS)

    cfg = OmegaConf.load("CONFIGS/config.yaml")
    predictors_all = OmegaConf.to_container(cfg.candidate_predictors)
    num_predictors = len(predictors_all)
    selector = PredictorSelector(predictors_all)
    run_kwargs.update(bistr=selector.get_binary_strings())  # add a bistr arg

    bi_strings = list()
    for _ in range(num_predictors):
        bi_strings.append(run_kwargs["bistr"])
        # if len(rundirs) < multirun:
        cmd_sufix = ""
        for key, val in run_kwargs.items():
            cmd_sufix += f" --{key} {val}"
        if not skip_runs:
            # run cnn
            run_cmd_cnn = f"python main_multirun.py --num {multirun} --re_pred {re_pred} --re_weight {re_weight} --re_score {re_score}" + cmd_sufix  # --weight_decay {run_kwargs['weight_decay']}
            logger.info(f"Execute run command {run_cmd_cnn} ......")
            subprocess.run(run_cmd_cnn.split(" "))
            logger.info(f"Execute run command {run_cmd_cnn} donw")

        root_dir = f"outputs_{run_kwargs['region']}"
        rundirs = get_rundirs(root_dir, run_kwargs)

        assert len(rundirs) >= multirun, "Not enough runs!"

        logger.info(f"Run predictor selection ......")
        # Compute the weights of each predictors
        weights, predictors = get_weights(rundirs)
        predictor_removed = selector.run_select(weights, predictors,
                                                reverse_sel)

        run_kwargs.update(bistr=selector.get_binary_strings())

        logger.info(
            f"Removed predictor: {predictor_removed}, Binary strings: {selector.get_binary_strings()}"
        )

    if not skip_runs:
        # run LM
        for bistr in bi_strings:
            run_cmd_ml = f"python main_ml.py --re_pred {re_pred} --re_score {re_score} --region {run_kwargs['region']} --bistr {bistr}"
            logger.info(f"Execute run command {run_cmd_ml} ......")
            subprocess.run(run_cmd_ml.split(" "))
            logger.info(f"Execute run command {run_cmd_ml} donw")
    return bi_strings


if __name__ == "__main__":
    coloredlogs.install(level="INFO", logger=logger)
    # reverse_select_ = False
    # multirun_ = 2
    # run_kwargs_ = dict(region="SC", model="CNN10")
    # res = run_select(reverse_select_, multirun_, False, False, False,
    #                  **run_kwargs_)
    # print(res)
    # res = run_select(reverse_select_, multirun_, False, False, False, True,
    #                  **run_kwargs_)
    # print(res)
    fire.Fire(run_select)

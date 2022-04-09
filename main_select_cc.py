# -*- coding: utf-8 -*-
""" 
Usages:
- for predictor selection based on correlation analysis
    python main_select_cc.py --reverse_sel False --multirun 10 --re_pred False --re_weight False --re_score False --region SC --model CNN10
"""
import subprocess
import pandas as pd
import fire
import numpy as np

from omegaconf import OmegaConf
from selector.predictor_selector import PredictorSelector
from utils.get_rundirs import get_rundirs
import logging, coloredlogs
import os
import xarray as xr
from utils.cli_utils import arguments_check
from utils.preprocessing import StandardisationByMonth
import xskillscore as xs

logger = logging.getLogger(os.path.basename(__file__))


def get_cc_sequence(region: str) -> pd.DataFrame:
    cfg = OmegaConf.load("CONFIGS/config.yaml")
    in_noaug_filepath = cfg.in_noaug_filepath.format(region=region)
    out_noaug_filepath = cfg.out_noaug_filepath.format(region=region)

    # load data during 1981 to 2010
    predictors_sub = OmegaConf.to_container(cfg.candidate_predictors)
    x_noaug = xr.open_dataset(in_noaug_filepath)[predictors_sub].sel(
        time=slice("1981-01-01", "2010-12-31"))
    y_noaug = xr.open_dataarray(out_noaug_filepath).sel(
        time=slice("1981-01-01", "2010-12-31"))
    # aggregate y over grids and perform preprocessing
    y_noaug = y_noaug.sum(["lat", "lon"])
    x_preprocessor = StandardisationByMonth()
    x_preprocessor.fit(x_noaug)
    x_noaug = x_preprocessor.process(x_noaug)
    y_preprocessor = StandardisationByMonth()
    y_preprocessor.fit(y_noaug)
    y_noaug = y_preprocessor.process(y_noaug)
    # compute correlation coefficients
    cc = xs.pearson_r(x_noaug, y_noaug, dim="time")

    cc = np.fabs(cc).mean(("lat", "lon"))

    return cc.to_array().to_dataframe(name="cc")


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
    cc_df = get_cc_sequence(region=run_kwargs["region"])

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
        cc_df = cc_df.loc[selector.predictors_left]
        weights, predictors = list(cc_df.values), list(cc_df.index)
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
    fire.Fire(run_select)

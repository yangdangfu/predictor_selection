# -*- coding: utf-8 -*-
"""
Aggregate scores in multiple-run directories or single-run directory
"""
from typing import Union
import os
import pickle
import copy


def agg_score_spatial(rdir: Union[str, list], dset: str):
    if isinstance(rdir, str):
        with open(
                os.path.join(rdir, "prediction", f"score_spatial_{dset}.pkl"),
                "rb") as f:
            scores_spatial = pickle.load(f)
    else:
        assert isinstance(rdir, list), f"Unsupported type {type(rdir)}"
        scores_spatial_all = list()
        for d in rdir:
            with open(
                    os.path.join(d, "prediction", f"score_spatial_{dset}.pkl"),
                    "rb") as f:
                scores_spatial_single = pickle.load(f)
            scores_spatial_all.append(scores_spatial_single)
        num = len(rdir)
        scores_spatial = copy.deepcopy(scores_spatial_all[0])
        for key in scores_spatial:
            for i in range(1, num):
                scores_spatial[
                    key] = scores_spatial[key] + scores_spatial_all[i][key]
            scores_spatial[key] = scores_spatial[key] / num

    return scores_spatial


def agg_score_temporal(rdir: Union[list, str], dset: str):
    if isinstance(rdir, str):
        with open(
                os.path.join(rdir, "prediction", f"score_temporal_{dset}.pkl"),
                "rb") as f:
            scores_temporal = pickle.load(f)
    else:
        assert isinstance(rdir, list), f"Unsupported type {type(rdir)}"
        scores_temporal_all = list()
        for d in rdir:
            with open(
                    os.path.join(d, "prediction",
                                 f"score_temporal_{dset}.pkl"), "rb") as f:
                scores_temporal_single = pickle.load(f)
            scores_temporal_all.append(scores_temporal_single)
        num = len(rdir)
        scores_temporal = copy.deepcopy(scores_temporal_all[0])
        for key in scores_temporal:
            for i in range(1, num):
                scores_temporal[
                    key] = scores_temporal[key] + scores_temporal_all[i][key]
            scores_temporal[key] = scores_temporal[key] / num

    return scores_temporal
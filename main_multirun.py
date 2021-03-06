# -*- coding: utf-8 -*-
""" 
Usages: 
    python main_multirun.py --num 2 --re_pred False --re_weight False --re_score False --bistr 00000100010101010101 --region SC
"""
from utils.get_rundirs import get_rundirs
import logging, coloredlogs
import os
import subprocess
import fire

logger = logging.getLogger(os.path.basename(__file__))


def multirun(num: int, re_pred: bool, re_weight: bool, re_score: bool,
             **run_kwargs):
    cmd_suffix = ""
    for key in run_kwargs:
        cmd_suffix += f" --{key} {run_kwargs[key]}"

    # --------------------------------- training --------------------------------- #
    runs = get_rundirs(f"outputs_{run_kwargs['region']}", run_kwargs)
    if len(runs) == num:
        logger.warning(
            f"The number of runs has reached the required {num}, skip the train"
        )
    elif len(runs) > num:
        logger.error(
            f"The number of runs has exceeded the required {num}, skip the train"
        )
    else:
        cmd_train = "python main_train.py" + cmd_suffix
        for _ in range(num - len(runs)):
            logger.info(f"Executing train command {cmd_train}")
            subprocess.run(cmd_train.split(" "))
            logger.info(f"Executing train command done!")

    # ------------------------- Prediction and evaluation ------------------------ #
    cmd_eval = f"python main_eval.py --re_pred {re_pred} --re_weight {re_weight} --re_score {re_score}" + cmd_suffix
    logger.info(f"Executing command {cmd_eval}")
    subprocess.run(cmd_eval.split(" "))
    logger.info(f"Executing command done!")


if __name__ == "__main__":
    coloredlogs.install(level="INFO", logger=logger)
    fire.Fire(multirun)
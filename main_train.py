# -*- coding: utf-8 -*-
"""
Usages
    python main_train.py --region=SC --bistr=11111111111111111111
TODO: There are bugs when use multirun feature in hydra
    python main_train.py --region=SC --bistr=[11111111111111111111,11111111111111111111,11111111111111111111] 
    python main_train.py --region=[SC,Yangtze] --bistr=[11111111111111111111,11110001111111111111,11111111100001111111] 
"""
import os
import logging
import fire
import subprocess

logger = logging.getLogger(os.path.basename(__file__))


def main(**kwargs):
    # CLI command
    cmd_train = f"python trainer.py"
    # if there is a list of CLI parameters, hydra multirun feature will be activated
    has_list = False
    for key in kwargs:
        if isinstance(kwargs[key], list):
            has_list = True
            cmd_train += f" {key}={','.join(kwargs[key])}"
        else:
            cmd_train += f" {key}={kwargs[key]}"
    if has_list:
        cmd_train += " --multirun"
    # execute
    subprocess.run(cmd_train.split(" "))


if __name__ == '__main__':
    fire.Fire(main)

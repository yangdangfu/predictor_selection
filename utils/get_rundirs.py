# -*- coding: utf-8 -*-
import os, re


def getdirs(root: str, regex: str = None):
    """ Get all subdirs in the given root. `regex` is the regular expression of subdirectory, if provided, only the subdirectories matching with `regex` will be returned """
    if not os.path.exists(root):
        return []
    run_dirs = os.listdir(root)
    run_dirs.sort()
    if regex is not None:
        run_dirs = [
            os.path.join(root, path) for path in run_dirs
            if (re.match(regex, path) is not None)
        ]
    else:
        run_dirs = [os.path.join(root, path) for path in run_dirs]
    run_dirs = [path for path in run_dirs if os.path.isdir(path)]
    return run_dirs


def get_rundirs(root_dir, req_kwargs_dict: dict, opt_kwargs_dict: dict = None):
    """ A general way to get the directories of model constructed under given arguments (required or optional) """
    keys = list(req_kwargs_dict.keys())
    if opt_kwargs_dict is not None:
        keys = keys + list(opt_kwargs_dict.keys())
    keys.sort()
    regex = "^\d{8}-\d{6}"
    for key in keys:
        if key in req_kwargs_dict.keys():
            regex += f"{key}={req_kwargs_dict[key]},"
        else:  # key is in `opt_kwargs_dict``
            regex += f"({key}={opt_kwargs_dict[key]},)"
    regex = regex[:-1] + "$"  # drop the last comma and append a end symbol to the regular expression
    run_dirs = getdirs(root_dir, regex)
    return run_dirs

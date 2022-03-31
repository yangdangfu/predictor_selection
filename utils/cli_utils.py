# -*- coding: utf-8 -*-
""" 
Some utility functions about the CLI (Command Line Interface) usages 
"""


def arguments_check(keys: list, required_keys: list):
    for k in keys:
        if k not in required_keys:
            raise KeyError(f"Unsupported argument {k}")
    for sk in required_keys:
        if sk not in keys:
            raise KeyError(f"Missing argument {sk}")
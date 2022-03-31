# -*- coding: utf-8 -*-
""" Implement several dataset split functions """
from typing import List, Tuple
import xarray as xr
import itertools
import math


def split_by_year(ref: xr.Dataset or xr.DataArray, train_ratio: float,
                  val_ratio: float) -> Tuple[List[int]]:
    """
    Split the dataset into train & val & test subsets in terms of years. The input `ref` is a reference argument that has a `time` dimension. The data in the `ref` won't be changed within the function .
    """
    assert train_ratio + val_ratio < 1.0, "Sum of train_ration and val_ratio should be less than 1.0."
    if 'year' in ref.dims:
        years = ref["year"].values.tolist()
        assert len(years) == len(
            set(years)), "Duplicated 'year's are not allowed"
        year_groups = dict(zip(years, [[x] for x in range(len(years))]))
    else:
        year_groups = ref.time.groupby("time.year").groups
    num_years = len(year_groups)
    left = round(num_years * train_ratio)
    right = round(num_years * (train_ratio + val_ratio))
    years = list(year_groups.keys())
    train_index = list(itertools.chain(*[year_groups[i]
                                         for i in years[:left]]))
    val_index = list(
        itertools.chain(*[year_groups[i] for i in years[left:right]]))
    test_index = list(itertools.chain(*[year_groups[i]
                                        for i in years[right:]]))
    return train_index, val_index, test_index


def kfold_split(num: int, k: int) -> list:
    """A k-fold split that will return train, validation and test dataset index. In the k folds, a continuous 2 folds will be used as validation and test datasets, respectively, the rest k-2 folds is used as train dataset. 

    Args:
        num (int): Total number of datasets
        k (int): number of folds of the datasets to be split

    Returns:
        list: return the index split list. Each element in the list is a 3-tuple of train_index, val_index, test_index
    """
    splits = [round(num / k * i) for i in range(k)]
    splits.append(None)
    index = list(range(num))

    index_splits = list()
    for i in range(k - 1):
        val_index = index[splits[i]:splits[i + 1]]
        test_index = index[splits[i + 1]:splits[i + 2]]
        train_index = list(set(index) - set(val_index) - set(test_index))
        index_splits.append((train_index, val_index, test_index))

    return index_splits[::-1]


def slide_split(num: int, num_val: int, num_test: int) -> list:
    """ A slide dataset split scheme. The slide reference is the test set, so that the test set among all reslut splits cover the whole dataset (execept for the first a few samples) 

    Args:
        num (int): Total number of samples in the dataset
        num_val (int): Number of samples for validation dataset
        num_test (int): Number of samples for test dataset

    Returns:
        list: The list of dataset split folds. Each fold (an entry in the list) is a 3-tuple whose elements are train_index, val_index, test_index in order
    """
    assert num_val + num_test < num, "Sum of num_val and num_test should be less than num."
    index_all = list(range(num))
    index_splits = list()
    num_folds = math.floor((num - num_val) / num_test)

    for fold_idx in range(num_folds):
        #  ... left ... val_index ... center ... test_index ... right
        left = num - (fold_idx + 1) * num_test - num_val
        center = num - (fold_idx + 1) * num_test
        right = num - fold_idx * num_test
        val_index = index_all[left:center]
        test_index = index_all[center:right]
        train_index = list(set(index_all) - set(val_index) - set(test_index))
        index_splits.append((train_index, val_index, test_index))
        print(index_splits[-1])
    return index_splits


def slide_ratio_split(num: int, val_ratio: float, test_ratio: float) -> list:
    """This is  a dataset split scheme very similar to the `slide_split.`

    Args:
        num (int): Total number of samples in the dataset
        val_ratio (float): Ratio of sample number for validation sample
        test_ratio (float): Ratio of sample number for validation sample

    Returns:
        list: The list of dataset split folds. Each fold (an entry in the list) is a 3-tuple whose elements are train_index, val_index, test_index in order
    """
    assert val_ratio + test_ratio < 1.0, "Sum of val_ratio and test_ratio should be less than 1.0."
    num_val = round(num * val_ratio)
    num_test = round(num * test_ratio)
    return slide_split(num, num_val, num_test)


if __name__ == "__main__":
    if False:
        _num = 40
        _k = 3
        splits = kfold_split(_num, _k)
        for splt in splits:
            train_index, val_index, test_index = splt
            print(val_index, test_index)
        # print(len(train_index), len(val_index), len(test_index))
    if True:
        _num = 40
        splits = slide_split(_num, 10, 5)
        # for splt in splits:
        #     train_index, val_index, test_index = splt
        #     print(val_index, test_index)
        # print(len(train_index), len(val_index), len(test_index))

    if True:
        _num = 40
        splits = slide_ratio_split(_num, 0.25, 0.125)
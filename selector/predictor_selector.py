# -*- coding: utf-8 -*-
import copy
from typing import List


class PredictorSelector:
    def __init__(
        self,
        predictors_all: list = [
            'air1000', 'air500', 'air700', 'air850', 'hgt1000', 'hgt500',
            'hgt700', 'hgt850', 'shum1000', 'shum500', 'shum700', 'shum850',
            'uwnd1000', 'uwnd500', 'uwnd700', 'uwnd850', 'vwnd1000', 'vwnd500',
            'vwnd700', 'vwnd850'
        ]
    ) -> None:
        self._PREDICTORS_ALL = copy.deepcopy(predictors_all)
        self.predictors_left = copy.deepcopy(predictors_all)
        self.predictors_fix = list()  # predictors that shouldn't be removed
        self.predictors_removed = list()

    def run_select(self,
                   weights: List[float],
                   predictors: List[str],
                   reverse: bool = False):
        """ Run select accoording to the given `weights` of `prdictors`
        """
        # Assert the number of entries in weights equals to the number of left predictors
        assert len(weights) == len(
            self.predictors_left), "Wrong weights length."
        assert len(weights) == len(predictors), "Wrong predictors length."
        # Find the predictor to remove
        idx_rmv = -1
        wgt_thd = None
        for i, (w, p) in enumerate(zip(weights, predictors)):
            if p in self.predictors_fix:
                continue
            if wgt_thd is None:
                idx_rmv, wgt_thd = i, w
            else:
                if not reverse:
                    idx_rmv, wgt_thd = (i, w) if w < wgt_thd else (idx_rmv,
                                                                   wgt_thd)
                else:
                    idx_rmv, wgt_thd = (i, w) if w > wgt_thd else (idx_rmv,
                                                                   wgt_thd)
        if idx_rmv < 0:
            return False
        # run selection
        predictors_left = [
            p for p in self.predictors_left if p != predictors[idx_rmv]
        ]
        self.predictors_left = predictors_left
        self.predictors_removed.append(predictors[idx_rmv])
        return True

    def select_invert(self):
        self.predictors_left.append(self.predictors_removed.pop())
        self.predictors_fix.append(self.predictors_left[-1])

    def add_fixed_predictor(self, name: str):
        """ Add a fixed predictor  """
        # Assert the predictor is within the lefted predictors
        assert name in self.predictors_left, "The added predictor is not within the left ones"
        self.predictors_fix.append(name)

    def get_binary_strings(self, predictors_sub: list = None):
        """ Generate binary strings based on the provided subset predictors """
        if predictors_sub is None:
            predictors_sub = self._PREDICTORS_ALL
        # Assert that all the input subset predictors is within the candidate predictors
        cover = [
            True if var in self._PREDICTORS_ALL else False
            for var in predictors_sub
        ]
        assert all(
            cover
        ), "Not all subset of predictors are with in the candicate predictors."
        # Generate the binary string
        strings = [
            "0" if var not in predictors_sub else "1"
            for var in self._PREDICTORS_ALL
        ]
        return "".join(strings)

    def get_predictors_sub(self, binary_strings: str):
        """ Return the subset predictors according to the input binary strings """
        # Assert the binary strings have a equal length to the candicate predictors
        if not isinstance(binary_strings, str):
            binary_strings = str(binary_strings)
        assert len(binary_strings) == len(
            self._PREDICTORS_ALL
        ), "The length of bistrings should be equal to the total number of candidate predictors. "
        # Collect the subset predictors based on the input binary strings
        predictors_sub = list()
        for i, s in enumerate(binary_strings):
            if s == "1":
                predictors_sub.append(self._PREDICTORS_ALL[i])
            else:
                assert s == "0", "The characters in the bistrings should be either 0 or 1."
        return predictors_sub


if __name__ == "__main__":
    selector = PredictorSelector()
    print(selector.get_binary_strings())

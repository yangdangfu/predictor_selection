# -*- coding: utf-8 -*-
""" A module for data pre-processing """
from typing import List, Any
import xarray as xr
import warnings
from abc import ABC, abstractmethod
import numpy as np


#ANCHOR Processor (Base)
class Processor(ABC):
    @abstractmethod
    def reducing(self, ) -> xr.Dataset or tuple:
        pass

    @abstractmethod
    def _fitted(self) -> bool:
        pass

    @abstractmethod
    def fit(self, data: xr.Dataset) -> None:
        pass

    @abstractmethod
    def process(self, data: xr.Dataset) -> xr.Dataset:
        pass

    @abstractmethod
    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        pass

    def _assert_fitted(self, need: bool) -> None:
        if not need:
            if self._fitted():
                warnings.warn(
                    f"{__class__} instance is aready fitted and will be overrided."
                )
        else:
            assert self._fitted(), f"{__class__} instance is not fitted."


# ANCHOR TimeIncrement
class TimeIncrement(Processor):
    """ Process data to its time increment """
    def __init__(self, base: Any = None) -> None:
        super().__init__()
        self._base = base

    def reducing(self, ) -> xr.Dataset:
        self._assert_fitted(True)
        return self._base

    def _fitted(self) -> bool:
        return (self._base is not None)

    def fit(self, data: xr.Dataset) -> None:
        self._assert_fitted(need=False)
        self._base = data.shift(time=1).drop_isel(time=0)

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data.diff(dim="time")
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = self._base.sel(time=data.time) + data
        return new_data


def minmax_scale(x, minval, maxval):
    mask = x > 0
    np.putmask(x, mask, x / maxval)
    np.putmask(x, ~mask, x / (-minval))
    return x


def i_minmax_scale(x, minval, maxval):
    """ Inverse minmax scale"""
    mask = x > 0
    np.putmask(x, mask, x * maxval)
    np.putmask(x, ~mask, x * (-minval))
    return x


# ANCHOR SignedScale
class SignedScale(Processor):
    """ Scale each feature by its maximum value (for positive) and minimum value (for negative) """
    def __init__(self, abs_max: Any = None, abs_min: Any = None) -> None:
        super().__init__()
        self._max = abs_max
        self._min = abs_min

    def reducing(self, ) -> xr.DataArray:
        self._assert_fitted(True)
        return self._max, self._min

    def _fitted(self) -> bool:
        return (self._max is not None and self._min is not None)

    def fit(self, data: xr.DataArray, dim: str = "time") -> None:
        self._assert_fitted(need=False)
        self._max = data.max(dim=dim).values
        self._min = data.min(dim=dim).values

    def process(self, data: xr.DataArray) -> xr.DataArray:
        self._assert_fitted(need=True)
        new_data = data + 0.0
        xr.apply_ufunc(minmax_scale, new_data, self._min, self._max)
        new_data = new_data * 0.5 + 0.5
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = (data - 0.5) * 2
        xr.apply_ufunc(i_minmax_scale, new_data, self._min, self._max)
        return new_data


# ANCHOR MaxAbsScale
class MaxAbsScale(Processor):
    """ Scale each feature by its maximum absolute value """
    def __init__(self, abs_max: Any = None) -> None:
        super().__init__()
        self._max = abs_max

    def reducing(self, ) -> xr.Dataset:
        self._assert_fitted(True)
        return self._max

    def _fitted(self) -> bool:
        return (self._max is not None)

    def fit(self, data: xr.Dataset, dim: str = "time") -> None:
        self._assert_fitted(need=False)
        self._max = abs(data).max(dim=dim)

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = (data / self._max) * 0.5 + 0.5
        new_data = new_data.clip(min=0, max=1)  # REVIEW: NEED OR NOT ?
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = (data - 0.5) * 2 * self._max
        return new_data


class Fillna(Processor):
    """Fill nan with 0
    TODO: More filled method; fit implementation
    """
    def __init__(self, ):
        self.na_mask = None

    def reducing(self, ) -> xr.Dataset:
        self._assert_fitted(True)
        return self.na_mask

    def _fitted(self) -> bool:
        return self.na_mask is not None

    def fit(self, data: xr.Dataset) -> None:
        self._assert_fitted(need=False)
        self.na_mask = 1  # (data.isnull().sum(dim="time")) > 0

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data.fillna(0.0)
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        raise NotImplementedError()
        new_data = data * 1
        return new_data


# ANCHOR Identity
class Identity(Processor):
    def __init__(self, ):
        self.is_fitted = False

    def reducing(self, ) -> xr.Dataset:
        self._assert_fitted(True)
        return None

    def _fitted(self) -> bool:
        return self.is_fitted

    def fit(self, data: xr.Dataset) -> None:
        self._assert_fitted(need=False)
        self.is_fitted = True

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data * 1
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data * 1
        return new_data


# ANCHOR Deviation
class Deviation(Processor):
    def __init__(self, mean: xr.Dataset = None):
        self._mean = mean

    def reducing(self, ) -> xr.Dataset:
        self._assert_fitted(True)
        return self._mean

    def _fitted(self) -> bool:
        return (self._mean is not None)

    def fit(self, data: xr.Dataset, dim: str = "time") -> None:
        self._assert_fitted(need=False)
        self._mean = data.mean(dim=dim)

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data - self._mean
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data + self._mean
        return new_data


# ANCHOR Standardisation
class Standardisation(Processor):
    def __init__(self,
                 mean: xr.Dataset = None,
                 std: xr.Dataset = None) -> None:
        self._mean, self._std = mean, std

    def reducing(self, ) -> tuple:
        self._assert_fitted(True)
        return self._mean, self._std

    def _fitted(self) -> bool:
        return ((self._mean is not None) and (self._std is not None))

    def fit(self, data: xr.Dataset, dim: str = "time") -> None:
        self._assert_fitted(need=False)
        self._mean = data.mean(dim=dim)
        self._std = data.std(dim=dim)

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = (data - self._mean) / self._std
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data * self._std + self._mean
        return new_data


# ANCHOR DeviationByMonth
class DeviationByMonth(Processor):
    def __init__(self, mean: xr.Dataset = None) -> None:
        self._mean = mean

    def reducing(self, ) -> xr.Dataset:
        self._assert_fitted(True)
        return self._mean

    def _fitted(self) -> bool:
        return self._mean is not None

    def fit(self, data: xr.Dataset) -> None:
        self._assert_fitted(need=False)
        grp = data.groupby("time.month")
        self._mean = grp.mean("time")

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        grp = data.groupby("time.month")
        new_data = grp - self._mean
        return new_data.drop("month")

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        grp = data.groupby("time.month")
        new_data = grp + self._mean
        return new_data.drop("month")


# ANCHOR StandardisationByMonth
class StandardisationByMonth(Processor):
    def __init__(self,
                 mean: xr.Dataset = None,
                 std: xr.Dataset = None) -> None:
        self._mean, self._std = mean, std

    def reducing(self, ) -> tuple:
        self._assert_fitted(True)
        return self._mean, self._std

    def _fitted(self) -> bool:
        return ((self._mean is not None) and (self._std is not None))

    def fit(self, data: xr.Dataset) -> None:
        self._assert_fitted(need=False)
        grp = data.groupby("time.month")
        self._mean = grp.mean("time")
        self._std = grp.std("time")

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        grp = data.groupby("time.month")
        new_data = (grp - self._mean).groupby("time.month") / self._std
        return new_data.drop("month")

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        grp = data.groupby("time.month")
        new_data = (grp * self._std).groupby("time.month") + self._mean
        return new_data.drop("month")


# ANCHOR pipeline processor
class SequentialProcessor(Processor):
    def __init__(self, processor_cls_names: List[str]) -> None:
        self._processors: List[Processor] = [
            globals()[name]() for name in processor_cls_names
        ]

    def reducing(self, ) -> list:
        return [processor.reducing() for processor in self._processors]

    def _fitted(self) -> bool:
        return all(p._fitted() for p in self._processors)

    def fit(self, data: xr.Dataset, dim: str or tuple = "time") -> None:
        self._assert_fitted(need=False)
        new_data = data
        for p in self._processors:
            p.fit(new_data, dim=dim)
            if len(self._processors) > 1:
                new_data = p.process(new_data)

    def process(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data
        for p in self._processors:
            new_data = p.process(new_data)
        return new_data

    def reverse(self, data: xr.Dataset) -> xr.Dataset:
        self._assert_fitted(need=True)
        new_data = data
        for p in reversed(self._processors):
            new_data = p.reverse(new_data)
        return new_data


# PROCESSORS = dict(signed_scale=SignedScale,
#                   max_abs_scale=MaxAbsScale,
#                   fillna=Fillna,
#                   identity=Identity,
#                   deviation=Deviation,
#                   standardisation=Standardisation,
#                   deviation_by_month=DeviationByMonth,
#                   standardisation_by_month=StandardisationByMonth)

# ANCHOR Main
if __name__ == "__main__":
    import sys
    sys.path.append("../")

    from data_utils.ncep_data_utils import read_monthly_cpc, read_monthly_ncep

    from datetime import date
    import numpy as np

    xmin, xmax, ymin, ymax = 72, 137, 15, 55

    if False:
        factor = "precip"
        cpc = read_monthly_cpc(factor,
                               date(2018, 1, 1),
                               date(2020, 12, 31),
                               lat_range=(ymin, ymax),
                               lon_range=(xmin, xmax))["precip"]
        cpc = cpc - cpc.mean(dim="time") + 0.1
        cpc = cpc.fillna(0.0)
        mm_scale = SignedScale()
        mm_scale.fit(cpc)
        cpc_prcsd = mm_scale.process(cpc)
        cpc_reversed = mm_scale.reverse(cpc_prcsd)
        print(cpc.min().values, cpc.max().values)
        print(cpc_prcsd.min().values, cpc_prcsd.max().values)
        print(cpc_reversed.min().values, cpc_reversed.max().values)

    if False:
        factor = "precip"
        cpc = read_monthly_cpc(factor,
                               date(2015, 1, 1),
                               date(2020, 12, 31),
                               lat_range=(ymin, ymax),
                               lon_range=(xmin, xmax))

        for ProcessorClass in [
                Deviation, DeviationByMonth, Standardisation,
                StandardisationByMonth
        ]:
            preprocess = ProcessorClass()
            preprocess.fit(cpc)
            cpc_preprocessed = preprocess.process(cpc)
            cpc_reversed = preprocess.reverse(cpc_preprocessed)
            print(ProcessorClass.__name__)
            print("CPC", "---" * 10)
            print(cpc)
            print("Processed", "---" * 10)
            print(cpc_preprocessed)
            print("Reversed", "---" * 10)
            print(cpc_reversed)
            print("Reducing", "---" * 10)
            print(preprocess.reducing())
            print("Shapes", "---" * 10)
            print(cpc[factor].shape, cpc_preprocessed[factor].shape,
                  cpc_reversed[factor].shape)
    if False:
        factors = {"slp": None, "air": [500, 850]}
        ncep = read_monthly_ncep(factors,
                                 date(2015, 1, 1),
                                 date(2020, 12, 31),
                                 lat_range=(ymin, ymax),
                                 lon_range=(xmin, xmax))

        for ProcessorClass in [
                Deviation, DeviationByMonth, Standardisation,
                StandardisationByMonth
        ]:
            preprocess = ProcessorClass()
            preprocess.fit(ncep)
            ncep_preprocessed = preprocess.process(ncep)
            ncep_reversed = preprocess.reverse(ncep_preprocessed)
            print(ProcessorClass.__name__)
            print("NCEP", "---" * 10)
            print(ncep)
            print("Processed", "---" * 10)
            print(ncep_preprocessed)
            print("Reversed", "---" * 10)
            print(ncep_reversed)
            print("Reducing", "---" * 10)
            print(preprocess.reducing())
            print("Shapes", "---" * 10)
            print(ncep.to_array().shape,
                  ncep_preprocessed.to_array().shape,
                  ncep_reversed.to_array().shape)

    if True:
        factor = "precip"
        cpc = read_monthly_cpc(factor,
                               date(2015, 1, 1),
                               date(2020, 12, 31),
                               lat_range=(ymin, ymax),
                               lon_range=(xmin, xmax))

        processor_cls_names = [
            "Deviation",
            # DeviationByMonth(),
            "Standardisation",
            # StandardisationByMonth()
        ]
        preprocess = SequentialProcessor(processor_cls_names)
        preprocess.fit(cpc, dim=("time", "lat", "lon"))
        cpc_preprocessed = preprocess.process(cpc)
        cpc_reversed = preprocess.reverse(cpc_preprocessed)
        print(SequentialProcessor.__name__)
        print("CPC", "---" * 10)
        print(cpc)
        print("Processed", "---" * 10)
        print(cpc_preprocessed)
        print("Reversed", "---" * 10)
        print(cpc_reversed)
        print("Reducing", "---" * 10)
        print(preprocess.reducing())
        print("Shapes", "---" * 10)
        print(cpc[factor].shape, cpc_preprocessed[factor].shape,
              cpc_reversed[factor].shape)

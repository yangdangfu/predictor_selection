# -*- coding: utf-8 -*-
"""General CNN architectures from the following two papers: 

[1] Baño-Medina, Jorge, Rodrigo Manzanas, and José Manuel Gutiérrez. “Configuration and Intercomparison of Deep Learning Neural Models for Statistical Downscaling.” Geoscientific Model Development 13, no. 4 (April 28, 2020): 2109–24. https://doi.org/10.5194/gmd-13-2109-2020.

[2] Sun, Lei, and Yufeng Lan. “Statistical Downscaling of Daily Temperature and Precipitation over China Using Deep Learning Neural Models: Localization and Comparison with Other Methods.” International Journal of Climatology 41, no. 2 (2021): 1128–47. https://doi.org/10.1002/joc.6769.

They are: 

- CNN_LM
- CNN1
- CNN10
- CNN_PR 
- CNNdense
- CNN_FC

Note: CNN_FC is only used in [2], the other 5 archs are used both in [1] and [2]

We provide a convenient function `instantiate_model` to instantiate the CNNs by providing names and args, for example: 
```python
cnn = instantiate_model(model_cls_name=..., in_channels=..., in_height=..., in_width=..., out_features=..., padding=...)
```
where the `model_cls_name` can be one of CNN_LM, CNN1, CNN10, CNN_PR, CNNdense, CNN_FC
"""
import torch.nn as nn


class GeneralCNN(nn.Module):
    """ Base class to all archs """
    def __init__(self):
        super().__init__()

        self.feature_extractor = None
        self.regressor = None

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x


class CNN_LM(GeneralCNN):
    def __init__(self,
                 in_channels,
                 in_height,
                 in_width,
                 out_features,
                 padding=1):
        """ 
        A CNN whose input is of shape (in_channels, in_height, in_width) and output is of shape (out_features, ). 

        Args:
            in_channels (int): number of input channels, i.e. number of input predictors
            in_height (int): height of input, i.e. number of latitude grids
            in_width (int): width of input, i.e. number of longitude grids
            out_features (int): output dimensions, i.e. number of target grids 
            padding (int, optional): padding of nn.Conv2d module. Defaults to 1.
        """
        super().__init__()
        num_dims = (in_height + 6 * (padding - 1)) * (in_width + 6 *
                                                      (padding - 1))
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=50,
                      kernel_size=3,
                      padding=padding),
            nn.Conv2d(in_channels=50,
                      out_channels=25,
                      kernel_size=3,
                      padding=padding),
            nn.Conv2d(in_channels=25,
                      out_channels=1,
                      kernel_size=3,
                      padding=padding),
            nn.Flatten(),
        )
        self.regressor = nn.Linear(in_features=num_dims,
                                   out_features=out_features)


class CNN1(GeneralCNN):
    def __init__(self,
                 in_channels,
                 in_height,
                 in_width,
                 out_features,
                 padding=1):
        """ 
        A CNN whose input is of shape (in_channels, in_height, in_width) and output is of shape (out_features, ). 

        Args:
            in_channels (int): number of input channels, i.e. number of input predictors
            in_height (int): height of input, i.e. number of latitude grids
            in_width (int): width of input, i.e. number of longitude grids
            out_features (int): output dimensions, i.e. number of target grids 
            padding (int, optional): padding of nn.Conv2d module. Defaults to 1.
        """
        super().__init__()
        num_dims = (in_height + 6 * (padding - 1)) * (in_width + 6 *
                                                      (padding - 1))
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=50,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=50,
                      out_channels=25,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=25,
                      out_channels=1,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.regressor = nn.Linear(in_features=num_dims,
                                   out_features=out_features)


class CNN10(GeneralCNN):
    def __init__(self,
                 in_channels,
                 in_height,
                 in_width,
                 out_features,
                 padding=1):
        """ 
        A CNN whose input is of shape (in_channels, in_height, in_width) and output is of shape (out_features, ). 

        Args:
            in_channels (int): number of input channels, i.e. number of input predictors
            in_height (int): height of input, i.e. number of latitude grids
            in_width (int): width of input, i.e. number of longitude grids
            out_features (int): output dimensions, i.e. number of target grids 
            padding (int, optional): padding of nn.Conv2d module. Defaults to 1.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=50,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=50,
                      out_channels=25,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=25,
                      out_channels=10,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Flatten(),
        )
        num_dims = (in_height + 6 * (padding - 1)) * (in_width + 6 *
                                                      (padding - 1)) * 10
        self.regressor = nn.Linear(in_features=num_dims,
                                   out_features=out_features)


class CNN_PR(GeneralCNN):
    def __init__(self,
                 in_channels,
                 in_height,
                 in_width,
                 out_features,
                 padding=1):
        """ 
        A CNN whose input is of shape (in_channels, in_height, in_width) and output is of shape (out_features, ). 

        Args:
            in_channels (int): number of input channels, i.e. number of input predictors
            in_height (int): height of input, i.e. number of latitude grids
            in_width (int): width of input, i.e. number of longitude grids
            out_features (int): output dimensions, i.e. number of target grids 
            padding (int, optional): padding of nn.Conv2d module. Defaults to 1.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=10,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=10,
                      out_channels=25,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=25,
                      out_channels=50,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Flatten(),
        )
        num_dims = (in_height + 6 * (padding - 1)) * (in_width + 6 *
                                                      (padding - 1)) * 50
        self.regressor = nn.Linear(in_features=num_dims,
                                   out_features=out_features)


class CNNdense(GeneralCNN):
    def __init__(self,
                 in_channels,
                 in_height,
                 in_width,
                 out_features,
                 padding=1):
        """ 
        A CNN whose input is of shape (in_channels, in_height, in_width) and output is of shape (out_features, ). 

        Args:
            in_channels (int): number of input channels, i.e. number of input predictors
            in_height (int): height of input, i.e. number of latitude grids
            in_width (int): width of input, i.e. number of longitude grids
            out_features (int): output dimensions, i.e. number of target grids 
            padding (int, optional): padding of nn.Conv2d module. Defaults to 1.
        """
        super().__init__()
        num_dims = (in_height + 6 * (padding - 1)) * (in_width + 6 *
                                                      (padding - 1)) * 10
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=50,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=50,
                      out_channels=25,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=25,
                      out_channels=10,
                      kernel_size=3,
                      padding=padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=num_dims, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=50),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(in_features=50, out_features=out_features)


class CNN_FC(GeneralCNN):
    def __init__(self,
                 in_channels,
                 in_height,
                 in_width,
                 out_features,
                 padding=1):
        super().__init__()
        num_dims = in_height * in_width * in_channels
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_dims, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=50),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(in_features=50, out_features=out_features)


def instantiate_model(model_cls_name, *args, **kwargs):
    Model = globals()[model_cls_name]
    return Model(*args, **kwargs)


if __name__ == "__main__":
    import torch
    batch_size = 32
    in_c, in_h, in_w = 7, 17, 28  # input channels, height, width
    out_dims = 3819
    padding = 0

    in_X = torch.rand(size=(batch_size, in_c, in_h, in_w))
    for model_name in [
            "CNN_LM", "CNN_FC", "CNN1", "CNN10", "CNN_PR", "CNNdense"
    ]:
        cnn = instantiate_model(model_name, in_c, in_h, in_w, out_dims,
                                padding)
        out_Y = cnn(in_X)

        assert out_Y.shape == torch.Size((batch_size, out_dims))
        num_params = [torch.numel(param) for param in cnn.parameters()]
        print("No. of parameters", model_name, sum(num_params))
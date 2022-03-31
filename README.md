Predictor Selection for CNNs Inputting Stacked Variables
---

# Work-in-progress Statement:
    
The repository is working in progress, organizing and migrating the code from original implementations. 

# Important scripts/modules/packages
- Data preparation
    - [data_utils](data_utils/readme.md): packages for downloading, loading raw data and other processes
    - [main_data_preparation.py](main_data_preparation.py): script to load data for given regions
- Utilities
  - [utils.preprocessing.py](utils/preprocessing.py): data preprocessing and anti-processing functions
  - [utils/dataset_splits.py](utils/dataset_splits.py): module to perform dataset split
- Selector
  - [selector.predictor_selector.py](selector/predictor_selector.py): predictor selector
- CNN model
  - [dataloader.py](dataloader.py): dataloader module for Pytorch training
  - [model_wrapper.py](model_wrapper.py): module to wrap all CNN archs together with pytorch_lightning
  - [trainer.py](trainer.py): module to train the CNN, implementing based on pytorch_lightning and hydra

# Key Python requirements
- Anaconda (numpy, pandas, etc.)
- hydra-core
- xarray
- pytorch
- pytorch-lightning  
- fire

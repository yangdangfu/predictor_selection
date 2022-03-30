# Predictor Selection for CNNs Inputting Stacked Variables

Work-in-progress Statement:
    
    The repository is working in progress, organizing and migrating the code in original implementations. 

Step-by-step: 
- Data preparation
    - [data_utils](data_utils/readme.md): packages for downloading, loading raw data and other processes
    - [main_data_preparation.py](main_data_preparation.py): script to load data for given regions
- Utilities
  - [utils.preprocessing.py](utils/preprocessing.py): data preprocessing and anti-processiing functions
  - [selector/predictor_selector.py](selector/predictor_selector.py): predictor selector
- CNN model
  - [dataloader.py][dataloader.py]: dataloader module for Pytorch training

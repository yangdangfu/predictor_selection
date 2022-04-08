Predictor Selection for CNNs Inputting Stacked Variables
---

# Work-in-progress Statement:
    
The repository is working in progress, organizing and migrating the code from original implementations. 

# Quick reproduction
## Reproduce the greedy elimination based on gradient-based importance metric
1. Prepare data
```
python main_data_preparation.py --region SC
```
2. Run predictor elimination algorithm 
```
python main_select.py --reverse_sel False --multirun 10 --re_pred False --re_weight False --re_score False --region SC --model CNN10
```
3. Run reverse predictor elimination algorithm
```
python main_select.py --reverse_sel True --multirun 10 --re_pred False --re_weight False --re_score False --region SC --model CNN10
```
4. Draw results
```
python main_draw_scores.py --region SC --model CNN10 --multirun 10
```
The last drawing command will plot all the experimental scores in folder `./IMAGES`.

Modify `--multirun 10` to `--multirun 1` for a quicker experiment, which will disable the multiple-run strategy. 

Modify the arguments `model` to `CNNdense`, `CNN1`, `CNN-FC`, `CNN-LM`, `CNN-PR` to run experiments using other models. 
## Reproduce the greedy elimination based on correlation analysis
...

# Important scripts/modules/packages
Note: Scripts are the main entry of the programs that can perform certain tasks, such as `main_data_preparation.py` can be executed to prepare data; `main_train.py` is used to train the models without multiple-run; 'main_multirun.py' wraps `main_train.py` and `main_eval.py` to perform model train & evaluation with multiple-run support.
- Data preparation
    - [data_utils](data_utils/readme.md): packages for downloading, loading raw data and other processes
    - [main_data_preparation.py](main_data_preparation.py): script to load data for given regions
- Utilities
  - [utils.preprocessing.py](utils/preprocessing.py): data preprocessing and anti-processing functions
  - [utils/dataset_splits.py](utils/dataset_splits.py): module to perform dataset split
  - [utils/get_rundirs.py](utils/get_rundirs.py): functions to fetch the directories according to given parameters (the trained models of different parameters are saved in directories in a regular way)
  - [utils/cli_utils.py](utils/cli_utils.py): defined a function to check required argument list in CLI
- Selector
  - [selector.predictor_selector.py](selector/predictor_selector.py): predictor selector
- CNN model
  - [dataloader.py](dataloader.py): dataloader module for Pytorch training
  - [model_wrapper.py](model_wrapper.py): module to wrap all CNN archs together with pytorch_lightning
  - [trainer.py](trainer.py): module to train the CNN, implementing based on pytorch_lightning and hydra
  - [main_train.py](main_train.py): Script of CLI that execute CNN trainer with given parameters
  - [predict.py](predict.py): A module to run prediction on given raw input data, as well as model and its metadata
  - [score.py](score.py): A module to compute scores of prediction
  - [weights_attribution.py](weights_attribution.py): Module to attribute weights to different predictors
  - [main_eval.py](main_eval.py): Script to run prediction, evaluation and other processes for trained CNN models, all intermediate results will be saved for latter use
  - [main_multirun.py](main_multirun.py): Script for multiple-run strategy, that is run the training and evaluation for multiple times
- Linear regression
  - [predict_ml.py](predict_ml.py): Module to predict and combine grid-wise predictions using linear regression models
  - [main_ml.py](main_ml.py): Script for fitting, prediction, evaluation of linear regression models
- Program
  - [main_select.py](main_select.py): Script CLI wrapping all the routines of predictor selection on CNNs
- Plot
  - `plots` package
    - [plots/agg_scores.py](plots/agg_scores.py): Module for aggregating and averaging scores over models of multiple-run.
    - [plots/draw_boxplot.py](plots/draw_boxplot.py): Module for drawing box plots
    - [plots/draw_dist.py](plots/draw_dist.py): Module for drawing geographic distribution
  - [main_draw_scores.py](main_draw_scores.py): Script for drawing RMSE, ATCC, CC scores in line, distribution and box plots.


The K-fold cross-validation is implemented in [train](main_train.py) and [eval](main_eval.py) processes. 

The Multiple-run strategy is implemented in [main_multirun.py](main_multirun.py) by executing the train and eval multiple times. Automatic runs counting is supported. 

# Key Python requirements
- Anaconda (numpy, pandas, etc.)
- hydra-core
- xarray
- pytorch
- pytorch-lightning  
- fire
- coloredlogs, prettytabble
- xskillscore
- cmaps
- geopandas

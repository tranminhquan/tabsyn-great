# GREAT Experiment

# Install the environment

```python
conda create -n venv python=3.9
conda activate venv
pip install --upgrade pip
pip install -r requirements.txt
```

<!-- # Download and extract processed Kaggle datasets

Create directory to store data  
`mkdir data`

Download 1k2 Kaggle datasets at [GGDrive](https://drive.google.com/file/d/1oIcTzLupszhIjy6VUG7WK5l5vOYCXOEi/view?usp=drive_link) into folder `data/`

Extract data  
`cd data`  
`unzip data_v3.zip`  
`cd ../`

The above command will extract datasets into `data/processed_dataset` -->

# Pretraining

Create directory to store experiment results  
`mkdir rs`  
`mkdir rs/pretraining`


Train a model on a large dataset (pretraining)  
`python pretrain_v2.py`


Configuration
```python
DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs/pretraining'
SPLIT_INFO_PATH = 'split_3sets.json'

TOTAL_EPOCHS = 500
CHECKPOINT_EPOCH = 25 # save after every checkpoint epoch
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
```

# Finetuning

Create directory to store experiment results  
`mkdir rs/finetune_val`  
`mkdir rs/finetune_test`  

Run finetuning  
`python finetune_v2.py`  

Configuration
```python
DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs/pretraining/weights.pt'
SAVE_PATH = 'rs/finetune_val'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 

TOTAL_EPOCHS = 500
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
```

To run for test set:
* Change `SAVE_PATH` from `'rs/finetune_val'` to `'rs/finetune_test'`
* Chnange `SET_NAME` from `'val_paths'` to `'test_paths'`

# Single training

Create directory to store experiment results  
`mkdir rs/single_val`  
`mkdir rs/single_test`  

Run finetuning  
`python singletrain_v2.py`  

Configuration
```python
DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs/pretraining/weights.pt'
SAVE_PATH = 'rs/single_val'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 

TOTAL_EPOCHS = 500
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
```

To run for test set:
* Change `SAVE_PATH` from `'rs/single_val'` to `'rs/single_test'`
* Change `SET_NAME` from `'val_paths'` to `'test_paths'`

# Evaluate synthetic data

Run `python evaluate_syndata_v2.py` to generate scoring

# Report
* Clone `report_template.ipynb` and set name
* Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH`
* Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres

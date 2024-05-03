# GREAT Experiment

# Install the environment

```python
conda create -n venv python=3.9
conda activate venv
pip install --upgrade pip
pip install -r requirements.txt
```

# Download and extract processed Kaggle datasets

Create directory to store data  
`mkdir data`

Download 1k2 Kaggle datasets at [GGDrive](https://drive.google.com/file/d/1VDAwgIp6Ts_rh3Vfm6dwOzrlFabiXkv8/view?usp=drive_link) into folder `data/`

Extract data  
`cd data`  
`unzip data_v3.zip`  
`cd ../`

The above command will extract datasets into `data/processed_dataset`

# Pretraining

Create directory to store experiment results  
`mkdir rs`  
`mkdir rs/pretraining`


Train a model on a large dataset (pretraining)  
`python pretrain.py`


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
# from ctgan.synthesizers.tvae import CustomTVAE
import random

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, EarlyStoppingCallback
from be_great.great_dataset import GReaTDataset
from sklearn.model_selection import train_test_split
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_trainer import GReaTTrainer

import pandas as pd
import pickle
import json
import os

import numpy as np

import matplotlib.pyplot as plt

from utils import *

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs/pretraining/weights.pt'
SAVE_PATH = 'rs/finetuning_val'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 

TOTAL_EPOCHS = 500
# CHECKPOINT_EPOCH = 25 # save after every checkpoint epoch
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
# EMBEDDING_DIM = 128
# ENCODERS_DIMS = (512, 256, 256, 128)
# DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

MODEL_CONFIG = {
    # "input_dim": get_max_input_dim(DATA_PATH),
    "epochs": TOTAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    # "embedding_dim": EMBEDDING_DIM,
    # "compress_dims": ENCODERS_DIMS,
    # "decompress_dims": DECODER_DIMS,
    "verbose": True
}

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.model_max_length = 512
tokenizer.pad_token = tokenizer.eos_token

training_hist = []

# list_data_paths = os.listdir(data_path)
split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info[SET_NAME]
list_data_paths

for i, path in enumerate(list_data_paths):
    
    dataset_save_path = os.path.join(SAVE_PATH, path)
    path = os.path.join(DATA_PATH, path)
    df = get_df(path)
    n_rows, n_cols = len(df), len(df.columns)
        
    print(f'path: {path} | dataset: {path} | n_cols: {n_cols}, n_rows: {n_rows}')
    
    print('\t - Split')
    df, df_val = train_test_split(df, test_size=0.3, random_state=121)

    print('\t - Create training set')
    # train set
    great_ds_train = GReaTDataset.from_pandas(df)
    great_ds_train.set_tokenizer(tokenizer)

    print('\t - Create validation set')
    # val set
    great_ds_val = GReaTDataset.from_pandas(df_val)
    great_ds_val.set_tokenizer(tokenizer)
    
    if 10 < n_cols <= 20:
        MODEL_CONFIG['batch_size'] = 16
        MODEL_CONFIG['batch_size'] = 16
    
    if 20 < n_cols <= 30:
        MODEL_CONFIG['batch_size'] = 8
        MODEL_CONFIG['batch_size'] = 8
        
    if n_cols > 30:
        MODEL_CONFIG['batch_size'] = 2
        MODEL_CONFIG['batch_size'] = 2
        
    model = AutoModelForCausalLM.from_pretrained(PRETRAIN_PATH)

    training_args = TrainingArguments(
                output_dir=dataset_save_path,
                save_strategy='epoch',
                num_train_epochs=MODEL_CONFIG['epochs'],
                per_device_train_batch_size=MODEL_CONFIG['batch_size'],
                per_device_eval_batch_size=MODEL_CONFIG['batch_size'],
                logging_strategy='epoch',
                do_eval=True,
                evaluation_strategy='epoch',
                metric_for_best_model = 'eval_loss',
                save_total_limit=1,
                load_best_model_at_end=True
            )
        
    great_trainer = GReaTTrainer(
        model,
        training_args,
        train_dataset=great_ds_train,
        eval_dataset=great_ds_val,
        tokenizer=tokenizer,
        data_collator=GReaTDataCollator(tokenizer),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=7)]
    )
    
    print('\t - Training')
    # Start training
    great_trainer.train()
    
    ds_name = os.path.basename(path)

    print('\t - Update training history')
    training_hist = merge_training_hist(get_training_hist(great_trainer), ds_name, training_hist)
    print('DEBUG training_hist tail: ', training_hist.tail(10))
    print('\t -> Finished')
    
    MODEL_CONFIG['batch_size'] = BATCH_SIZE
    
    save_training_history(training_hist, SAVE_PATH)
    
    
    
# from ctgan.synthesizers.tvae import CustomTVAE
import random

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
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
SAVE_PATH = 'rs_tvaev2/pretraining_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'

TOTAL_EPOCHS = 500
CHECKPOINT_EPOCH = 20 # save after every checkpoint epoch
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
# EMBEDDING_DIM = 128
# ENCODERS_DIMS = (512, 256, 256, 128)
# DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

MODEL_CONFIG = {
    # "input_dim": get_max_input_dim(DATA_PATH),
    "epochs": 1,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    # "embedding_dim": EMBEDDING_DIM,
    # "compress_dims": ENCODERS_DIMS,
    # "decompress_dims": DECODER_DIMS,
    "verbose": True
}

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

training_args = TrainingArguments(
            output_dir=SAVE_PATH,
            save_strategy="no",
            num_train_epochs=MODEL_CONFIG['epochs'],
            per_device_train_batch_size=MODEL_CONFIG['batch_size'],
            # per_device_eval_batch_size=32,
            logging_strategy='epoch',
            do_eval=True,
            evaluation_strategy='epoch',
        )

training_hist = []

# list_data_paths = os.listdir(data_path)
split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info['pretrain_paths']
list_data_paths

for epoch in range(TOTAL_EPOCHS):
    
    # if epoch == 0:
    #     init_transformer = True
    # else:
    #     init_transformer = False
    
    random.shuffle(list_data_paths)
    print(f'Epoch {epoch} with shuffled datasets {list_data_paths}')
    
    for i, path in enumerate(list_data_paths):
        
        print(f'\t{path}')
        
        path = os.path.join(DATA_PATH, path)
        
        df, df_val = train_test_split(df, test_size=0.3, random_state=121)

        # train set
        great_ds_train = GReaTDataset.from_pandas(df)
        great_ds_train.set_tokenizer(tokenizer)

        # val set
        great_ds_val = GReaTDataset.from_pandas(df_val)
        great_ds_val.set_tokenizer(tokenizer)
        
        great_trainer = GReaTTrainer(
            model,
            training_args,
            train_dataset=great_ds_train,
            eval_dataset=great_ds_val,
            tokenizer=tokenizer,
            data_collator=GReaTDataCollator(tokenizer),
        )

        # Start training
        great_trainer.train()
        
        ds_name = os.path.basename(path)
        
        training_hist = merge_training_hist(get_training_hist(model), ds_name, training_hist)
        
    # save checkpoint
    if epoch >= CHECKPOINT_EPOCH and epoch % CHECKPOINT_EPOCH == 0:
        checkpoint = f'checkpoint_{epoch}'
        model_save_path = os.path.join(SAVE_PATH, f'weights_{checkpoint}.pt')
        great_trainer.save_model(model_save_path)
    
    # save training history at each epoch    
    save_training_history(training_hist, SAVE_PATH)

save_model_weights(model, SAVE_PATH)
save_training_history(training_hist, SAVE_PATH)
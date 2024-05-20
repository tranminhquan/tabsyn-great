# import os
# import json
# from utils import *

# DATA_PATH = 'data/processed_dataset'
# SPLIT_INFO_PATH = 'split_3sets.json'
# SET_NAME = 'val_paths'
# FINETUNE_PATH = 'rs/finetune_val'
# SINGLETRAIN_PATH = 'rs/single_val'
# SCORE_SAVE_PATH = 'rs/scores_val.csv'

# TOTAL_EPOCHS = 500
# BATCH_SIZE = 32 # paper
# LR = 5.e-5 # paper
# # EMBEDDING_DIM = 128
# # ENCODERS_DIMS = (512, 256, 256, 128)
# # DECODER_DIMS = (128, 256, 256, 512)

# MODEL_CONFIG = {
#     # "input_dim": get_max_input_dim(DATA_PATH),
#     "epochs": 1,
#     "batch_size": BATCH_SIZE,
#     "lr": LR,
#     # "embedding_dim": EMBEDDING_DIM,
#     # "compress_dims": ENCODERS_DIMS,
#     # "decompress_dims": DECODER_DIMS,
#     "verbose": True
# }

# list_data_paths = json.load(open(SPLIT_INFO_PATH, 'r'))[SET_NAME]

# ft_merged_df = []
# st_merged_df = []

# for ds in list_data_paths:
    
#     # path
#     path = os.path.join(DATA_PATH, ds)
    
#     # load artifacts
#     real_data = get_df(path)
#     transformer = get_transformer_v3(path)
#     metadata = get_metadata(path)
#     n_samples = len(real_data)
    
#     # find the optimal checkpoint
#     # ft_loss, ft_val_loss = process_df_by_dataset(ft_training_hist, ds)
#     # st_loss, st_val_loss = process_df_by_dataset(st_training_hist, ds)
    
#     # ft_encoder_info, ft_decoder_info = get_checkpoint_info(ft_val_loss, ds)
#     # st_encoder_info, st_decoder_info = get_checkpoint_info(st_val_loss, ds)
    
#     ft_encoder_info, ft_decoder_info = str(ds) + '_encoder', str(ds) + '_decoder'
#     st_encoder_info, st_decoder_info = str(ds) + '_encoder', str(ds) + '_decoder'
    
#     # load weights
#     ft_model = CustomTVAEv2(**FINETUNE_MODEL_CONFIG)
#     ft_model = load_model_weights(ft_model, FINETUNE_PATH, [ft_encoder_info, ft_decoder_info])
    
#     SINGLE_MODEL_CONFIG['input_dim'] = transformer.output_dimensions
#     st_model = CustomTVAEv2(**SINGLE_MODEL_CONFIG)
#     st_model = load_model_weights(st_model, SINGLETRAIN_PATH, [st_encoder_info, st_decoder_info])
    
#     # generate
#     ft_syn_data = ft_model.sample(n_samples, transformer)
#     st_syn_data = st_model.sample(n_samples, transformer)
#     filtered_metadata = filter_metdata(metadata, st_syn_data.columns)
    
#     # scoring
#     ft_report = scoring(real_data, ft_syn_data, filtered_metadata)
#     st_report = scoring(real_data, st_syn_data, filtered_metadata)
    
#     ft_merged_df = add_score_df(ft_report, ds, ft_merged_df)
#     st_merged_df = add_score_df(st_report, ds, st_merged_df)
    

# # save ft and st scores
# ft_merged_df.to_csv(os.path.join(FINETUNE_PATH, 'scores.csv'))
# st_merged_df.to_csv(os.path.join(SINGLETRAIN_PATH, 'scores.csv'))

# # merge scores
# score_df = ft_merged_df.merge(st_merged_df, on='dataset', suffixes=['_ft', '_st'])
    
# # add average row
# average_data = score_df.drop(columns=['dataset']).mean().to_dict()
# average_data['dataset'] = 'AVERAGE'
# score_df.loc[len(score_df)] = average_data

# score_df.to_csv(SCORE_SAVE_PATH)
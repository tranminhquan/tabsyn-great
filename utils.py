
import os
import pandas as pd

def get_training_hist(trainer):
    return pd.DataFrame(trainer.state.log_history)

def merge_training_hist(new_hist, dataset_name, merged_hist):
    
    hist = new_hist.copy()
    hist['dataset'] = len(hist) * [str(dataset_name)]
    
    if len(merged_hist) == 0:
        merged_hist = hist
    
    else:
        merged_hist = pd.concat([merged_hist, hist])
        
    return merged_hist

def save_training_history(training_hist: pd.DataFrame, path: str):
    # index as orders to remember the sequence of training
    training_hist.index = range(len(training_hist))
    training_hist.to_csv(os.path.join(path, 'training_hist.csv'))
    
def save_model_weights(trainer, save_path):
    trainer.save_model()
    
def save_model_weights(trainer, path: str, save_name=None):
    
    if save_name is None:
        trainer.save_model(os.path.join(path, 'weights.pt'))
    else:
        trainer.save_model(os.path.join(path, f'{save_name}.pt'))
        

def get_df(path) -> pd.DataFrame:
    return pd.read_csv(path + '/' + path.split('/')[-1] + '.csv', index_col=False)
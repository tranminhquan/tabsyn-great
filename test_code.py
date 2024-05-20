from utils import *
from sklearn.model_selection import train_test_split
from be_great.great_dataset import GReaTDataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from be_great.great_trainer import GReaTTrainer
from be_great.great_dataset import GReaTDataset, GReaTDataCollator

DATA_PATH = 'data/processed_dataset'
# path = 'korea-corona'
path = 'diabetes-readmissions-column-annotation'

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

training_args = TrainingArguments(
            output_dir='test_code',
            save_strategy="no",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            # per_device_eval_batch_size=32,
            logging_strategy='epoch',
            do_eval=True,
            evaluation_strategy='epoch',
        )


path = os.path.join(DATA_PATH, path)
df = get_df(path)

n_rows, n_cols = len(df), len(df.columns)

print(f'dataset: {path} | n_cols: {n_cols}, n_rows: {n_rows}')

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
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 16

if 20 < n_cols <= 30:
    training_args.per_device_train_batch_size = 8
    training_args.per_device_eval_batch_size = 8
    
if n_cols > 30:
    training_args.per_device_train_batch_size = 2
    training_args.per_device_eval_batch_size = 2

great_trainer = GReaTTrainer(
    model,
    training_args,
    train_dataset=great_ds_train,
    eval_dataset=great_ds_val,
    tokenizer=tokenizer,
    data_collator=GReaTDataCollator(tokenizer),
)

print('\t - Training')
# Start training
great_trainer.train()
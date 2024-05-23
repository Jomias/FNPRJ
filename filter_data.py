import pandas as pd
import numpy as np
import time
from fen import start_position
df = pd.read_csv("data/chessData.csv")
test_df = pd.read_csv('data/tactic_evals.csv')

df = df[~df['Evaluation'].str.startswith('#')].sample(n=12000000, random_state=42)
df['Evaluation'] = df['Evaluation'].astype(int)
test_df = test_df.iloc[:, :-1]
test_df = test_df[~test_df['Evaluation'].str.startswith('#')].sample(n=2200000, random_state=42)
test_df['Evaluation'] = test_df['Evaluation'].astype(int)
split_train = 0.9
split_val = 0.1
n = df.shape[0]
m_train = int(n * split_train)
m_val = n - m_train
train_df = df[:m_train]
val_df = df[-m_val:]

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)


# final_df = pd.concat([train_df, val_df], ignore_index=True)
# final_df.to_csv('data/final_data.csv', index=False)
final_df = pd.read_csv('data/final_data.csv')

random_state = 42
sample_df = final_df.sample(n=2400000, random_state=random_state)
sample_df.to_csv('data/small_samples.csv', index=False)
import pandas as pd

train_df = pd.read_csv('data/modified_train.csv')
test_df = pd.read_csv('data/modified_test.csv')

print(train_df.shape)
print(test_df.shape)
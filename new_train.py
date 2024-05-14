import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import chess
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
validation_df = pd.read_csv('data/val.csv')

pieces = list('rnbqkpRNBQKP.')


def one_hot_encode_piece(piece):
    return np.eye(len(pieces))[pieces.index(piece)]


def encode_board(board):
    board_str = str(board).replace(' ', '').replace('\n', '')  # Exclude newline characters
    return np.array([one_hot_encode_piece(piece) for piece in board_str])


def encode_fen_string(fen_str):
    return encode_board(chess.Board(fen=fen_str))


def encode_data(df):
    X = np.stack(df['FEN'].apply(encode_fen_string))
    y = df['Evaluation'].values
    return X, y


split_train = 0.9
split_val = 0.1
n = train_df.shape[0]
m_train, m_val = int(n * split_train), int(n * split_val)
train_df = train_df[:m_train]
val_df = train_df[-m_val:]


class ChessDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fen_str = self.df.iloc[idx]['FEN']
        evaluation = self.df.iloc[idx]['Evaluation']
        X = encode_fen_string(fen_str).flatten()
        y = evaluation
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Datasets
train_dataset = ChessDataset(train_df)
val_dataset = ChessDataset(val_df)
test_dataset = ChessDataset(test_df)

X_train, y_train = encode_data(train_df)
X_val, y_val = encode_data(val_df)
X_test, y_test = encode_data(test_df)

X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# Data Loaders
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize weights
k = 4
weights = 10 ** (-k) * np.array([-500, -300, -300, -900, -500, -100, 500, 300, 300, 900, 500, 100, 0])
init_1 = np.zeros((64 * 13, 256))
for n in range(64 * 13):
    square = n // 13
    piece = n % 13
    init_1[n][square] = weights[piece]

init_2 = np.eye(256, 64)
init_3 = np.eye(64)
init_4 = np.zeros((64, 8))
for n in range(64):
    row = n // 8
    init_4[n][row] = 1

init_5 = 10 ** k * np.ones((8, 1))


# PyTorch model definition
class ChessModel(nn.Module):
    def __init__(self, init_1, init_2, init_3, init_4, init_5, drop_out_1=1e-4, drop_out_2=1e-4):
        super(ChessModel, self).__init__()
        self.fc1 = nn.Linear(64 * 13, 256)
        self.fc1.weight.data = torch.tensor(init_1, dtype=torch.float32).t()
        self.fc1.bias.data.fill_(0)

        self.dropout1 = nn.Dropout(drop_out_1)
        self.fc2 = nn.Linear(256, 64)
        self.fc2.weight.data = torch.tensor(init_2, dtype=torch.float32).t()
        self.fc2.bias.data.fill_(0)

        self.fc3 = nn.Linear(64, 64)
        self.fc3.weight.data = torch.tensor(init_3, dtype=torch.float32).t()
        self.fc3.bias.data.fill_(0)

        self.fc4 = nn.Linear(64, 8)
        self.fc4.weight.data = torch.tensor(init_4, dtype=torch.float32).t()
        self.fc4.bias.data.fill_(0)

        self.dropout2 = nn.Dropout(drop_out_2)
        self.fc5 = nn.Linear(8, 1)
        self.fc5.weight.data = torch.tensor(init_5, dtype=torch.float32).t()
        self.fc5.bias.data.fill_(0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.dropout2(x)
        x = self.fc5(x)
        return x


# Hyperparameters
learning_rates = [1e-3, 5e-4, 1e-4]
rhos = [0.95, 0.9]
drop_outs = [0.001, 0.005]
epochs = 40
last_val_loss = {}

# Training loop
for rho in rhos:
    for drop_out in drop_outs:
        for lr in learning_rates:
            model = ChessModel(init_1, init_2, init_3, init_4, init_5, drop_out, drop_out)
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=rho)
            criterion = nn.L1Loss()
            model.train()

            train_losses, val_losses = [], []

            for epoch in range(epochs):
                epoch_train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch).squeeze()
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()

                epoch_train_loss /= len(train_loader)
                train_losses.append(epoch_train_loss)

                model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_pred = model(X_batch).squeeze()
                        loss = criterion(y_pred, y_batch)
                        epoch_val_loss += loss.item()

                epoch_val_loss /= len(val_loader)
                val_losses.append(epoch_val_loss)

                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

            plt.plot(train_losses, label='train')
            plt.plot(val_losses, label='val')

            string = f'rho={rho} drop_out={drop_out} lr={lr}'
            plt.title(string)
            plt.xlabel('Epochs')
            plt.ylabel('Mean Absolute Error')
            plt.legend()

            last_val_loss[string] = val_losses[-1]
            plt.text(epochs - 1, val_losses[-1], f'{val_losses[-1]:.4f}', ha='center', va='bottom')
            plt.show()

print(last_val_loss)

# Final model
rho = 0.95
lr = 0.001
drop_out = 0.001
epochs = 50
batch_size = 256

model = ChessModel(init_1, init_2, init_3, init_4, init_5, drop_out, drop_out)
optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=rho)
criterion = nn.L1Loss()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

train_losses, val_losses = [], []

for epoch in range(epochs):
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            epoch_val_loss += loss.item()

    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

# Plot MAE for training and validation
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')

string = f'rho={rho} bs={batch_size} lr={lr}'
plt.title(string)
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

last_val_loss = val_losses[-1]
plt.text(epochs - 1, last_val_loss, f'{last_val_loss:.4f}', ha='center', va='bottom')
plt.show()

#
# # Evaluate the model
# model.eval()
# y_pred_val = []
# with torch.no_grad():
#     for X_batch in val_loader:
#         y_pred = model(X_batch[0]).squeeze()
#         y_pred_val.extend(y_pred.numpy())
#
# print('MAE Score: ', mean_absolute_error(y_pred_val, y_val))
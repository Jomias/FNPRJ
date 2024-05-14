# !pip install chess
import chess
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt


pieces = list('rnbqkpRNBQKP.')
piece_to_index = {p: i for i, p in enumerate(pieces)}


def one_hot_encode_piece(piece):
    return np.eye(len(pieces))[pieces.index(piece)]


def encode_board(board):
    board_str = str(board).replace(' ', '').replace('\n', '')  # Exclude newline characters
    return np.array([one_hot_encode_piece(piece) for piece in board_str])

def encode_fen_string(fen_str):
    return encode_board(chess.Board(fen=fen_str))


import chess
import json
import numpy as np
import pandas as pd
import sqlite3
from fen import start_position
import torch

# Định nghĩa các hàm mã hóa
pieces = list('rnbqkpRNBQKP.')
piece_to_index = {p: i for i, p in enumerate(pieces)}


def one_hot_encode_piece(piece):
    return np.eye(len(pieces))[pieces.index(piece)]


def encode_board(board):
    board_str = str(board).replace(' ', '').replace('\n', '')  # Loại bỏ kí tự newline
    return np.array([one_hot_encode_piece(piece) for piece in board_str])


def encode_fen_string(fen_str):
    return encode_board(chess.Board(fen=fen_str)).flatten()


# Đọc dữ liệu từ CSV
small_sample_df = pd.read_csv('data/final_data.csv')

# Kết nối đến cơ sở dữ liệu SQLite
conn = sqlite3.connect('database/final.db')
cursor = conn.cursor()

# Drop bảng nếu tồn tại
cursor.execute('''DROP TABLE IF EXISTS chess_samples''')

# Tạo bảng mới trong cơ sở dữ liệu
cursor.execute('''CREATE TABLE chess_samples (id INTEGER PRIMARY KEY, FEN TEXT, Evaluation REAL)''')

# Lưu từng hàng từ DataFrame vào cơ sở dữ liệu
for index, row in small_sample_df.iterrows():
    print(index)
    fen_json = row['FEN']  # Chuyển đổi FEN thành chuỗi JSON
    evaluation = row['Evaluation']  # Lấy giá trị đánh giá
    cursor.execute('''INSERT INTO chess_samples (FEN, Evaluation) VALUES (?, ?)''', (fen_json, evaluation))

# Lưu các thay đổi và đóng kết nối
conn.commit()
print("done")
conn.close()

# id_to_query = 2400000
# cursor.execute('''SELECT * FROM chess_samples WHERE id = ?''', (1,))
# row = cursor.fetchone()
#
# # Kiểm tra xem dữ liệu có tồn tại không
# if row:
#     fen_json = row[1]  # Lấy dữ liệu JSON từ cột FEN
#     evaluation = row[2]  # Lấy giá trị đánh giá từ cột Evaluation
#
#     print("FEN JSON:", fen_json)
#     print("Evaluation:", evaluation)
# else:
#     print("Không tìm thấy dữ liệu cho id:", id_to_query)
#
# # Đóng kết nối
# conn.close()
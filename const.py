import numpy as np


"""
Find bit in bit board helper
"""
de_bruijn = np.uint64(0x03f79d71b4cb0a89)


"""
Basic const for board
"""
ROWS, COLS = 8, 8
EMPTY, BIT, FULL = np.uint64(0), np.uint64(1), np.uint64(0xFFFFFFFFFFFFFFFF)
a, b, c, d, e, f, g, h = range(8)
a8, b8, c8, d8, e8, f8, g8, h8, \
    a7, b7, c7, d7, e7, f7, g7, h7, \
    a6, b6, c6, d6, e6, f6, g6, h6, \
    a5, b5, c5, d5, e5, f5, g5, h5, \
    a4, b4, c4, d4, e4, f4, g4, h4, \
    a3, b3, c3, d3, e3, f3, g3, h3, \
    a2, b2, c2, d2, e2, f2, g2, h2, \
    a1, b1, c1, d1, e1, f1, g1, h1, no_sq = range(65)

sq_to_fr = (
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "-"
)

reversed_board = (
    h1, g1, f1, e1, d1, c1, b1, a1,
    h2, g2, f2, e2, d2, c2, b2, a2,
    h3, g3, f3, e3, d3, c3, b3, a3,
    h4, g4, f4, e4, d4, c4, b4, a4,
    h5, g5, f5, e5, d5, c5, b5, a5,
    h6, g6, f6, e6, d6, c6, b6, a6,
    h7, g7, f7, e7, d7, c7, b7, a7,
    h8, g8, f8, e8, d8, c8, b8, a8
)

"""
Rank and file
"""
a_file, b_file, c_file, d_file, e_file, f_file, g_file, h_file = np.array(
    [0x0101010101010101 << i for i in range(8)], dtype=np.uint64
)
FILES = np.array((a_file, b_file, c_file, d_file, e_file, f_file, g_file, h_file))
rank_8, rank_7, rank_6, rank_5, rank_4, rank_3, rank_2, rank_1 = np.array(
    [0x00000000000000FF << 8 * i for i in range(8)], dtype=np.uint64
)
RANKS = np.array((rank_8, rank_7, rank_6, rank_5, rank_4, rank_3, rank_2, rank_1))


"""
Side
"""
WHITE, BLACK, BOTH = range(3)
USER, COMPUTER = range(2)


"""
Pieces
"""
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
ascii_pieces = "PNBRQKpnbrqk."
unicode_pieces = ("♟", "♞", "♝", "♜", "♛", "♚", "♙", "♘", "♗", "♖", "♕", "♔", " ")


"""
State
"""
NORMAL, STALEMATE, CHECKMATE, DRAW, RESIGN, INSUFFICIENT_MATERIAL = range(6)
OPENING, END_GAME = np.arange(2, dtype=np.uint8)


"""
Mark for castle position
"""
wk, wq, bk, bq = (1 << i for i in range(4))


"""
Zobrist Hashing
"""
MAX_HASH_SIZE = 0x400000
seed_value = 42


"""
AI const
"""
MAX_PLY = 64
BOUND_INF = 50000
LOWER_MATE = 48000
UPPER_MATE = 49000

# LMR
full_depth_moves = 4
reduction_limit = 3

# transposition flag
hash_flag_exact, hash_flag_alpha, hash_flag_beta = range(3)
no_hash_entry = 100000
time_precision = 2047


"""
UI const
"""
GAME_WIDTH = 640
GAME_HEIGHT = 640
RECORD_WIDTH = 400
BOARD_WIDTH = 640
BOARD_HEIGHT = 640
SQUARE_SIZE = BOARD_WIDTH // COLS
WHITE_COLOR = (119, 154, 88)
DARK_COLOR = (234, 235, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LAST_MOVE_COLOR = (171, 214, 248, 127)
SQUARE_MARK = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h'}


"""
game const
"""
EASY, MEDIUM, HARD = range(3)
stopped = False


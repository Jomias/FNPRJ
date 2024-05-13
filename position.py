import numpy as np
from numba import uint64, int64, njit, typed
from numba.experimental import jitclass
from const import WHITE, BLACK, no_sq, NORMAL, sq_to_fr, ascii_pieces, wk, wq, bk, bq
from type import position_type
from bitboard_helper import lsb_index, pop_bit
from prettytable import PrettyTable, ALL
from bitboard_helper import get_bit


@jitclass(position_type)
class Position:
    def __init__(self):
        self.pieces = np.zeros((2, 6), dtype=np.uint64)
        self.occupancies = np.zeros(3, dtype=np.uint64)
        self.side = WHITE
        self.eps = no_sq
        self.castle = no_sq
        self.hash_key = uint64(0)
        self.state = NORMAL
        self.repetitions = typed.Dict.empty(key_type=uint64, value_type=int64)
        self.half_move_clock = 0
        self.full_move_count = 1

    def extract_bit_vector(self):
        # Create a bit vector including side, eps, castle, half_move_clock, and full_move_count
        bit_vector = np.concatenate((
            self.pieces.ravel(),
            np.array([self.side], dtype=np.uint8),
            np.array([self.eps], dtype=np.uint8),
            np.array([self.castle], dtype=np.uint8),
            np.array([self.half_move_clock], dtype=np.int64),
            np.array([self.full_move_count], dtype=np.int64)
        ))
        return bit_vector



# Load the arrays from files
piece_keys = np.load('pre_calculate/data/piece_keys.npy')
en_passant_keys = np.load('pre_calculate/data/en_passant_keys.npy')
castle_keys = np.load('pre_calculate/data/castle_keys.npy')
side_key = uint64(np.load('pre_calculate/data/side_key.npy'))


@njit(uint64(Position.class_type.instance_type))
def generate_hash_key(pos: Position):
    final_key = 0
    for color in range(2):
        for piece in range(6):
            bb = pos.pieces[color][piece]
            while bb:
                sq = lsb_index(bb)
                final_key ^= piece_keys[color][piece][sq]
                bb = pop_bit(bb, sq)
    final_key ^= en_passant_keys[pos.eps] ^ castle_keys[pos.castle]
    if pos.side == BLACK:
        final_key ^= side_key
    return final_key


def print_board(cur_pos, details=True):
    print()
    cur_board = PrettyTable()
    cur_board.header = False
    cur_board.hrules = ALL
    for rank in range(8):
        row_output = []
        for file in range(8):
            square = rank * 8 + file
            piece = -1
            for i in range(2):
                for j in range(6):
                    if get_bit(cur_pos.pieces[i][j], square):
                        piece = 6 * i + j
            if piece != -1:
                row_output.append(ascii_pieces[piece])
            else:
                row_output.append('')
        cur_board.add_row(row_output + [8 - rank])
    cur_board.add_row([f"{col}" for col in 'ABCDEFGH '])
    print(cur_board)
    if details:
        print("Side: " + ("white" if cur_pos.side == WHITE else "black"))
        print("Enpassant: " + (sq_to_fr[cur_pos.eps] if cur_pos.eps != no_sq else "no"))
        casl = (
            f"{'K' if cur_pos.castle & wk else ''}{'Q' if cur_pos.castle & wq else ''}"
            f"{'k' if cur_pos.castle & bk else ''}{'q' if cur_pos.castle & bq else ''} "
        )
        print("Castling:", casl if casl else "-")

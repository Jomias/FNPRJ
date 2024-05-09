from numba.experimental import jitclass
from numba import njit, b1, uint8, uint64
from const import WHITE, sq_to_fr, ascii_pieces, unicode_pieces

"""
           Binary move bits             Meaning          Hexadecimal

    0000 0000 0000 0000 0011 1111    source square       0x3f           6bit to occupied source
    0000 0000 0000 1111 1100 0000    target square       0xfc0          6bit to occupied target square h1:63
    0000 0000 0111 0000 0000 0000    piece               0x7000         6 piece each
    0000 0000 1000 0000 0000 0000    side                0x8000         1 bit for side
    0000 1111 0000 0000 0000 0000    promoted piece      0xf0000        1 bit for each piece from N to Q
    0001 0000 0000 0000 0000 0000    capture flag        0x100000
    0010 0000 0000 0000 0000 0000    double push flag    0x200000
    0100 0000 0000 0000 0000 0000    enpassant flag      0x400000
    1000 0000 0000 0000 0000 0000    castling flag       0x800000

"""


@njit(cache=True)
def encode_move(source, target, piece, side=WHITE, promote_to=0, capture=False, double_push=False, enpassant=False, castling=False):
    return source | target << 6 | piece << 12 | side << 15 | promote_to << 16 | capture << 20 | \
        double_push << 21 | enpassant << 22 | castling << 23


@njit(uint8(uint64), cache=True)
def get_move_source(move):
    return move & 0x3f


@njit(uint8(uint64), cache=True)
def get_move_target(move):
    return (move & 0xfc0) >> 6


@njit(uint8(uint64), cache=True)
def get_move_piece(move):
    return (move & 0x7000) >> 12


@njit(uint8(uint64), cache=True)
def get_move_side(move):
    return bool(move & 0x8000)


@njit(uint8(uint64), cache=True)
def get_move_promote_to(move):
    return (move & 0xf0000) >> 16


@njit(b1(uint64), cache=True)
def get_move_capture(move):
    return bool(move & 0x100000)


@njit(b1(uint64), cache=True)
def get_move_double_push(move):
    return bool(move & 0x200000)


@njit(uint8(uint64), cache=True)
def get_move_en_passant(move):
    return bool(move & 0x400000)


@njit(b1(uint64), cache=True)
def get_move_castling(move):
    return bool(move & 0x800000)


@njit(cache=True)
def get_move_uci(move):
    return str(sq_to_fr[move.source]) + str(sq_to_fr[move.target]) + \
        (ascii_pieces[5 + move.promote_to] if move.promote_to else '')


move_spec = [
    ('source', uint8),          # 6 bits for source square
    ('target', uint8),          # 6 bits for target square
    ('piece', uint8),           # 4 bits for piece
    ('side', b1),            # 1 bit for side
    ('promote_to', uint8),      # 4 bits for promoted piece
    ('capture', b1),         # 1 bit for capture flag
    ('double_push', b1),     # 1 bit for double push flag
    ('enpassant', b1),       # 1 bit for en passant flag
    ('castling', b1),        # 1 bit for castling flag
    ('encode', uint64)   # first encode move
]


@jitclass(move_spec)
class Move:
    def __init__(self, move: uint64):
        self.source = get_move_source(move)
        self.target = get_move_target(move)
        self.piece = get_move_piece(move)
        self.side = get_move_side(move)
        self.promote_to = get_move_promote_to(move)
        self.capture = get_move_capture(move)
        self.double_push = get_move_double_push(move)
        self.enpassant = get_move_en_passant(move)
        self.castling = get_move_castling(move)
        self.encode = move

    def __eq__(self, other):
        return self.encode == other.encode


def print_move_list(move_list):
    if not move_list:
        print("Empty move_list")
        return
    print("\n  {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format("Move", "Piece", "Promote", "Capture", "Double", "Castling"))
    for move in move_list:
        move_str = f"{sq_to_fr[move.source]}{sq_to_fr[move.target]}"
        promote_str = unicode_pieces[move.side * 6 + move.promote_to] if move.promote_to else ''
        piece_str = unicode_pieces[move.side * 6 + move.piece]
        capture_str = move.capture
        double_str = move.double_push
        castling_str = move.castling
        print("  {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(move_str, piece_str, promote_str, capture_str, double_str, castling_str))
    print("Total number of moves:", len(move_list))

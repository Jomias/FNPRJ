from position import Position, generate_hash_key
from const import WHITE, BLACK, BOTH, no_sq, sq_to_fr
from numba import njit, types, typed
from bitboard_helper import set_bit

empty_board = "8/8/8/8/8/8/8/8 w - - "
start_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
tricky_position = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
killer_position = "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1"
cmk_position = "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9"
repetitions_position = "2r3k1/R7/8/1R6/8/8/P4KPP/8 w - - 0 40"
mate_in_3 = "1k6/6R1/3K4/8/8/8/8/8 w - - 18 10"
mate_in_2 = "k7/6R1/2K5/8/8/8/8/8 w - - 16 9"
mate_in_4 = "2k5/5R2/3K4/8/8/8/8/8 w - - 12 7"
mate_in_1 = "4k3/8/5K2/8/1Q6/8/8/8 w - - 0 1"
bnk_mate = "1k6/8/8/8/8/8/8/1K2BN2 w - - 0 1"
stale_mate = "k7/8/1K1Q4/8/8/8/8/8 w - - 0 1"


@njit(Position.class_type.instance_type(types.string))
def parse_fen(cur_fen: str):
    cur_pos = Position()
    num_str_to_int = typed.Dict.empty(types.string, types.int64)
    for num in range(1, 9):
        num_str_to_int[str(num)] = num
    let_str_to_int = typed.Dict.empty(types.unicode_type, types.int64)
    for side in (("P", "N", "B", "R", "Q", "K"), ("p", "n", "b", "r", "q", "k")):
        for code, letter in enumerate(side):
            let_str_to_int[letter] = code
    cur_board, color, castle, ep, _, _ = cur_fen.split()
    cur_pos.side = WHITE if color == "w" else BLACK

    if ep == "-":
        cur_pos.eps = no_sq
    else:
        for i, sq in enumerate(sq_to_fr):
            if sq == ep:
                cur_pos.eps = i
                break
    cur_pos.castle = 0
    for i, character in enumerate("KQkq"):
        if character in castle:
            cur_pos.castle += 1 << i
    sq = 0
    for character in cur_board:
        if character.isupper():  # WHITE
            piece = let_str_to_int[character]
            cur_pos.pieces[WHITE][piece] = set_bit(cur_pos.pieces[WHITE][piece], sq)
            sq += 1
        elif character.islower():  # BLACK
            piece = let_str_to_int[character]
            cur_pos.pieces[BLACK][piece] = set_bit(cur_pos.pieces[BLACK][piece], sq)
            sq += 1
        elif character.isnumeric():  # Empty
            sq += num_str_to_int[character]

    for i in range(2):
        for bb in cur_pos.pieces[i]:
            cur_pos.occupancies[i] |= bb
    cur_pos.occupancies[BOTH] = cur_pos.occupancies[WHITE] | cur_pos.occupancies[BLACK]
    cur_pos.hash_key = generate_hash_key(cur_pos)
    cur_pos.repetitions[cur_pos.hash_key] = 1
    return cur_pos

import numpy as np
from position import Position
from const import BOTH, KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN
from numba import njit, b1, uint8, uint64, int64
from stored import rook_magic_numbers, bishop_magic_numbers, bishop_relevant_bits, rook_relevant_bits


bishop_attacks = np.load('pre_calculate/data/bishop_attacks.npy')
rook_attacks = np.load('pre_calculate/data/rook_attacks.npy')
pawn_attacks = np.load('pre_calculate/data/pawn_attacks.npy')
knight_attacks = np.load('pre_calculate/data/knight_attacks.npy')
king_attacks = np.load('pre_calculate/data/king_attacks.npy')
bishop_masks = np.load('pre_calculate/data/bishop_masks.npy')
rook_masks = np.load('pre_calculate/data/rook_masks.npy')


@njit(uint64(uint8, uint64), cache=True)
def get_bishop_attacks(sq, occ):
    occ &= bishop_masks[sq]
    occ *= bishop_magic_numbers[sq]
    occ >>= 64 - bishop_relevant_bits[sq]
    return bishop_attacks[sq][occ]


@njit(uint64(uint8, uint64))
def get_rook_attacks(sq, occ):
    occ &= rook_masks[sq]
    occ *= rook_magic_numbers[sq]
    occ >>= 64 - rook_relevant_bits[sq]
    return rook_attacks[sq][occ]


@njit(uint64(uint8, uint64))
def get_queen_attacks(sq, occ):
    return get_rook_attacks(sq, occ) | get_bishop_attacks(sq, occ)


@njit(uint64(int64, uint8, Position.class_type.instance_type))
def get_attacks(piece, src, pos):
    ps = pos.occupancies
    if piece == KNIGHT:
        return knight_attacks[src] & ~ps[pos.side]
    if piece == BISHOP:
        return get_bishop_attacks(src, ps[BOTH]) & ~ps[pos.side]
    if piece == ROOK:
        return get_rook_attacks(src, ps[BOTH]) & ~ps[pos.side]
    if piece == QUEEN:
        return get_queen_attacks(src, ps[BOTH]) & ~ps[pos.side]
    if piece == KING:
        return king_attacks[src] & ~ps[pos.side]
    return 0


@njit(uint64(int64, uint8, uint64, int64))
def get_control_squares(piece, src, bot_occ, side):
    if piece == PAWN:
        return pawn_attacks[side][src]
    if piece == KNIGHT:
        return knight_attacks[src]
    if piece == BISHOP:
        return get_bishop_attacks(src, bot_occ)
    if piece == ROOK:
        return get_rook_attacks(src, bot_occ)
    if piece == QUEEN:
        return get_queen_attacks(src, bot_occ)
    if piece == KING:
        return king_attacks[src]
    return 0


@njit(b1(Position.class_type.instance_type, uint8))
def is_sq_attacked(cur_pos, sq):
    opp, pieces, occ = cur_pos.side ^ 1, cur_pos.pieces, cur_pos.occupancies
    return pawn_attacks[cur_pos.side][sq] & pieces[opp][PAWN] \
        or get_bishop_attacks(sq, occ[BOTH]) & pieces[opp][BISHOP] \
        or knight_attacks[sq] & pieces[opp][KNIGHT] \
        or get_rook_attacks(sq, occ[BOTH]) & pieces[opp][ROOK] \
        or get_queen_attacks(sq, occ[BOTH]) & pieces[opp][QUEEN] \
        or king_attacks[sq] & cur_pos.pieces[opp][KING]
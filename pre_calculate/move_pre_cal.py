from const import FILES, BIT, EMPTY, WHITE, a, b, g, h
from numba import njit, b1, uint, uint8, uint16, uint64
from bitboard_helper import set_bit, pop_bit, lsb_index, count_bits
from stored import rook_magic_numbers, bishop_magic_numbers, bishop_relevant_bits, rook_relevant_bits
import numpy as np


@njit(uint64(uint, uint8), cache=True)
def mask_pawn_attacks(color, sq):
    bb, attacks = set_bit(EMPTY, sq), 0
    attacks |= ((bb >> 7) & ~FILES[a]) | ((bb >> 9) & ~FILES[h]) if color == WHITE \
        else ((bb << 7) & ~FILES[h]) | ((bb << 9) & ~FILES[a])
    return attacks


@njit(uint64(uint), cache=True)
def mask_knight_attacks(sq):
    bb = set_bit(EMPTY, sq)
    return (((bb & ~(FILES[a] | FILES[b])) << 6) | ((bb & ~FILES[a]) << 15) | ((bb & ~FILES[h]) << 17) |
            ((bb & ~(FILES[h] | FILES[g])) << 10) | ((bb & ~(FILES[h] | FILES[g])) >> 6) | ((bb & ~FILES[h]) >> 15) |
            ((bb & ~FILES[a]) >> 17) | ((bb & ~(FILES[a] | FILES[b])) >> 10))


@njit(uint64(uint8), cache=True)
def mask_bishop_attacks(sq):
    tr, tf = divmod(sq, 8)
    attacks = 0
    directions = np.array([(1, 1), (-1, 1), (1, -1), (-1, -1)])
    for dr, df in directions:
        rank, file = tr + dr, tf + df
        while 0 < rank < 7 and 0 < file < 7:
            attacks |= BIT << (rank * 8 + file)
            rank, file = rank + dr, file + df
    return attacks


@njit(uint64(uint), cache=True)
def mask_king_attacks(sq):
    bb = set_bit(EMPTY, sq)
    return (((bb & ~FILES[a]) << 7) | (bb << 8) | ((bb & ~FILES[h]) << 9) | ((bb & ~FILES[h]) << 1) |
            ((bb & ~FILES[h]) >> 7) | (bb >> 8) | ((bb & ~FILES[a]) >> 9) | ((bb & ~FILES[a]) >> 1))


@njit(uint64(uint8), cache=True)
def mask_rook_attacks(sq):
    attacks = 0
    tr, tf = divmod(sq, 8)
    for rank in range(tr + 1, 7):
        attacks |= BIT << (rank * 8 + tf)
    for rank in range(tr - 1, 0, -1):
        attacks |= BIT << (rank * 8 + tf)
    for file in range(tf + 1, 7):
        attacks |= BIT << (tr * 8 + file)
    for file in range(tf - 1, 0, -1):
        attacks |= BIT << (tr * 8 + file)
    return attacks


@njit(uint64(uint8, uint64), cache=True)
def bishop_attacks_on_the_fly(sq, block):
    attacks = 0
    tr, tf = divmod(sq, 8)
    directions = np.array([(1, 1), (-1, 1), (1, -1), (-1, -1)])
    for direction in directions:
        for reach in range(1, 8):
            rank, file = tr + direction[0] * reach,  tf + direction[1] * reach
            if not 0 <= rank <= 7 or not 0 <= file <= 7:
                break
            attacked_bit = BIT << (rank * 8 + file)
            attacks |= attacked_bit
            if attacked_bit & block:
                break
    return attacks


@njit(uint64(uint8, uint64), cache=True)
def rook_attacks_on_the_fly(sq, block):
    attacks = 0
    tr, tf = divmod(sq, 8)
    for rank in range(tr + 1, 8):
        attacks |= BIT << (rank * 8 + tf)
        if (1 << (rank * 8 + tf)) & block:
            break
    for rank in range(tr - 1, -1, -1):
        attacks |= BIT << (rank * 8 + tf)
        if BIT << (rank * 8 + tf) & block:
            break
    for file in range(tf + 1, 8):
        attacks |= BIT << (tr * 8 + file)
        if BIT << (tr * 8 + file) & block:
            break
    for file in range(tf - 1, -1, -1):
        attacks |= BIT << (tr * 8 + file)
        if BIT << (tr * 8 + file) & block:
            break
    return attacks


@njit(uint64(uint16, uint8, uint64), cache=True)
def set_occ(index, bits_in_mask, attack_mask):
    occ = EMPTY
    for count in range(bits_in_mask):
        sq = lsb_index(attack_mask)
        attack_mask = pop_bit(attack_mask, sq)
        if index & (BIT << count):
            occ |= BIT << sq
    return occ


bishop_masks = np.fromiter((mask_bishop_attacks(i) for i in range(64)), dtype=np.uint64)
rook_masks = np.fromiter((mask_rook_attacks(i) for i in range(64)), dtype=np.uint64)


@njit(uint64[:, :](uint64[:, :], b1), cache=True)
def init_sliders(attacks, is_bishop):
    for i in range(64):
        attacks_mask = bishop_masks[i] if is_bishop else rook_masks[i]
        relevant_bits_count = count_bits(attacks_mask)
        occ_indices = (1 << relevant_bits_count)
        for index in range(occ_indices):
            if is_bishop:
                occ = set_occ(index, relevant_bits_count, attacks_mask)
                magic_index = (occ * bishop_magic_numbers[i]) >> (64 - bishop_relevant_bits[i])
                attacks[i][magic_index] = bishop_attacks_on_the_fly(i, occ)
            else:
                occ = set_occ(index, relevant_bits_count, attacks_mask)
                magic_index = (occ * rook_magic_numbers[i]) >> (64 - rook_relevant_bits[i])
                attacks[i][magic_index] = rook_attacks_on_the_fly(i, occ)
    return attacks


bishop_attacks = init_sliders(np.empty((64, 512), dtype=np.uint64), is_bishop=True)
rook_attacks = init_sliders(np.empty((64, 4096), dtype=np.uint64), is_bishop=False)
pawn_attacks = np.fromiter((mask_pawn_attacks(color, i) for color in range(2) for i in range(64)), dtype=np.uint64)
pawn_attacks.shape = (2, 64)
knight_attacks = np.fromiter((mask_knight_attacks(i) for i in range(64)), dtype=np.uint64)
king_attacks = np.fromiter((mask_king_attacks(i) for i in range(64)), dtype=np.uint64)

# Save the arrays to files
np.save('data/bishop_attacks.npy', bishop_attacks)
np.save('data/rook_attacks.npy', rook_attacks)
np.save('data/pawn_attacks.npy', pawn_attacks)
np.save('data/knight_attacks.npy', knight_attacks)
np.save('data/king_attacks.npy', king_attacks)
np.save('data/bishop_masks.npy', bishop_masks)
np.save('data/rook_masks.npy', rook_masks)
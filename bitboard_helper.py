from numba import njit, b1, uint64, uint8
from stored import lsb_lookup
from const import de_bruijn


@njit(b1(uint64, uint8), cache=True)
def get_bit(bitboard, square) -> b1:
    return bitboard & (1 << square)


@njit(uint64(uint64, uint8), cache=True)
def set_bit(bitboard, square) -> uint64:
    return bitboard | (1 << square)


@njit(uint64(uint64, uint8), cache=True)
def pop_bit(bitboard, square) -> uint8:
    return bitboard & ~(1 << square)


@njit(uint8(uint64), cache=True)
def count_bits(bb):
    count = 0
    while bb:
        count += 1
        bb &= bb - uint64(1)
    return count


@njit(uint8(uint64), cache=True)
def lsb_index(bitboard):
    return lsb_lookup[((bitboard & -bitboard) * de_bruijn) >> uint64(58)]


@njit(cache=True)
def i_to_rc(idx):
    return int(idx//8), int(idx % 8)


@njit(cache=True)
def rc_to_i(row, col):
    return row*8+col

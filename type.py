from numba import uint8, uint64, types, b1, from_dtype, uint32, int64
import numpy as np

position_type = [
    ("pieces", uint64[:, :]),
    ("occupancies", uint64[:]),
    ("side", uint8),
    ("eps", uint8),
    ("castle", uint8),
    ("hash_key", uint64),
    ("state", uint8),
    ("repetitions", types.DictType(uint64, int64)),
    ("half_move_clock", int64),
    ("full_move_count", int64)
]

move_type = [
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


hash_numpy_type = np.dtype(
    [("key", np.uint64),  # unique position
     ("depth", np.uint8),  # current depth
     ("flag", np.uint8),  # fail low, fail high, PV
     ("score", np.int64)]  # alpha, beta, PV
)

hash_numba_type = from_dtype(hash_numpy_type)

ai_type = [
    ("nodes", uint64),   # calculate total position
    ("ply", uint32),     # depth
    ("killer_moves", uint64[:, :]),  # 2 x 64    => move that lead to beta cut off
    ("history_moves", uint8[:, :, :]),    # 2 x 6 x 64    => best quite move
    ("pv_table", uint64[:, :]),  # principal variation search
    ("pv_length", uint64[:]),
    ("follow_pv", b1),
    ("score_pv", b1),
    ("hash_table", hash_numba_type[:]),
    ("time_limit", uint64),
    ("node_limit", uint64),
    ("start", uint64),
    ("stopped", b1),
]

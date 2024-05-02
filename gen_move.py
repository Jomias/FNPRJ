import numba as nb
import numpy as np
from position import Position, side_key, piece_keys, en_passant_keys, castle_keys, print_board
from move import Move, encode_move, print_move_list
from numba import njit, typed
from bitboard_helper import get_bit, set_bit, pop_bit, lsb_index
from get_attack import pawn_attacks, is_sq_attacked, get_attacks
from const import WHITE, BLACK, BOTH, PAWN, KNIGHT, ROOK, KING, a1, a2, a7, a8, h1, h2, h7, h8, b1, c1, d1, e1, f1, g1, \
    b8, c8, d8, e8, f8, g8, EMPTY, BIT, wk, wq, bk, bq, no_sq, DRAW
from stored import castling_rights
from fen import parse_fen, killer_position


def random_move(pos):
    legal_moves = generate_legal_moves(pos)
    legal_moves_array = np.array(legal_moves)
    return legal_moves_array[np.random.choice(len(legal_moves_array))]


@njit
def generate_legal_moves(cur_pos: Position, only_captures=False):
    move_list = typed.List()
    for piece in range(6):
        bb, opp = cur_pos.pieces[cur_pos.side][piece], cur_pos.side ^ 1
        if cur_pos.side == WHITE:
            if piece == PAWN:
                while bb:
                    source = lsb_index(bb)
                    target = source - 8
                    if not only_captures:
                        if not target < a8 and not get_bit(cur_pos.occupancies[BOTH], target):  # can go forward
                            if a7 <= source <= h7:  # promotion
                                for i in range(KNIGHT, KING):
                                    m = Move(encode_move(source, target, piece, cur_pos.side, i))
                                    if is_legal_move(cur_pos, m):
                                        move_list.append(m)
                            else:
                                m = Move(encode_move(source, target, piece, cur_pos.side))
                                if is_legal_move(cur_pos, m):
                                    move_list.append(m)
                                if a2 <= source <= h2 and not get_bit(cur_pos.occupancies[BOTH], target - 8):  # double push
                                    double_push_move = Move(encode_move(source, target - 8, piece, cur_pos.side, double_push=True))
                                    if is_legal_move(cur_pos, double_push_move):
                                        move_list.append(double_push_move)
                    attacks = nb.uint64(pawn_attacks[WHITE][source] & cur_pos.occupancies[BLACK])
                    while attacks:
                        cur_target = lsb_index(attacks)
                        if a7 <= source <= h7:  # promotion
                            for i in range(KNIGHT, KING):
                                m = Move(encode_move(source, cur_target, piece, cur_pos.side, promote_to=i, capture=True))
                                if is_legal_move(cur_pos, m):
                                    move_list.append(m)
                        else:
                            m = Move(encode_move(source, cur_target, piece, cur_pos.side, capture=True))
                            if is_legal_move(cur_pos, m):
                                move_list.append(m)
                        attacks = pop_bit(attacks, cur_target)

                    if cur_pos.eps != no_sq:
                        en_ps_at = pawn_attacks[WHITE][source] & (BIT << cur_pos.eps)
                        if en_ps_at:
                            target_en_ps = lsb_index(en_ps_at)
                            m = Move(encode_move(source, target_en_ps, piece, cur_pos.side, capture=True, enpassant=True))
                            if is_legal_move(cur_pos, m):
                                move_list.append(m)
                    bb = pop_bit(bb, source)
            if not only_captures and piece == KING and not is_sq_attacked(cur_pos, e1):
                if (cur_pos.castle & wk and not get_bit(cur_pos.occupancies[BOTH], f1) and not get_bit(cur_pos.occupancies[BOTH], g1)
                        and not is_sq_attacked(cur_pos, f1) and not is_sq_attacked(cur_pos, g1)):
                    m = Move(encode_move(e1, g1, piece, cur_pos.side, castling=True))
                    if is_legal_move(cur_pos, m):
                        move_list.append(m)
                if (cur_pos.castle & wq and not is_sq_attacked(cur_pos, e1) and not get_bit(cur_pos.occupancies[BOTH], d1)
                        and not get_bit(cur_pos.occupancies[BOTH], c1) and not get_bit(cur_pos.occupancies[BOTH], b1)
                        and not is_sq_attacked(cur_pos, d1) and not is_sq_attacked(cur_pos, c1)):
                    m = Move(encode_move(e1, c1, piece, cur_pos.side, castling=True))
                    if is_legal_move(cur_pos, m):
                        move_list.append(m)
        if cur_pos.side == BLACK:
            if piece == PAWN:
                while bb:
                    source = lsb_index(bb)
                    target = source + 8
                    if not only_captures:
                        if not target > h1 and not get_bit(cur_pos.occupancies[BOTH], target):  # can go forward
                            if a2 <= source <= h2:  # promotion
                                for i in range(KNIGHT, KING):
                                    m = Move(encode_move(source, target, piece, cur_pos.side, i))
                                    if is_legal_move(cur_pos, m):
                                        move_list.append(m)
                            else:
                                m = Move(encode_move(source, target, piece, cur_pos.side))
                                if is_legal_move(cur_pos, m):
                                    move_list.append(m)
                                if a7 <= source <= h7 and not get_bit(cur_pos.occupancies[BOTH], target + 8):  # double push
                                    double_push_move = Move(encode_move(source, target + 8, piece, cur_pos.side, double_push=True))
                                    if is_legal_move(cur_pos, double_push_move):
                                        move_list.append(double_push_move)
                    attacks = pawn_attacks[BLACK][source] & cur_pos.occupancies[WHITE]
                    while attacks:
                        cur_target = lsb_index(attacks)
                        if a2 <= source <= h2:  # promotion
                            for i in range(KNIGHT, KING):
                                m = Move(encode_move(source, cur_target, piece, cur_pos.side, promote_to=i, capture=True))
                                if is_legal_move(cur_pos, m):
                                    move_list.append(m)
                        else:
                            m = Move(encode_move(source, cur_target, piece, cur_pos.side, capture=True))
                            if is_legal_move(cur_pos, m):
                                move_list.append(m)
                        attacks = pop_bit(attacks, cur_target)
                    if cur_pos.eps != no_sq:
                        en_ps_at = pawn_attacks[BLACK][source] & (BIT << cur_pos.eps)
                        if en_ps_at:
                            target_en_ps = lsb_index(en_ps_at)
                            m = Move(encode_move(source, target_en_ps, piece, cur_pos.side, capture=True, enpassant=True))
                            if is_legal_move(cur_pos, m):
                                move_list.append(m)
                    bb = pop_bit(bb, source)
            if not only_captures and piece == KING and not is_sq_attacked(cur_pos, e8):
                if (cur_pos.castle & bk and not get_bit(cur_pos.occupancies[BOTH], f8) and not get_bit(cur_pos.occupancies[BOTH], g8)
                        and not is_sq_attacked(cur_pos, f8) and not is_sq_attacked(cur_pos, g8)):
                    m = Move(encode_move(e8, g8, piece, cur_pos.side, castling=True))
                    if is_legal_move(cur_pos, m):
                        move_list.append(m)
                if (cur_pos.castle & bq and not get_bit(cur_pos.occupancies[BOTH], d8) and not get_bit(cur_pos.occupancies[BOTH], c8)
                        and not get_bit(cur_pos.occupancies[BOTH], b8) and not is_sq_attacked(cur_pos, d8)
                        and not is_sq_attacked(cur_pos, c8)):
                    m = Move(encode_move(e8, c8, piece, cur_pos.side, castling=True))
                    if is_legal_move(cur_pos, m):
                        move_list.append(m)
        if piece in range(KNIGHT, KING + 1):
            while bb:
                source = lsb_index(bb)
                attacks = get_attacks(piece, source, cur_pos)
                while attacks != EMPTY:
                    cur_target = lsb_index(attacks)
                    is_capture = get_bit(cur_pos.occupancies[opp], cur_target)
                    m = Move(encode_move(source, cur_target, piece, cur_pos.side, capture=is_capture))
                    if is_legal_move(cur_pos, m) and ((not only_captures) or (only_captures and is_capture)):
                        move_list.append(m)
                    attacks = pop_bit(attacks, cur_target)
                bb = pop_bit(bb, source)
    return move_list


@njit(nb.b1(Position.class_type.instance_type))
def is_king_in_check(cur_pos):
    return is_sq_attacked(cur_pos, lsb_index(cur_pos.pieces[cur_pos.side][KING]))


@njit(nb.b1(Position.class_type.instance_type, Move.class_type.instance_type))
def is_legal_move(old_pos: Position, cur_move: Move):
    new_pos = Position()
    new_pos.pieces = np.copy(old_pos.pieces)
    new_pos.side = old_pos.side
    opp, side = new_pos.side ^ 1, new_pos.side
    new_pos.pieces[side][cur_move.piece] = pop_bit(new_pos.pieces[side][cur_move.piece], cur_move.source)
    new_pos.pieces[side][cur_move.piece] = set_bit(new_pos.pieces[side][cur_move.piece], cur_move.target)
    if cur_move.capture:        # Piece give u check and capture it
        for piece in range(PAWN, KING):
            if get_bit(new_pos.pieces[opp][piece], cur_move.target):
                new_pos.pieces[opp][piece] = pop_bit(new_pos.pieces[opp][piece], cur_move.target)
                break
    if cur_move.enpassant:      # enpassant could make the king in check
        delta = 8 if new_pos.side == WHITE else - 8
        new_pos.pieces[opp][PAWN] = pop_bit(new_pos.pieces[opp][PAWN], cur_move.target + delta)
    for color in range(2):
        for bb in new_pos.pieces[color]:
            new_pos.occupancies[color] |= bb
        new_pos.occupancies[BOTH] |= new_pos.occupancies[color]
    return not is_king_in_check(new_pos)


@njit(Position.class_type.instance_type(Position.class_type.instance_type, Move.class_type.instance_type))
def apply_move(old_pos: Position, cur_move: Move):
    new_pos = Position()
    new_pos.pieces = np.copy(old_pos.pieces)
    new_pos.repetitions = old_pos.repetitions.copy()
    new_pos.fifty_moves_cnt = old_pos.fifty_moves_cnt + 1
    new_pos.side, new_pos.eps, new_pos.castle, new_pos.hash_key = old_pos.side, old_pos.eps, old_pos.castle, old_pos.hash_key
    opp = new_pos.side ^ 1
    side, piece, source, target, capture = int(cur_move.side), cur_move.piece, cur_move.source, cur_move.target, cur_move.capture
    promote_to, enpassant, double_push, castling = cur_move.promote_to, cur_move.enpassant, cur_move.double_push, cur_move.castling
    delta = 8 if new_pos.side == WHITE else -8
    new_pos.pieces[side][piece] = pop_bit(new_pos.pieces[side][piece], source)
    new_pos.pieces[side][piece] = set_bit(new_pos.pieces[side][piece], target)
    new_pos.hash_key ^= piece_keys[side][piece][source] ^ piece_keys[side][piece][target]
    if capture:
        new_pos.fifty_moves_cnt = 0
        for p in range(PAWN, KING):
            if get_bit(new_pos.pieces[opp][p], target):
                new_pos.pieces[opp][p] = pop_bit(new_pos.pieces[opp][p], target)
                new_pos.hash_key ^= piece_keys[opp][p][target]
                break
    if promote_to:
        new_pos.pieces[side][piece] = pop_bit(new_pos.pieces[side][piece], target)
        new_pos.pieces[side][promote_to] = set_bit(new_pos.pieces[side][promote_to], target)
        new_pos.hash_key ^= piece_keys[side][piece][target] ^ piece_keys[side][promote_to][target]
    if enpassant:
        new_pos.pieces[opp][PAWN] = pop_bit(new_pos.pieces[opp][PAWN], target + delta)
        new_pos.hash_key ^= piece_keys[opp][PAWN][target + delta]
    new_pos.hash_key ^= en_passant_keys[new_pos.eps]
    new_pos.eps = target + delta if double_push else no_sq
    new_pos.hash_key ^= en_passant_keys[new_pos.eps]
    if castling:
        rook_moves = {g1: (h1, f1), c1: (a1, d1), g8: (h8, f8), c8: (a8, d8)}
        rook_source, rook_target = rook_moves[target]
        new_pos.pieces[side][ROOK] = pop_bit(new_pos.pieces[side][ROOK], rook_source)
        new_pos.pieces[side][ROOK] = set_bit(new_pos.pieces[side][ROOK], rook_target)
        new_pos.hash_key ^= piece_keys[side][ROOK][rook_source] ^ piece_keys[side][ROOK][rook_target]
    new_pos.hash_key ^= castle_keys[new_pos.castle]     # reset
    new_pos.castle &= castling_rights[source] & castling_rights[target]
    new_pos.hash_key ^= castle_keys[new_pos.castle]     # update
    for color in range(2):
        for bb in new_pos.pieces[color]:
            new_pos.occupancies[color] |= bb
        new_pos.occupancies[BOTH] |= new_pos.occupancies[color]
    new_pos.side = opp
    new_pos.hash_key ^= side_key
    if piece == PAWN:
        new_pos.fifty_moves_cnt = 0
    if new_pos.hash_key in new_pos.repetitions:
        new_pos.repetitions[new_pos.hash_key] += 1
    else:
        new_pos.repetitions[new_pos.hash_key] = 1
    if new_pos.repetitions[new_pos.hash_key] == 3 or new_pos.fifty_moves_cnt == 50:
        new_pos.state = DRAW
    return new_pos


@njit(Position.class_type.instance_type(Position.class_type.instance_type))
def make_null_move(old_pos: Position):
    pos = Position()
    pos.pieces, pos.occupancies = old_pos.pieces.copy(), old_pos.occupancies.copy()
    pos.castle, pos.hash_key, pos.eps = old_pos.castle, old_pos.hash_key, no_sq
    pos.side = old_pos.side ^ 1
    pos.hash_key ^= en_passant_keys[old_pos.eps] ^ en_passant_keys[no_sq] ^ side_key
    return pos


def parse_move(pos, uci_move: str) -> int:
    source = (ord(uci_move[0]) - ord('a')) + ((8 - int(uci_move[1])) * 8)
    target = (ord(uci_move[2]) - ord('a')) + ((8 - int(uci_move[3])) * 8)

    for move in generate_legal_moves(pos):
        if move.source == source and move.target == target:
            promoted_piece = move.promote_to
            if promoted_piece:
                for p, s in enumerate(('n', 'b', 'r', 'q'), 1):
                    if promoted_piece == p and uci_move[4] == s:
                        return move
                return 0
            return move
    return 0

if __name__ == "__main__":
    abc = parse_fen(killer_position)
    print_board(abc)
    print_move_list(generate_legal_moves(abc, False))

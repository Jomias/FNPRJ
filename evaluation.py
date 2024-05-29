from position import Position
import numba as nb
from bitboard_helper import *
from pst import *
from get_attack import get_bishop_attacks, get_queen_attacks, king_attacks


@njit(nb.uint16(Position.class_type.instance_type), cache=True)
def get_game_phase_score(pos):
    score = 0
    for color in (BLACK, WHITE):
        for piece in range(KNIGHT, QUEEN + 1):
            score = count_bits(pos.pieces[color][piece]) * material_score[opening][color][piece]
    return score


@njit(nb.int64(Position.class_type.instance_type))
def brute_force_evaluate(pos: Position) -> int:
    game_phase_score = get_game_phase_score(pos)
    if game_phase_score > opening:
        game_phase = opening
    elif game_phase_score < endgame_phase_score:
        game_phase = endgame
    else:
        game_phase = middle_game
    score, score_opening, score_endgame = 0, 0, 0
    for color in (WHITE, BLACK):
        color_temp = 1 if color == WHITE else -1
        for piece in range(6):
            bb = pos.pieces[color][piece]
            while bb:
                sq = lsb_index(bb)
                score_opening += material_score[opening][color][piece]
                score_endgame += material_score[endgame][color][piece]
                score_opening += -positional_score[opening][piece][mirror_pst[sq]] if color else positional_score[opening][piece][sq]
                score_endgame += -positional_score[endgame][piece][mirror_pst[sq]] if color else positional_score[endgame][piece][sq]
                if piece == PAWN:
                    double_pawns = count_bits(pos.pieces[color][PAWN] & file_masks[sq])
                    score_opening += max(double_pawns - 1, 0) * double_pawn_penalty_opening * color_temp
                    score_endgame += max(double_pawns - 1, 0) * double_pawn_penalty_endgame * color_temp
                    if (pos.pieces[color][PAWN] & isolated_masks[sq]) == 0:
                        score_opening += isolated_pawn_penalty_opening * color_temp
                        score_endgame += isolated_pawn_penalty_endgame * color_temp
                    if color == WHITE and (white_passed_masks[sq] & pos.pieces[BLACK][PAWN]) == 0:
                        score_opening += passed_pawn_bonus[get_rank[sq]]
                        score_endgame += passed_pawn_bonus[get_rank[sq]]
                    if color == BLACK and (black_passed_masks[sq] & pos.pieces[WHITE][PAWN]) == 0:
                        score_opening -= passed_pawn_bonus[get_rank[sq]]
                        score_endgame -= passed_pawn_bonus[get_rank[sq]]
                elif piece == KNIGHT:
                    pass
                elif piece == BISHOP:
                    bishop_counts = count_bits(get_bishop_attacks(sq, pos.occupancies[BOTH]))
                    score_opening += (bishop_counts - bishop_unit) * bishop_mobility_opening * color_temp
                    score_endgame += (bishop_counts - bishop_unit) * bishop_mobility_endgame * color_temp
                elif piece == ROOK:
                    if (pos.pieces[color][PAWN] & file_masks[sq]) == 0:
                        score_opening += semi_open_file_score * color_temp
                        score_endgame += semi_open_file_score * color_temp
                    if ((pos.pieces[BLACK][PAWN] | pos.pieces[WHITE][PAWN]) & file_masks[sq]) == 0:
                        score_opening += open_file_score * color_temp
                        score_endgame += open_file_score * color_temp
                elif piece == QUEEN:
                    queen_counts = count_bits(get_queen_attacks(sq, pos.occupancies[BOTH]))
                    score_opening += (queen_counts - queen_unit) * queen_mobility_opening * color_temp
                    score_endgame += (queen_counts - queen_unit) * queen_mobility_endgame * color_temp
                elif piece == KING:
                    if (pos.pieces[color][PAWN] & file_masks[sq]) == 0:
                        score_opening -= semi_open_file_score * color_temp
                        score_endgame -= semi_open_file_score * color_temp
                    if ((pos.pieces[BLACK][PAWN] | pos.pieces[WHITE][PAWN]) & file_masks[sq]) == 0:
                        score_opening -= open_file_score * color_temp
                        score_endgame -= open_file_score * color_temp
                    king_shield_counts = count_bits(king_attacks[sq] & pos.occupancies[color])
                    score_opening += king_shield_counts * king_shield_bonus * color_temp
                    score_endgame += king_shield_counts * king_shield_bonus * color_temp
                bb = pop_bit(bb, sq)

    if game_phase == middle_game:
        score = (score_opening * game_phase_score + score_endgame * (opening_phase_score - game_phase_score)) // opening_phase_score
    elif game_phase == opening:
        score = score_opening
    elif game_phase == endgame:
        score = score_endgame
    return score if pos.side == WHITE else -score

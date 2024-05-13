import numba as nb
from numba.experimental import jitclass
from numba import njit
from type import ai_type, hash_numpy_type
import numpy as np
from const import MAX_PLY, MAX_HASH_SIZE, LOWER_MATE, hash_flag_alpha, hash_flag_exact, hash_flag_beta, no_hash_entry, \
    PAWN, DRAW, STALEMATE, full_depth_moves, reduction_limit, time_precision, BOUND_INF, UPPER_MATE, WHITE
from position import Position
from gen_move import apply_move, generate_legal_moves, make_null_move, is_king_in_check
from bitboard_helper import get_bit
from move import Move, get_move_uci
from stored import MVV_LVA
import time
from train import predict_evaluation, load_model

nn = load_model("adam_model_0.pth")

@njit(nb.int64(Position.class_type.instance_type))
def evaluate(pos: Position):
    with nb.objmode(val='int64'):
        val = predict_evaluation(nn, pos.extract_bit_vector())
        val = int(val * (100 if pos.side == WHITE else -100))
        return val

@jitclass(ai_type)
class Stupid:
    def __init__(self):
        self.nodes, self.ply = 0, 0
        # Killer moves [id][ply]
        self.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
        # History moves [side][piece][square]
        self.history_moves = np.zeros((2, 6, 64), dtype=np.uint8)
        # PV
        self.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
        self.pv_length = np.zeros(MAX_PLY, dtype=np.uint64)
        self.follow_pv, self.score_pv = False, False
        # Transposition Table
        self.hash_table = np.zeros(MAX_HASH_SIZE, dtype=hash_numpy_type)
        self.time_limit = 10000
        self.node_limit = 10**7
        self.start = 0
        self.stopped = True

    def reset_bot(self, time_limit, node_limit):
        self.nodes, self.score_pv, self.follow_pv = 0, False, False
        self.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
        self.history_moves = np.zeros((2, 6, 64), dtype=np.uint8)
        self.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
        self.pv_length = np.zeros(MAX_PLY, dtype=np.uint64)
        self.nodes = 0
        self.stopped = False
        self.time_limit = time_limit
        self.node_limit = node_limit
        with nb.objmode(start=nb.uint64):
            start = time.time() * 1000
        self.start = start

    def read_hash_entry(self, pos, depth, alpha, beta):
        entry = self.hash_table[pos.hash_key % MAX_HASH_SIZE]
        if entry.key == pos.hash_key:
            if entry.depth >= depth:
                score = entry.score
                if score < -LOWER_MATE:
                    score += self.ply
                elif score > LOWER_MATE:
                    score -= self.ply
                if entry.flag == hash_flag_exact:
                    return score
                if entry.flag == hash_flag_alpha and entry.score <= alpha:
                    return alpha
                if entry.flag == hash_flag_beta and entry.score >= beta:
                    return beta
        return no_hash_entry

    def write_hash_entry(self, pos, score, depth, hash_flag):
        i = pos.hash_key % MAX_HASH_SIZE
        if score < -LOWER_MATE:
            score -= self.ply
        elif score > LOWER_MATE:
            score += self.ply
        self.hash_table[i].key = pos.hash_key
        self.hash_table[i].depth = depth
        self.hash_table[i].flag = hash_flag
        self.hash_table[i].score = score

    def communicate(self):
        with nb.objmode(spent=nb.uint64):
            spent = time.time() * 1000 - self.start
        if spent > self.time_limit or self.nodes > self.node_limit:
            self.stopped = True


@njit(nb.uint64(Stupid.class_type.instance_type, Position.class_type.instance_type, Move.class_type.instance_type), cache=True)
def score_move(bot, pos, move) -> int:
    if bot.score_pv:
        if bot.pv_table[0][bot.ply] == move.encode:
            bot.score_pv = False
            return 20000    # using the pv move first
    if move.capture:
        attacker, victim_sq = move.piece, move.target
        victim = PAWN   # enps
        for piece, bb in enumerate(pos.pieces[pos.side ^ 1]):
            if get_bit(bb, victim_sq):
                victim = piece
                break
        return MVV_LVA[attacker][victim] + 10000
    else:
        if bot.killer_moves[0][bot.ply] == move.encode:
            return 9000
        elif bot.killer_moves[1][bot.ply] == move.encode:
            return 8000
        else:
            return bot.history_moves[int(move.side)][move.piece][move.target]


@njit
def quiescence_search(bot, pos, alpha, beta):
    if not bot.nodes & time_precision:
        bot.communicate()
    bot.nodes += 1
    if pos.state == DRAW or pos.state == STALEMATE:
        return 0
    evaluation = evaluate(pos)
    if bot.ply > MAX_PLY - 1:  # limit it because search might take too long
        return evaluation
    if evaluation >= beta:
        return beta
    if evaluation > alpha:
        alpha = evaluation

    moves = generate_legal_moves(pos, only_captures=True)
    move_list = [(move, score_move(bot, pos, move)) for move in moves]
    move_list.sort(reverse=True, key=lambda x: x[1])
    for move, _ in move_list:
        new_pos = apply_move(pos, move)
        bot.ply += 1
        score = -quiescence_search(bot, new_pos, -beta, -alpha)     # score of current move
        bot.ply -= 1
        if bot.stopped:
            return 0
        if score > alpha:
            alpha = score
            if score >= beta:
                return beta
    return alpha

@njit
def negamax(bot: Stupid, pos: Position, alpha, beta, depth):
    if pos.state == DRAW or pos.state == STALEMATE:
        return 0
    hash_flag = hash_flag_alpha
    hash_entry = bot.read_hash_entry(pos, depth, alpha, beta)
    pv_node = beta - alpha > 1
    if bot.ply and hash_entry != no_hash_entry and not pv_node:     # This position already been search
        return hash_entry
    if not bot.nodes & time_precision:  # each 2047 node
        bot.communicate()
    bot.pv_length[bot.ply] = bot.ply
    if depth == 0:
        return quiescence_search(bot, pos, alpha, beta)
    if bot.ply > MAX_PLY - 1:
        return evaluate(pos)
    bot.nodes += 1

    in_check = is_king_in_check(pos)
    if in_check:
        depth += 1  # if checking then adding 1 more depth

    if depth >= 3 and not in_check and bot.ply:     # null move pruning
        pruning_pos = make_null_move(pos)
        bot.ply += 1
        score = -negamax(bot, pruning_pos, -beta, -beta + 1, depth - 1 - 2)     # 2 is reduction limit, find beta cutoff
        bot.ply -= 1
        if bot.stopped:
            return 0
        if score >= beta:
            return beta

    moves = generate_legal_moves(pos)
    if bot.follow_pv:
        bot.follow_pv = False
        move = Move(bot.pv_table[0][bot.ply])
        if move in moves:
            bot.score_pv, bot.follow_pv = True, True
    move_list = [(move, score_move(bot, pos, move)) for move in moves]
    move_list.sort(reverse=True, key=lambda x: x[1])
    moves_searched = 0
    for move, _ in move_list:
        new_pos = apply_move(pos, move)
        bot.ply += 1
        if moves_searched == 0:   # LMR
            score = -negamax(bot, new_pos, -beta, -alpha, depth - 1)
        else:
            if moves_searched >= full_depth_moves and depth >= reduction_limit and not in_check\
                    and not move.capture and move.promote_to == 0 and not is_king_in_check(new_pos):
                score = -negamax(bot, new_pos, -alpha - 1, -alpha, depth - 2)    # reduce depth
            else:
                score = alpha + 1
            if score > alpha:   # pvs
                score = -negamax(bot, new_pos, -alpha - 1, -alpha, depth - 1)
                if alpha < score < beta:    # really good
                    score = -negamax(bot, new_pos, -beta, -alpha, depth - 1)
        bot.ply -= 1
        if bot.stopped:
            return 0
        moves_searched += 1
        if score > alpha:
            hash_flag = hash_flag_exact
            if not move.capture:     # best quite move
                bot.history_moves[pos.side][move.piece][move.target] += depth
            alpha = score
            bot.pv_table[bot.ply][bot.ply] = move.encode
            for next_ply in range(bot.ply + 1, bot.pv_length[bot.ply + 1]):
                bot.pv_table[bot.ply][next_ply] = bot.pv_table[bot.ply + 1][next_ply]
            bot.pv_length[bot.ply] = bot.pv_length[bot.ply + 1]
            # beta cut
            if score >= beta:
                bot.write_hash_entry(pos, beta, depth, hash_flag_beta)
                if not move.capture:  # stored killer which is the move lead to beta cut off
                    bot.killer_moves[1][bot.ply] = bot.killer_moves[0][bot.ply]
                    bot.killer_moves[0][bot.ply] = move.encode
                return beta
    if len(moves) == 0:     # Check mate
        if in_check:    # assuming to find the closet mate
            return -UPPER_MATE + bot.ply
        else:
            return 0

    bot.write_hash_entry(pos, alpha, depth, hash_flag)
    return alpha

@njit
def search(bot, pos, print_info=False, depth_limit=32, time_limit=100000, node_limit=10**9):
    bot.reset_bot(time_limit, node_limit)
    depth, value = 0, 0
    alpha, beta = -BOUND_INF, BOUND_INF
    for depth in range(1, depth_limit + 1):
        if bot.stopped or not -LOWER_MATE < value < LOWER_MATE:
            break
        bot.follow_pv = True
        value = negamax(bot, pos, alpha, beta, depth)
        if value <= alpha or value >= beta:
            alpha, beta = -BOUND_INF, BOUND_INF
            continue
        alpha, beta = value - 50, value + 50
        if print_info and bot.pv_table[0][0]:
            pv_line = " ".join([get_move_uci(Move(bot.pv_table[0][c])) for c in range(bot.pv_length[0])])
            s_score = "mate"
            if -UPPER_MATE < value < -LOWER_MATE:
                score = -(value + UPPER_MATE) // 2
            elif LOWER_MATE < value < UPPER_MATE:
                score = (UPPER_MATE - value) // 2 + 1
            else:
                s_score = "cp"
                score = value
            print("info", "depth", depth, "score", s_score, int(score), "nodes", bot.nodes, "pv", pv_line)
    if print_info:
        print("best move: " + get_move_uci(Move(bot.pv_table[0][0])))


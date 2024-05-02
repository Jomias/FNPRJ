from fen import start_position, parse_fen, tricky_position, killer_position
from perf import uci_perf


board_1 = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
board_2 = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
board_3 = parse_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
board_4 = parse_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")

uci_perf(board_1, 4)
uci_perf(board_2, 4)
uci_perf(board_3, 4)
uci_perf(board_4, 4)

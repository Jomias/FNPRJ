from fen import start_position, parse_fen, tricky_position, killer_position
from perf import uci_perf
from search import Stupid, search
import unittest


class TestPerft(unittest.TestCase):
    def test_init(self):
        board = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        results = [1, 20, 400, 8902, 197281]
        for i in range(len(results)):
            self.assertEqual(uci_perf(board, i), results[i])

    def test_pos2(self):
        board = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        results = [1, 48, 2039, 97862, 4085603]
        for i in range(len(results)):
            self.assertEqual(uci_perf(board, i), results[i])

    def test_pos3(self):
        board = parse_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
        results = [1, 14, 191, 2812, 43238, 674624]
        for i in range(len(results)):
            self.assertEqual(uci_perf(board, i), results[i])

    def test_pos4(self):
        board = parse_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")
        results = [1, 44, 1486, 62379, 2103487]
        for i in range(len(results)):
            self.assertEqual(uci_perf(board, i), results[i])


if __name__ == "__main__":
    unittest.main()
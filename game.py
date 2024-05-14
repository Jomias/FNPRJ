import math
import sys
import numba as nb
import pygame
from tkinter import messagebox, Tk
from gui import DisplayBoard, GUI, MainMenu
from search import Stupid, search
from const import (USER, WHITE, EASY, GAME_WIDTH, GAME_HEIGHT, DRAW, CHECKMATE, STALEMATE, RESIGN, BOARD_WIDTH, MEDIUM,
                   HARD, BLACK, reversed_board, SQUARE_SIZE, NORMAL, COMPUTER)
from bitboard_helper import i_to_rc
from gen_move import apply_move, random_move
from move import Move
from old_train import load_model


def get_board_pos(pos):
    return math.floor(pos[1] / SQUARE_SIZE) * 8 + math.floor(pos[0] / SQUARE_SIZE)


class Controller:
    def __init__(self):
        self.game_board = DisplayBoard()
        self.ai = Stupid()
        # search(self.ai, self.game_board.pos, time_limit=3000)
        self.game_type = USER
        self.selected_square = None
        self.user_color = WHITE
        self.difficulty = EASY
        self.screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
        self.move_count = 0
        pygame.display.set_caption('CHESS GAME')
        self.gui = GUI(self.screen)

    def play(self):
        self.selected_square = None
        self.move_count = 0
        self.game_board.reset_board()
        if self.game_type == USER:
            self.play_user()
        elif self.game_type == COMPUTER:
            self.play_computer()
        else:
            self.training()

    def training(self):
        pass

    def play_user(self):
        self.gui.draw_game(self.game_board, self.selected_square)
        pygame.display.update()
        while True:
            if self.game_board.pos.state != NORMAL:
                window = Tk()
                window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
                window.withdraw()
                if self.game_board.pos.state == DRAW or self.game_board.pos.state == STALEMATE:
                    messagebox.showinfo("Game ended", "Draw !")
                if self.game_board.pos.state == CHECKMATE:
                    if self.game_board.pos.side == WHITE:
                        messagebox.showinfo("Game ended", "Black Win !")
                    else:
                        messagebox.showinfo("Game ended", "White Win !")
                if self.game_board.pos.state == RESIGN:
                    if self.game_board.pos.side == WHITE:
                        messagebox.showinfo("Game ended", "White Resigned! Black Win!")
                    else:
                        messagebox.showinfo("Game ended", "Black Resigned! White Win!")
                window.deiconify()
                window.destroy()
                window.quit()
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game_board.pos.state = RESIGN
                        break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if pos[0] > BOARD_WIDTH:
                        continue
                    pos = get_board_pos(pos)
                    if self.selected_square is None:
                        _x, _y = i_to_rc(pos)
                        if self.game_board.ui_board[_x][_y] is not None:
                            self.selected_square = pos
                    else:
                        target = pos
                        if nb.uint64(1 << target) & self.game_board.pos.occupancies[self.game_board.pos.side]:
                            self.selected_square = target
                            continue
                        if self.selected_square in self.game_board.moves and target in self.game_board.moves[self.selected_square]:
                            move = self.game_board.moves[self.selected_square][target]
                            self.game_board.pos = apply_move(self.game_board.pos, move)
                            self.game_board.last_move = (self.selected_square, target)
                            self.game_board.update_board()
                        self.selected_square = None
                self.gui.draw_game(self.game_board, self.selected_square)
                pygame.display.update()

    def play_computer(self):
        self.gui.draw_game(self.game_board, self.selected_square, user_side=self.user_color)
        pygame.display.update()
        find_move = self.user_color != WHITE
        while True:
            if self.game_board.pos.state != NORMAL:
                window = Tk()
                window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
                window.withdraw()
                if self.game_board.pos.state == DRAW or self.game_board.pos.state == STALEMATE:
                    messagebox.showinfo("Game ended", "Draw !")
                if self.game_board.pos.state == CHECKMATE:
                    if self.game_board.pos.side == self.user_color:
                        messagebox.showinfo("Game ended", "You Lose !")
                    else:
                        messagebox.showinfo("Game ended", "You Win !")
                if self.game_board.pos.state == RESIGN:
                    messagebox.showinfo("Game ended", "You Lose !")
                window.deiconify()
                window.destroy()
                window.quit()
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game_board.pos.state = RESIGN
                        break
                elif self.game_board.pos.side != self.user_color and find_move:
                    if self.difficulty == MEDIUM:
                        search(self.ai, self.game_board.pos, print_info=True, node_limit=2000)
                    elif self.difficulty == HARD:
                        search(self.ai, self.game_board.pos, print_info=True, depth_limit=7)
                    move = random_move(self.game_board.pos) if self.difficulty == EASY else Move(self.ai.pv_table[0][0])
                    self.game_board.pos = apply_move(self.game_board.pos, move)
                    self.game_board.last_move = (move.source, move.target)
                    self.game_board.update_board()
                    find_move = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not find_move:
                    pos = pygame.mouse.get_pos()
                    if pos[0] > BOARD_WIDTH:
                        continue
                    pos = get_board_pos(pos)
                    if self.user_color == BLACK:
                        pos = reversed_board[pos]
                    if self.selected_square is None:
                        _x, _y = i_to_rc(pos)
                        if self.game_board.ui_board[_x][_y] is not None:
                            self.selected_square = pos
                    else:
                        target = pos
                        if nb.uint64(1 << target) & self.game_board.pos.occupancies[self.game_board.pos.side]:
                            self.selected_square = target
                            continue
                        if self.selected_square in self.game_board.moves and target in self.game_board.moves[self.selected_square]:
                            move = self.game_board.moves[self.selected_square][target]
                            self.game_board.pos = apply_move(self.game_board.pos, move)
                            self.game_board.last_move = (self.selected_square, target)
                            self.game_board.update_board()
                            find_move = True
                        self.selected_square = None
                self.gui.draw_game(self.game_board, self.selected_square, user_side=self.user_color)
                pygame.display.update()


class Game:
    def __init__(self):
        pygame.init()
        controller = Controller()
        cur_menu = MainMenu(controller)
        cur_menu.main_loop()

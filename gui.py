import pygame_menu
import pygame
from pygame_menu import themes
from const import (WHITE, BLACK, GAME_WIDTH, GAME_HEIGHT, USER, COMPUTER, WHITE_COLOR, DARK_COLOR, ROWS, COLS, PAWN,
                   KNIGHT, BISHOP, ROOK, QUEEN, KING, SQUARE_SIZE, SQUARE_MARK, LAST_MOVE_COLOR, reversed_board, RED,
                   GREEN, BLUE, CHECKMATE, STALEMATE, NORMAL)
from bitboard_helper import (i_to_rc, rc_to_i, pop_bit, lsb_index)
from fen import parse_fen, start_position
from gen_move import generate_legal_moves, is_king_in_check


class DisplayBoard:
    def __init__(self):
        self.pos = parse_fen(start_position)
        self.ui_board = [[None for _ in range(8)] for _ in range(8)]
        self.last_move = None
        self.moves = {}
        self.update_board()

    def reset_board(self):
        self.pos = parse_fen(start_position)
        self.last_move = None
        self.update_board()

    def update_board(self):
        self.moves = {}
        self.ui_board = [[None for _ in range(8)] for _ in range(8)]
        moves_list = generate_legal_moves(self.pos)
        for cur_move in moves_list:
            source = cur_move.source
            target = cur_move.target
            if source not in self.moves.keys():
                self.moves[source] = {}
            self.moves[source][target] = cur_move
        for color in range(2):
            for piece in range(PAWN, KING + 1):
                piece_list = self.pos.pieces[color][piece]
                while piece_list:
                    p = lsb_index(piece_list)
                    row, col = i_to_rc(p)
                    self.ui_board[row][col] = (color, piece, False)
                    if color == int(self.pos.side) and piece == KING:
                        self.ui_board[row][col] = (color, piece, is_king_in_check(self.pos))
                    piece_list = pop_bit(piece_list, p)
        if len(self.moves) == 0 and self.pos.state == NORMAL:
            self.pos.state = CHECKMATE if is_king_in_check(self.pos) else STALEMATE


class MainMenu:
    def __init__(self, game_controller):
        self.level = pygame_menu.Menu('Select Settings', GAME_WIDTH, GAME_HEIGHT, theme=themes.THEME_BLUE)
        self.level.add.selector('Difficulty :', [('Easy', 0), ('Medium', 1), ('Hard', 2)], onchange=self.set_difficulty)
        self.level.add.selector('Color :', [('White', 0), ('Black', 1)], onchange=self.set_color)
        self.main_menu = pygame_menu.Menu('Welcome', GAME_WIDTH, GAME_HEIGHT, theme=themes.THEME_GREEN)
        self.main_menu.add.button('Play Computer', self.start_computer_mode)
        self.main_menu.add.button('2 Players', self.start_user_mode)
        self.main_menu.add.button('Setting', self.level_menu)
        self.main_menu.add.button('Quit', pygame_menu.events.EXIT)
        self.controller = game_controller

    def set_difficulty(self, value, difficulty):
        self.controller.difficulty = difficulty

    def set_color(self, value, color):
        self.controller.user_color = color

    def start_user_mode(self):
        self.controller.game_type = USER
        self.controller.play()

    def start_computer_mode(self):
        self.controller.game_type = COMPUTER
        self.controller.play()

    def level_menu(self):
        self.main_menu._open(self.level)

    def main_loop(self):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    exit()
            if self.main_menu.is_enabled():
                self.main_menu.update(events)
                self.main_menu.draw(self.controller.screen)
                if self.main_menu.get_current().get_selected_widget():
                    self.main_menu.draw(self.controller.screen)
            pygame.display.update()


class BoardPanel:

    def __init__(self, screen):
        self.screen = screen
        self.piece_images = \
            {WHITE: {
                PAWN: pygame.image.load('assets/images/80px/wP.png'),
                KNIGHT: pygame.image.load('assets/images/80px/wN.png'),
                BISHOP: pygame.image.load('assets/images/80px/wB.png'),
                ROOK: pygame.image.load('assets/images/80px/wR.png'),
                QUEEN: pygame.image.load('assets/images/80px/wQ.png'),
                KING: pygame.image.load('assets/images/80px/wK.png'),

            }, BLACK: {
                ROOK: pygame.image.load('assets/images/80px/bR.png'),
                KNIGHT: pygame.image.load('assets/images/80px/bN.png'),
                BISHOP: pygame.image.load('assets/images/80px/bB.png'),
                QUEEN: pygame.image.load('assets/images/80px/bQ.png'),
                KING: pygame.image.load('assets/images/80px/bK.png'),
                PAWN: pygame.image.load('assets/images/80px/bP.png'),
            }}

    def show_board_panel(self, display_board: DisplayBoard, selected_square=None, user_side=WHITE):
        for row in range(ROWS):
            for col in range(COLS):
                color = DARK_COLOR if (row + col) % 2 == 0 else WHITE_COLOR
                rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        if user_side == WHITE:
            row_labels = [pygame.font.SysFont('arial', 18, bold=True)
                          .render(str(ROWS - row), 1, DARK_COLOR if row % 2 == 1 else WHITE_COLOR) for row in range(ROWS)]
            col_labels = [pygame.font.SysFont('arial', 18, bold=True)
                          .render(SQUARE_MARK[col + 1], 1, DARK_COLOR if col % 2 == 0 else WHITE_COLOR) for col in
                          range(COLS)]
        else:
            row_labels = [pygame.font.SysFont('arial', 18, bold=True)
                          .render(str(1 + row), 1, DARK_COLOR if row % 2 == 1 else WHITE_COLOR) for row in range(ROWS)]
            col_labels = [pygame.font.SysFont('arial', 18, bold=True)
                          .render(SQUARE_MARK[8 - col], 1, DARK_COLOR if col % 2 == 0 else WHITE_COLOR) for col in
                          range(COLS)]
        for row, lbl in enumerate(row_labels):
            self.screen.blit(lbl, (5, 5 + row * SQUARE_SIZE))
        for col, lbl in enumerate(col_labels):
            self.screen.blit(lbl, (col * SQUARE_SIZE + SQUARE_SIZE - 10, GAME_HEIGHT - 21))

        cur_board = display_board.ui_board
        for i in range(8):
            for j in range(8):
                if cur_board[i][j] is not None:
                    _c, p, in_check = cur_board[i][j]
                    if user_side == WHITE:
                        img_center = j * SQUARE_SIZE + SQUARE_SIZE // 2, i * SQUARE_SIZE + SQUARE_SIZE // 2
                        if in_check:
                            pygame.draw.rect(self.screen, RED,
                                             (j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
                    else:
                        _i, _j = i_to_rc(reversed_board[rc_to_i(i, j)])
                        img_center = _j * SQUARE_SIZE + SQUARE_SIZE // 2, _i * SQUARE_SIZE + SQUARE_SIZE // 2
                        if in_check:
                            pygame.draw.rect(self.screen, RED,
                                             (_j * SQUARE_SIZE, _i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
                    self.screen.blit(self.piece_images[_c][p], self.piece_images[_c][p].get_rect(center=img_center))

        if selected_square is not None:     # move hint
            row_sq, col_sq = i_to_rc(selected_square)
            if cur_board[row_sq][col_sq][0] == display_board.pos.side:
                if user_side == BLACK:
                    row_sq, col_sq = i_to_rc(reversed_board[selected_square])
                pygame.draw.rect(self.screen, GREEN, (col_sq * SQUARE_SIZE, row_sq * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
            for square in display_board.moves.get(selected_square, {}).keys():
                x, y = i_to_rc(square) if user_side == WHITE else i_to_rc(reversed_board[square])
                pygame.draw.rect(self.screen, BLUE, (y * SQUARE_SIZE, x * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)

        if display_board.last_move is not None:
            source, target = display_board.last_move
            if user_side == BLACK:
                source, target = reversed_board[source], reversed_board[target]
            row_from, col_from = i_to_rc(source)
            row_to, col_to = i_to_rc(target)
            target_rect1 = pygame.Rect((col_from * SQUARE_SIZE, row_from * SQUARE_SIZE), (SQUARE_SIZE, SQUARE_SIZE))
            target_rect2 = pygame.Rect((col_to * SQUARE_SIZE, row_to * SQUARE_SIZE), (SQUARE_SIZE, SQUARE_SIZE))
            shape_surface = pygame.Surface(target_rect1.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surface, LAST_MOVE_COLOR, shape_surface.get_rect())
            self.screen.blit(shape_surface, target_rect1)
            self.screen.blit(shape_surface, target_rect2)


class GUI:

    def __init__(self, screen):
        self.screen = screen
        self.board_panel = BoardPanel(screen)

    def draw_game(self, display_board, selected_square=None, user_side=WHITE):
        self.screen.fill(0)
        self.board_panel.show_board_panel(display_board, selected_square, user_side)
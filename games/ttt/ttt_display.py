from tkinter import Frame, Label, SUNKEN

from games.ttt.ttt_move import TTTMove

from willsmith.gui_display_controller import GUIDisplayController


class TTTDisplay(GUIDisplayController):
    """
    The display controller for NestedTTT games.

    Creates a Tkinter GUI that displays each of the boards.
    """

    WINDOW_TITLE = "Nested Tic-Tac-Toe"

    BOARD_DIM = 3

    LABEL_FONT = ("Comic Sans MS", 10)
    LABEL_HEIGHT = 2
    LABEL_WIDTH = 5

    FRAME_BORDER_WIDTH = 1
    FRAME_RELIEF = SUNKEN

    def __init__(self):
        super().__init__()
        self.outer_board = None
        self.inner_boards = None

    def _reset_display(self, game):
        """
        Update each board position label with a blank move.
        """
        for r, orow in enumerate(self.inner_boards):
            for c, board in enumerate(orow):
                for ir, irow in enumerate(board):
                    for ic, _ in enumerate(irow):
                        self._update_label((r,c), (ir,ic), TTTMove.BLANK)

    def _update_display(self, game, action):
        """
        Update the board labels with the latest action taken.
        """
        self._update_label(action.outer_pos, action.inner_pos, action.move)
    
    def _update_label(self, outer_pos, inner_pos, new_label):
        """
        Update the label at the given position with the new text.
        """
        r, c = outer_pos
        ir, ic = inner_pos
        self.inner_boards[r][c][ir][ic]["text"] = new_label

    def _initialize_widgets(self):
        """
        Create the outer and inner boards for display.

        The outer board consists of frames to group the labels for the inner
        board labels.
        """
        self.outer_board = [[Frame(self.root, bd = self.FRAME_BORDER_WIDTH, 
                                    relief = self.FRAME_RELIEF) 
                                for _ in range(self.BOARD_DIM)] 
                                    for _ in range(self.BOARD_DIM)]
        self.inner_boards = [[self._generate_inner_board(r, c) 
                                for c in range(self.BOARD_DIM)]
                                    for r in range(self.BOARD_DIM)]

    def _generate_inner_board(self, row, col):
        """
        Instantiate the labels used to display the board squares.
        """
        return [[Label(self.outer_board[row][col], width = self.LABEL_WIDTH, 
                            height = self.LABEL_HEIGHT, 
                            font = self.LABEL_FONT, text = TTTMove.BLANK) 
                    for _ in range(self.BOARD_DIM)]
                        for _ in range(self.BOARD_DIM)]

    def _place_widgets(self):
        self._place_board(self.outer_board)
        for row in self.inner_boards:
            for board in row:
                self._place_board(board)

    def _place_board(self, board):
        """
        Place the widgets using the grid geometry manager.
        """
        for i, row in enumerate(board):
            for j, widget in enumerate(row):
                widget.grid(row = i, column = j)

from copy import copy
from itertools import product

from games.ttt.ttt_move import TTTMove


class TTTBoard:
    """
    The gameboard for Tic-Tac-Toe.

    Holds the state for the board and keeps track of the winner of the board.

    The class attribute winning_positions is used for constant-time checking 
    if a board is in a winning state.
    """

    BOARD_SIZE = 3
    winning_positions = None

    def __init__(self):
        """
        Initialize the board with blank squares and no winner.

        On the first instantiation of the class it also generates the set 
        of winning board configurations.
        """
        self.board = [[TTTMove.BLANK for _ in range(TTTBoard.BOARD_SIZE)] for _ in range(TTTBoard.BOARD_SIZE)]
        self.winner = None

        if TTTBoard.winning_positions is None:
            TTTBoard.winning_positions = TTTBoard.generate_winning_positions()

    def take_action(self, position, move):
        """
        Applies the move to the board position.
        """
        r, c = position
        self.board[r][c] = move

    def get_winner(self):
        """
        Returns the move that won the board or None if it is still ongoing.
        """
        return self.winner

    def check_for_winner(self, move):
        """
        Check if the board has been won, returning a boolean to indicate this.

        If the board is won, update the winner attribute to the given move.
        """
        won = self._check_if_won()
        if won:
            self.winner = move
        return won

    def _check_if_won(self):
        """
        Check if the board is won by comparing it to the set of winning boards.
        """
        return tuple(map(tuple, self.board)) in self.winning_positions

    def __eq__(self, other):
        equal = False
        if isinstance(self, other.__class__):
            equal = self.winner == other.winner and self.board == other.board
        return equal

    def __hash__(self):
        return hash((self.winner, frozenset(self.board)))

    def __deepcopy__(self, memo):
        new = TTTBoard.__new__(TTTBoard)
        memo[id(self)] = new
        new.board = [copy(iboard) for iboard in self.board]
        new.winner = self.winner
        return new

    @staticmethod
    def generate_winning_positions():
        """
        Calculate every possible winning configuration of the board and 
        return the set.
        """
        bs = TTTBoard.BOARD_SIZE
        winning_positions = set()
        other_moves = [TTTMove.BLANK, TTTMove.X, TTTMove.O]
        all_possibilities = list(product(other_moves, repeat = (bs ** 2 - bs)))

        for move in [TTTMove.X, TTTMove.O]:
            winner = (move,) * bs
            for possibility in all_possibilities:
                #Creating row wins
                for r in range(bs):
                    board = []
                    pcopy = list(possibility)
                    for i in range(bs):
                        if i == r:
                            board.append(winner)
                        else:
                            board.append((pcopy.pop(), pcopy.pop(), pcopy.pop()))
                    winning_positions.add(tuple(board))

                #Creating column wins
                for col in range(bs):
                    board = []
                    pcopy = list(possibility)
                    for _ in range(bs):
                        board.append(tuple((move if curr == col else pcopy.pop() for curr in range(bs))))
                    winning_positions.add(tuple(board))

                #Creating diagonal wins
                board = []
                pcopy = list(possibility)
                for d in range(bs):
                    board.append(tuple(move if i == d else pcopy.pop() for i in range(bs)))
                winning_positions.add(tuple(board))

                #Creating counter-diagonal wins
                board = []
                pcopy = list(possibility)
                for d in range(bs):
                    board.append(tuple(move if (i + d) == (bs - 1) else pcopy.pop() for i in range(bs)))
                winning_positions.add(tuple(board))
        return winning_positions

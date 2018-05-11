from copy import deepcopy

from games.ttt.ttt_action import TTTAction
from games.ttt.ttt_board import TTTBoard
from games.ttt.ttt_display import TTTDisplay
from games.ttt.ttt_move import TTTMove

from willsmith.game import Game


class NestedTTT(Game):
    """
    A game of Nested Tic-Tac-Toe, where each square on the outer board 
    is another Tic-Tac-Toe board.

    The game is played as standard Tic-Tac-Toe, except there are 9 inner 
    boards to make moves on.  Winning an inner board claims that space for 
    the winner on the outer board.  Draws result in a square that does not 
    count for either side.  Winning the game requires winning 3 inner boards 
    in a row, forming a win on the outer board.

    The state of the game is stored using the TTTBoard class:  one for the 
    outer board and one for each of the 9 inner boards.  Actions are never 
    explicitly taken by the players on the outer board, only as a result of a 
    win on an inner board.

    See Game documentation for the public API expected by other classes.
    """

    ACTION = TTTAction
    DISPLAY = TTTDisplay
    NUM_PLAYERS = 2

    def __init__(self, use_display):
        super().__init__(use_display)
        self._reset()

    def _reset(self):
        """
        Initialize the state for the game and legal moves.
        """
        self.outer_board = TTTBoard()
        self.inner_boards = [[TTTBoard() for _ in range(TTTBoard.BOARD_SIZE)] 
                                for _ in range(TTTBoard.BOARD_SIZE)]

        self.legal_actions = self._generate_initial_legal_actions()

    def _generate_initial_legal_actions(self):
        """
        Generate a dictionary of actions for each position on each inner 
        board.
        """
        cur_move = self._agent_id_to_move(self.current_agent_id)
        return {((r, c), (ir, ic)) : self.ACTION((r, c), (ir, ic), cur_move) 
                    for r in range(TTTBoard.BOARD_SIZE)
                    for c in range(TTTBoard.BOARD_SIZE)
                        for ir in range(TTTBoard.BOARD_SIZE)
                        for ic in range(TTTBoard.BOARD_SIZE)}

    def _get_legal_actions(self):
        self._update_legal_actions()
        return list(self.legal_actions.values())

    def _update_legal_actions(self):
        cur_move = self._agent_id_to_move(self.current_agent_id)
        for action in self.legal_actions.values():
            action.move = cur_move

    def get_winning_id(self):
        winner_id = None
        if self.outer_board.winner is not None:
            winner_id = self._move_to_agent_id(self.outer_board.winner)
        return winner_id

    def is_legal_action(self, action):
        """
        Check that the action's position is legal and that the action's move 
        matches the expected move for the current turn.
        """
        legal_position = (action.outer_pos, action.inner_pos) in self.legal_actions
        legal_move = action.move == self._agent_id_to_move(self.current_agent_id)
        return legal_position and legal_move

    def _take_action(self, action):
        """
        Take the given action on the indicated inner board.  

        If this triggers a win, update the outer board with the players 
        move.  Either way, remove the positions that are no longer legal from 
        the set of legal positions.
        """
        r, c = action.outer_pos

        self.inner_boards[r][c].take_action(action.inner_pos, action.move)
        board_won = self.inner_boards[r][c].check_for_winner(action.move)
        if board_won:
            self.outer_board.take_action(action.outer_pos, action.move)
            self.outer_board.check_for_winner(action.move)

        self._remove_illegal_actions(action.outer_pos, action.inner_pos, board_won)

    def _remove_illegal_actions(self, outer_pos, inner_pos, board_won):
        """
        Remove the position from the legal actions.

        If the position won the board, all the remaining open squares on 
        that inner board are now illegal and removed as well.
        """
        del self.legal_actions[(outer_pos, inner_pos)]
        if board_won:
            for r in range(TTTBoard.BOARD_SIZE):
                for c in range(TTTBoard.BOARD_SIZE):
                    try:
                        del self.legal_actions[(outer_pos, (r, c))]
                    except KeyError:    # moves that have already been taken
                        pass

    def is_terminal(self):
        is_winner = self.outer_board.get_winner() is not None
        moves_left = bool(self.legal_actions)
        return is_winner or not moves_left

    def _agent_id_to_move(self, agent_id):
        lookup = {0 : TTTMove.X, 1 : TTTMove.O}
        return lookup[agent_id]

    def _move_to_agent_id(self, move):
        lookup = {TTTMove.X : 0, TTTMove.O: 1}
        return lookup[move]

    def __str__(self):
        """
        Return a string of the gameboard that looks like this:

        [  X  ] | [  O  ] | [O X O]
        [     ] | [X   O] | [O    ]
        [O X  ] | [X O  ] | [  O  ]
        ---------------------------
        [     ] | [O   X] | [X   X]
        [  X  ] | [  X  ] | [     ]
        [    X] | [X   O] | [     ]
        ---------------------------
        [     ] | [     ] | [O   O]
        [     ] | [  O  ] | [O    ]
        [X    ] | [    O] | [X X  ]
        """
        result = []
        for i in range(TTTBoard.BOARD_SIZE):
            for j in range(TTTBoard.BOARD_SIZE):
                result.append(" | ".join([str(inner_board.board[j]) for inner_board in self.inner_boards[i]]))
            if i < 2:
                result.append("-"*27)
        board_string = "\n".join(result)
        return board_string.replace(',', '')

    def __eq__(self, other):
        equal = False
        if isinstance(self, other.__class__):
            equal = (super().__eq__(other)
                        and self.outer_board == other.outer_board
                        and self.inner_boards == other.inner_boards
                        and self.legal_actions == other.legal_actions)
        return equal

    def __hash__(self):
        return hash((self.num_agents, self.current_agent_id, 
                        self.outer_board, self.inner_boards, 
                        frozenset(self.legal_actions)))

    def __deepcopy__(self, memo):
        new = NestedTTT.__new__(NestedTTT)
        memo[id(self)] = new
        self.deepcopy_game_attrs(new)

        new.outer_board = deepcopy(self.outer_board, memo)
        new.inner_boards = [deepcopy(board, memo) for board in self.inner_boards]
        new.legal_actions = {k : deepcopy(v, memo) 
                                for k, v in self.legal_actions.items()}
        return new

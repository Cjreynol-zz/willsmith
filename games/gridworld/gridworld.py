from copy import deepcopy
from random import choices

from games.gridworld.gridworld_action import GridworldAction
from games.gridworld.gridworld_direction import GridworldDirection

from willsmith.game import Game
from willsmith.simple_displays import ConsoleDisplay


class Gridworld(Game):
    """
    A wrapper for the standard example of a Markov Decision Process.  
    """

    ACTION = GridworldAction
    DISPLAY = ConsoleDisplay
    NUM_PLAYERS = 1

    def __init__(self, grid, transition_func):
        """
        """
        super().__init__()

        self.grid = grid
        self.transition_func = transition_func
        self.player_pos = (0,0)
        self.terminal = False
        self.legal_actions = [GridworldAction(direction) for direction in GridworldDirection]

    def _get_legal_actions(self):
        return self.legal_actions

    def is_legal_action(self, action):
        return action.direction in GridworldDirection

    def _take_action(self, action):
        """
        """
        actions, weights = self.transition_func(action)
        # choices returns a list but we only ever return one result
        resulting_action = choices(actions, weights = weights)[0]
        return self._apply_action(resulting_action)

    def _apply_action(self, action):
        """
        """
        x, y = self.player_pos
        dx, dy = GridworldDirection.get_offset(action.direction)
        next_coord = (x + dx, y + dy)

        if self._valid_position(next_coord):
            self.player_pos = next_coord
            self.terminal = self.grid[next_coord].terminal

        result_pos = self.player_pos
        return result_pos, self.grid[result_pos].reward

    def _valid_position(self, next_coord):
        return next_coord in self.grid

    def get_winning_id(self):
        raise NotImplementedError("You Win!")

    def is_terminal(self):
        return self.terminal

    def __str__(self):
        results = []
        for y in range(self.grid.size[1]-1, -1, -1):
            row = []
            for x in range(self.grid.size[0]):
                square = "   "
                if (x,y) in self.grid:
                    if self.player_pos == (x, y):
                        square = "[A]"
                    elif self.grid[(x, y)].terminal:
                        square = "[T]"
                    else:
                        square = "[ ]"
                row.append(square)
            results.append("".join(row))
        return "\n".join(results)

    def __eq__(self, other):
        equal = False
        if isinstance(self, other__class__):
            equal = (self.grid == other.grid
                        and self.transition_func == other.transition_func
                        and self.player_pos == other.player_pos
                        and self.terminal == other.terminal
                        and self.legal_actions == other.legal_actions)
        return equal

    def __hash__(self):
        return hash((self.grid, self.transition_func, self.player_pos,
                        self.terminal, self.legal_actions))

    def __deepcopy__(self, memo):
        new = Gridworld.__new__(Gridworld)
        memo[id(self)] = new

        # should rely on inheritance for these attributes
        new.num_agents = self.num_agents
        new.current_agent_id = self.current_agent_id

        new.grid = deepcopy(self.grid, memo)
        new.transition_func = self.transition_func
        new.player_pos = self.player_pos
        new.terminal = self.terminal
        new.legal_actions = [deepcopy(action, memo) 
                                for action in self.legal_actions]
        return new
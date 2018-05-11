from abc import ABC, abstractmethod
from copy import deepcopy
from random import choice

from willsmith.action import Action
from willsmith.display_controller import DisplayController
from willsmith.simple_displays import ConsoleDisplay, NoDisplay


class Game(ABC):
    """
    Abstract base class for games.

    Subclasses are turn-based games where players alternate taking single 
    actions until a winner is decided.  
    
    The interface enforced by this class is used by GameAgent instances to 
    determine their next action, and by the simulator module to control the 
    game as it runs agents through a match.

    The ACTION class attribute is expected to be a subclass of the Action 
    base class.

    The DISPLAY class attribute is used by subclasses to register custom 
    displays.

    The NUM_PLAYERS class attribute is required to set the number of agents 
    expected for the game.
    """

    ACTION = None
    DISPLAY = None
    NUM_PLAYERS = None

    def __init__(self, use_display):
        """
        Set the game to start with the first player.

        Properly sets the display attribute based on the use_display argument, 
        and also checks that the required class attributes are set.
        """
        self.num_agents = self.NUM_PLAYERS
        self.current_agent_id = 0

        self.display = NoDisplay()
        if use_display is not None:
            if not use_display or self.DISPLAY is None:
                self.display = ConsoleDisplay()
            else:
                self.display = self.DISPLAY()
        self.display.start(is_main = True)

        if (self.ACTION is None 
                or not issubclass(self.ACTION, Action)):
            raise RuntimeError("Game must set its own action, which must subclass Action.")
        if self.NUM_PLAYERS is None:
            raise RuntimeError("Game must set expected number of players.")

    def get_legal_actions(self):
        """
        Return a list of the legal actions available to the current player.  

        In the case where the game is in a terminal state, this is an empty 
        list regardless of any other game state.
        """
        results = []
        if not self.is_terminal():
            results = self._get_legal_actions()
        return results

    def take_action(self, action):
        """
        Apply the action, which must be legal, to the game state.

        Also update the current player id and the display if it is set.
        """
        if not self.is_legal_action(action):
            raise RuntimeError("Received illegal action: {}".format(action))

        self._take_action(action)
        self._increment_current_agent_id()
        if self.display is not None:    # display is None in copies
            self.display.update_display(self, action)

    def reset(self):
        """
        Revert the bookkeeping attributes back to their initial state, as 
        well as the display.
        """
        self.current_agent_id = 0
        self._reset()

        if self.display is not None:    # display is None in copies
            self.display.reset_display(self)

    @abstractmethod
    def _reset(self):
        """
        Revert the game back to its initial state.
        """
        pass

    @abstractmethod
    def _get_legal_actions(self):
        """
        Return a list of the available actions for the current agent in the
        current state of the game.
        """
        pass

    @abstractmethod
    def is_legal_action(self, action):
        """
        Return a boolean indicating if the action is valid and legal.
        """
        pass

    @abstractmethod
    def _take_action(self, action):
        """
        Progress the internal game state by the given action.
        """
        pass

    @abstractmethod
    def get_winning_id(self):
        """
        Return the agent id of the player that won the game.

        None typically indicates a draw or an ongoing game.
        """
        pass

    @abstractmethod
    def is_terminal(self):
        """
        Return a boolean indicating if the game is in a terminal state.
        """
        pass

    def generate_random_action(self):
        """
        Make a random choice of the available legal actions for the current 
        player.
        """
        return choice(self.get_legal_actions())

    def copy(self):
        """
        Return a deepcopy of the game state where references are not shared.
        """
        return deepcopy(self)

    def _increment_current_agent_id(self):
        """
        Increment the agent id of the current player, indicating a new turn.

        This method enforces the id stay within the legal range of 
        [0, num_agents).
        """
        self.current_agent_id += 1
        if self.current_agent_id == self.num_agents:
            self.current_agent_id = 0

    def __eq__(self, other):
        return self.num_agents == other.num_agents and self.current_agent_id == other.current_agent_id

    def deepcopy_game_attrs(self, new):
        """
        Used by subclasses to copy over the game attributes to a new deepcopy 
        of the subclass.

        The display attribute is not copied, so that copies cannot modify the 
        current game display.
        """
        new.num_agents = self.num_agents
        new.current_agent_id = self.current_agent_id
        new.display = None
        return new

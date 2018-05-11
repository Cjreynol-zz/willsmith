from abc import ABC, abstractmethod
from copy import copy, deepcopy
from random import choice

from willsmith.simple_displays import ConsoleDisplay, NoDisplay


class MDP(ABC):
    """
    Abstract base class for Markov Decision Processes.

    The interface enforced by this class is used by MDPAgent instances to 
    provide the necessary information for them to learn the MDP, and by the 
    simulator module to control the MDP as it runs an agent through trials of 
    the MDP.

    The DISPLAY class attribute is used by subclasses to register custom 
    displays.
    """

    DISPLAY = None

    def __init__(self, use_display):
        """
        Initialize the bookkeeping attributes for use during a trial.

        Properly sets the display attribute based on the use_display argument.
        """
        self.timesteps = 0
        self.total_reward = 0
        self.reward_history = []

        self.display = NoDisplay()
        if use_display is not None:
            if not use_display or self.DISPLAY is None:
                self.display = ConsoleDisplay()
            else:
                self.display = self.DISPLAY()
        self.display.start(is_main = True)

    @abstractmethod
    def get_action_space(self):
        """
        Return a list of the legal actions that can be taken by agents.
        """
        pass

    @abstractmethod
    def _step(self, action):
        """
        Transition the MDP using the given action.
        """
        pass

    @abstractmethod
    def _undo(self):
        """
        Undo the last action taken by the MDP.
        """
        pass

    @abstractmethod
    def _reset(self):
        """
        Revert the MDP back to its initial state.
        """
        pass

    @abstractmethod
    def is_terminal(self):
        """
        Return a boolean indicating if the MDP is in a terminal state.
        """
        pass

    def step(self, action):
        """
        Transition the MDP by the given action, provided it is legal.

        Also update the bookkeeping attributes and the display if it is set.
        """
        if not self.is_legal_action(action):
            raise RuntimeError("Received illegal action: {}".format(action))

        reward, terminal = self._step(action)

        self.timesteps += 1
        self.total_reward += reward
        self.reward_history.append(reward)

        if self.display is not None:
            self.display.update_display(self, action)
        
        return reward, terminal

    def undo(self, action):
        """
        Undo the last action taken by the MDP.
        """
        self.timesteps -= 1
        self.total_reward -= self.reward_history.pop()
        self._undo(action)

        if self.display is not None:
            self.display.update_display(self, action)

    def reset(self):
        """
        Revert the bookkeeping attributes back to their initial state, as 
        well as the display.
        """
        self.timesteps = 0
        self.total_reward = 0
        self.reward_history = []
        self._reset()
        
        if self.display is not None:
            self.display.reset_display(self)

    def is_legal_action(self, action):
        """
        Return a boolean indicating if the action is valid and legal.
        """
        return not self.is_terminal() and action in self.action_space

    def action_space_sample(self):
        """
        Make a random choice from the MDP's action space.
        """
        return choice(self.action_space)

    def copy(self):
        """
        Return a deepcopy of the MDP state where references are not shared.
        """
        return deepcopy(self)

    def __eq__(self, other):
        return (self.timesteps == other.timesteps
                    and self.total_reward == other.total_reward
                    and self.reward_history == other.reward_history)

    def deepcopy_mdp_attrs(self, new):
        """
        Used by subclasses to copy over the game attributes to a new deepcopy 
        of the subclass.

        The display attribute is not copied, so that copies cannot modify the 
        current game display.
        """
        new.timesteps = self.timesteps
        new.total_reward = self.total_reward
        new.reward_history = copy(self.reward_history)
        new.display = None
        return new

from abc import ABC, abstractmethod


class GameAgent(ABC):
    """
    Abstract base class for game playing agents.
    
    Subclasses are agents that are capable of searching for and returning 
    actions within an allotted time and updating their internal state based 
    on a given action taken in the game.

    Subclasses that set the DISPLAY attribute will have that display 
    instantiated when Simulator runs a game using a GUI.
    """

    GUI_DISPLAY = None

    def __init__(self, agent_id, use_gui):
        self.agent_id = agent_id
        self.display = None
        if use_gui and self.GUI_DISPLAY is not None:
            self.display = self.GUI_DISPLAY()
            self.display.start(is_main = False)

    @abstractmethod
    def _reset(self):
        """
        Revert the agent back to its initial state.
        """
        pass

    @abstractmethod
    def search(self, state, allotted_time):
        """
        Search the action space for the next action to take.  

        Search strategy depends on the subclass implementation.
        """
        pass

    @abstractmethod
    def _take_action(self, action):
        """
        Update the agent's internal state by the latest action taken in the 
        game.
        """
        pass

    def take_action(self, action, is_my_action):
        if is_my_action and self.display is not None:
            self.display.update_display(self, action)
        self._take_action(action)
        
    def reset(self):
        self._reset()
        if self.display is not None:
            self.display.reset_display(self)

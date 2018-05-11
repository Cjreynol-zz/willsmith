from abc import ABC, abstractmethod
from random import random


class MDPAgent(ABC):
    """
    Abstract base class for MDP learning agents.

    Subclasses are agents that are capable of determining actions given the 
    state of the MDP, as well as updating their internal knowledge based on 
    the result of the given action.
    """

    def __init__(self, action_space, learning_rate, discount, 
                    exploration_rate):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate

    @abstractmethod
    def _get_next_action(self, state):
        """
        Return the next action the agent decides to take in the given state.
        """
        pass

    @abstractmethod
    def update(self, prev_state, curr_state, reward, action, terminal):
        """
        Update the agent's internal state provided with the result of their 
        last taken action.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Revert the agent back to its initial state.
        """
        pass

    def get_next_action(self, state):
        """
        Return an action for the agent to take in the MDP.

        Balances exploitation versus exploration by choosing random actions 
        at a rate determined by the agent's exploration rate.
        """
        action = state.action_space_sample()
        if random() > self.exploration_rate:
            action = self._get_next_action(state)
        return action

from random import random

from willsmith.mdp_agent import MDPAgent


class ApproxQLearningAgent(MDPAgent):
    """
    A template for an agent that uses appoximate Q-learning to learn a policy.

    The expectation is that this base class subclassed with a given set of 
    feature functions that provide it with some domain specific knowledge 
    about and MDP and that agent will attempt to learn an optimal policy.

    This class takes the given set of feature functions and slowly updates a 
    weighted value for each of them while progressing through the MDP.  These 
    weights are updated based on the difference between the expected reward 
    for a transition from one state to antoher and the actual reward received.
    """

    def __init__(self, action_space, learning_rate, discount, 
                    exploration_rate, feature_functions):
        super().__init__(action_space, learning_rate, discount, 
                            exploration_rate)

        self.features = feature_functions
        self.weights = self.create_random_weight_list()

    def reset(self):
        self.weights = self.create_random_weight_list()
        
    def create_random_weight_list(self):
        return [random() for _ in range(len(self.features))]

    def _get_next_action(self, state):
        """
        Returns the best(determined by sum of weighted feature functions)
        available action given the current state.
        """
        values = {action : self._get_q_value(state, action) 
                    for action in self.action_space}
        return max(values, key=values.get)

    def update(self, prev_state, curr_state, reward, last_action, terminal):
        """
        Updates the feature weights based on expected future rewards and
        the difference between the previous reward and what it was expected
        to be.
        """
        difference = self._calculate_difference(prev_state, curr_state, 
                                                reward, last_action, 
                                                terminal)
        self.weights = self._update_weights(prev_state, last_action, 
                                                self.learning_rate, 
                                                difference, self.features, 
                                                self.weights)

    def _update_weights(self, prev_state, last_action, learning_rate, 
                            difference, features, weights):
        return [(weight + (learning_rate 
                            * difference 
                            * feature(prev_state, last_action)))
                    for feature, weight in zip(features, weights)]

    def _get_q_value(self, state, action):
        """
        Calculate the q-value for a given state, action pair.
        """
        state.step(action)
        q_value = sum([feature(state, action) * weight 
                        for (feature, weight) 
                            in zip(self.features, self.weights)])
        state.undo(action)
        return q_value

    def _get_max_q_value(self, state):
        values = [self._get_q_value(state, action) 
                    for action in self.action_space]
        return max(values)

    def _calculate_difference(self, prev_state, curr_state, reward, last_action, 
                                terminal):
        """
        Calculates the difference received reward and what was predicted.
        """
        future_reward = 0
        if not terminal:
            future_reward = self._get_max_q_value(curr_state)

        expected_reward = self._get_q_value(prev_state, last_action)
        return (reward + self.discount * future_reward) - expected_reward

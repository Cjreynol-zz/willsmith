from random import random

from willsmith.mdp_agent import MDPAgent


class ApproxQLearningAgent(MDPAgent):
    """
    A template for an agent that uses appoximate Q-learning to learn a policy.

    This agent expects to be given domain-specific knowledge about the MDP it 
    is supposed to learn in the form of a list of feature functions.

    This class weights the set of feature functions, updating their values 
    after each action taken.  The amount to update is calculated as the 
    difference between the expected reward from the previous action and the 
    actual reward received.
    """

    def __init__(self, action_space, learning_rate, discount, 
                    exploration_rate, feature_functions):
        super().__init__(action_space, learning_rate, discount, 
                            exploration_rate)

        self.features = feature_functions
        self.reset()

    def reset(self):
        self.weights = self._create_random_weight_list()
        
    def _create_random_weight_list(self):
        return [random() for _ in range(len(self.features))]

    def _get_next_action(self, state):
        """
        Return the best(determined by sum of weighted feature functions)
        available action given the current state.
        """
        values = {action : self._get_q_value(state, action) 
                    for action in self.action_space}
        return max(values, key=values.get)

    def update(self, prev_state, curr_state, reward, last_action, terminal):
        """
        Update the feature weights based on expected future rewards and
        the difference between the expected previous reward and its actual 
        value.
        """
        difference = self._calculate_difference(prev_state, curr_state, 
                                                reward, last_action, 
                                                terminal)
        self.weights = [(weight + (self.learning_rate 
                            * difference 
                            * feature(prev_state, last_action)))
                        for feature, weight in zip(self.features, self.weights)]

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
        Calculate the difference between the received reward and what was 
        predicted.
        """
        future_reward = 0
        if not terminal:
            future_reward = self._get_max_q_value(curr_state)

        expected_reward = self._get_q_value(prev_state, last_action)
        return (reward + self.discount * future_reward) - expected_reward

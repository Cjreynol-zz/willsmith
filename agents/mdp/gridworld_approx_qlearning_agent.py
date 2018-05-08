from agents.mdp.approx_qlearning_agent import ApproxQLearningAgent


class GridworldApproxQLearningAgent(ApproxQLearningAgent):
    """
    """
    
    def __init__(self, action_space, learning_rate, discount, exploration_rate):
        feature_functions = [self.x_pos_feature, self.y_pos_feature,
                                self.dist_from_goal]
        super().__init__(action_space, learning_rate, discount, 
                            exploration_rate, feature_functions)

    def x_pos_feature(self, state, action):
        return state.player_pos[0]

    def y_pos_feature(self, state, action):
        return state.player_pos[1]

    def dist_from_goal(self, state, action):
        x, y = state.player_pos
        dist = (abs(3 - x) + abs(2 - y))
        if dist == 0:   # make dist 0 the largest value
            dist = 0.5
        return 1 / dist

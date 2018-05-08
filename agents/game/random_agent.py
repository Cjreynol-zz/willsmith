from willsmith.game_agent import GameAgent


class RandomAgent(GameAgent):
    """
    Agent that chooses random actions regardless of the game state.
    """

    def __init__(self, agent_id, use_gui):
        super().__init__(agent_id, use_gui)

    def search(self, state, allotted_time):
        """
        Request a random, legal action from the game state, then return it.
        """
        return state.generate_random_action()

    def _reset(self):
        pass

    def _take_action(self, action):
        pass

    def __str__(self):
        return ""

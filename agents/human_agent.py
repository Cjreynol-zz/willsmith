from agents.displays.human_display import HumanDisplay
from willsmith.game_agent import GameAgent


class HumanAgent(GameAgent):
    """
    Agent that relies on user input to make action choices.

    It relies on its action_prompt attribute, set externally by the 
    simulator, to provide the proper prompts and to construct the action.
    """
    
    GUI_DISPLAY = None #HumanDisplay is not yet ready
    INPUT_PROMPT = None
    INPUT_PARSER = None

    def __init__(self, agent_id, use_gui, action):
        super().__init__(agent_id, use_gui)

        self.add_input_info(action)

    def add_input_info(self, action):
        HumanAgent.INPUT_PROMPT = action.INPUT_PROMPT
        HumanAgent.INPUT_PARSER = action.parse_action

    def search(self, state, allotted_time):
        """
        Prompt the player for an action until a legal action is chosen, then 
        return it.
        """
        legal_actions = state.get_legal_actions()
        player_action = HumanAgent.INPUT_PARSER(input(HumanAgent.INPUT_PROMPT))
        while player_action not in legal_actions:
            print("Last move was not legal, please try again.\n")
            player_action = HumanAgent.INPUT_PARSER(input(HumanAgent.INPUT_PROMPT))

        return player_action

    def _take_action(self, action):
        pass

    def _reset(self):
        pass

    def __str__(self):
        return ""

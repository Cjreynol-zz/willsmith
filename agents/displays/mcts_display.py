from tkinter import Label, GROOVE

from willsmith.gui_display_controller import GUIDisplayController


class MCTSDisplay(GUIDisplayController):
    """
    Creates a Tkinter GUI that displays stats about the agent's latest moves.
    """

    WINDOW_TITLE = "MCTS Agent"

    LABEL_FONT = ("Courier New", 14)
    LABEL_WIDTH = 25
    LABEL_BORDER_WIDTH = 1
    LABEL_RELIEF = GROOVE

    def __init__(self):
        super().__init__()

        self.playouts_label = None
        self.action_label = None
        self.win_pct_label = None
        self.depth_label = None

    def _initialize_widgets(self):
        self.playouts_label = self._create_label()
        self.action_label = self._create_label()
        self.win_pct_label = self._create_label()
        self.depth_label = self._create_label()

    def _create_label(self):
        return Label(self.root, font = self.LABEL_FONT, 
                        width = self.LABEL_WIDTH, 
                        bd = self.LABEL_BORDER_WIDTH, relief = GROOVE)

    def _place_widgets(self):
        self.playouts_label.grid(row = 0, column = 0)
        self.action_label.grid(row = 1, column = 0)
        self.win_pct_label.grid(row = 2, column = 0)
        self.depth_label.grid(row = 3, column = 0)

    def _update_display(self, agent, action):
        self._update_labels_from_agent(agent, action)

    def _reset_display(self, agent):
        self._update_labels_from_agent(agent, None)

    def _update_labels_from_agent(self, agent, action):
        self.action_label["text"] = "Latest action:\n{}".format(action)
        self.playouts_label["text"] = "Latest playout count:\n{}".format(agent.playout_total)
        win_pct = 0
        if agent.action_node is not None:
            win_pct = agent.action_node.value_estimate()
        self.win_pct_label["text"] = "Node sim win rate:\n{:.2%}".format(win_pct)
        self.depth_label["text"] = "Node tree depth:\n{}".format(agent.root.depth())

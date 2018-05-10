from tkinter import Button, Entry, Label, END

from willsmith.gui_display_controller import GUIDisplayController


class HumanDisplay(GUIDisplayController):
    """
    Creates a Tkinter GUI that allows the user to input their moves.
    """

    WINDOW_TITLE = "Human Agent"

    LABEL_FONT = ("Courier New", 14)
    
    def __init__(self):
        super().__init__()

        self.input_prompt_label = None
        self.input_entry = None
        self.submit_button = None

    def _initialize_widgets(self):
        self.input_prompt_label = Label(self.root, font = self.LABEL_FONT,
                                            text = "<prompt here>")
        self.input_entry = Entry(self.root)
        self.submit_button = Button(self.root, text = "Submit") 

    def _place_widgets(self):
        self.input_prompt_label.grid(row = 0, column = 0, columnspan = 2)
        self.input_entry.grid(row = 1, column = 0)
        self.submit_button.grid(row = 1, column = 1)

    def _update_display(self, agent, action):
        self._reset_display(agent)

    def _reset_display(self, agent):
        self.input_entry.delete(0, END)

    def _submit_entry():
        pass

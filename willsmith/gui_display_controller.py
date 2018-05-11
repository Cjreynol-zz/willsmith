from abc import abstractmethod
from tkinter import Tk, Toplevel

from willsmith.display_controller import DisplayController


class GUIDisplayController(DisplayController):
    """
    Abstract base class for Tkinter GUI-based display controllers.

    Sub-classes are expected to provide functionality for widget creation and 
    management.
    """

    def __init__(self):
        self.root = None

    @abstractmethod
    def _initialize_widgets(self):
        """
        Instantiate and assign all the widget attributes for the object.
        """
        pass

    @abstractmethod
    def _place_widgets(self):
        """
        Use a tkinter geometry manager to put the widgets in the main window.
        """
        pass

    @abstractmethod
    def _update_display(self, state, action):
        pass

    @abstractmethod
    def _reset_display(self, state):
        pass

    def start(self, is_main):
        self.root = self._initialize_root(is_main)
        self._initialize_widgets()
        self._place_widgets()

    def _initialize_root(self, is_main):
        """
        Create the window root and update its configuration.
        """
        if is_main:
            root = Tk()
        else:
            root = Toplevel()
        root.title(self.WINDOW_TITLE)
        return root

    def update_display(self, state, action):
        """
        Ensure _update_window method is called when the display is updated.
        """
        self._update_display(state, action)
        self._update_window()

    def reset_display(self, state):
        """
        Ensure _update_window method is called when the display is updated.
        """
        self._reset_display(state)
        self._update_window()

    def _update_window(self):
        """
        Update the GUI window with any changes that have taken place.
       
        This is used to manually update the window only after the display has 
        actually had a background update, instead of letting it control the 
        thread of execution.
        """
        self.root.update_idletasks()
        self.root.update()

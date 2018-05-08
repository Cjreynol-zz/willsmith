from tkinter import Canvas, BOTH

from mdps.gridworld.gridworld_direction import GridworldDirection

from willsmith.gui_display_controller import GUIDisplayController


class GridworldDisplay(GUIDisplayController):
    """
    The display controller for Gridworld.

    Creates a Tkinter GUI with a Canvas that is uses to draw the grid squares.

    Grid coordinates are converted to pixel coordinates taking into account 
    the canvas swapping the y-axis from a standard coordinate system.
    """

    WINDOW_TITLE = "Gridworld"

    SQUARE_WIDTH = 64
    SQUARE_BORDER = 2

    BLACK = "#000000"
    MAX_COLOR_VALUE = 256

    BACKGROUND_COLOR = "#FFFFFF"
    SQUARE_EMPTY_COLOR = "#A9A9A9"
    SQUARE_BORDER_COLOR = BLACK
    SQUARE_AGENT_COLOR = "#FFFF00"

    def __init__(self):
        super().__init__()
        self.canvas = None
        self.height = None
        self.terminal_colors = None

    def _initialize_widgets(self):
        self.canvas = Canvas(self.root)
        self.root["background"] = self.BACKGROUND_COLOR
        self.canvas["background"] = self.BACKGROUND_COLOR

    def _place_widgets(self):
        self.canvas.pack(fill = BOTH, expand = True)

    def _update_display(self, state, action):
        """
        """
        self._draw_grid(state)

    def _reset_display(self, state):
        """
        Resize the window/canvas, then draw the entire board as is.

        Expects the agent to have been put back in its starting position.
        """
        width, height = state.grid.size
        self.height = (height * self.SQUARE_WIDTH 
                        + self.SQUARE_BORDER * 4)
        self.root.geometry("{}x{}".format((width * self.SQUARE_WIDTH 
                                            + self.SQUARE_BORDER * 4), 
                                            self.height))
        self.lookup = self._create_color_lookup(state)
        self._draw_grid(state)

    def _draw_grid(self, state):
        for pos in state.grid.keys():
            agent_square = False
            if pos == state.player_pos:
                agent_square = True
            canvas_pos = self._square_coord_to_canvas_coord(*pos)
            self._draw_square(state.grid[pos], canvas_pos, agent_square)

    def _create_color_lookup(self, state):
        """
        Create a dictionary from terminal reward values to a hex color string.

        0 is black, negative rewards are shades of red, positive rewards are 
        shades of green.
        """
        neg_rewards, pos_rewards = self._get_terminal_rewards_lists(state)

        # the - len() calls are to remove a single step from range, so that 
        # 0 * step can be skipped to avoid black in both ranges
        neg_color_step = (self.MAX_COLOR_VALUE - len(neg_rewards)) // len(neg_rewards)
        pos_color_step = (self.MAX_COLOR_VALUE - len(pos_rewards)) // len(pos_rewards)
        
        lookup = {0 : self.BLACK}
        lookup.update({reward : self._num_to_hex_string((i + 1) * neg_color_step, True)
                    for i, reward in enumerate(neg_rewards)})
        lookup.update({reward : self._num_to_hex_string((i + 1) * pos_color_step, False)
                        for i, reward in enumerate(pos_rewards)})
        return lookup

    def _num_to_hex_string(self, num, is_red):
        """
        Return an RGB hex string with the hex value of the given num.

        is_red determines which position the hex value from num is used in, 
        with the other two colors being 00.
        """
        hex_str = hex(num)[2:]      # remove 0x from the front of the str
        if is_red:
            color = "#{}0000".format(hex_str)
        else:
            color = "#00{}00".format(hex_str)
        return color

    def _get_terminal_rewards_lists(self, state):
        """
        Return two lists, one each for the negative and positive terminal 
        rewards.
        """
        neg_rewards = []
        pos_rewards = []
        for square in state.grid.values():
            if square.terminal:
                if square.reward < 0 and square.reward not in neg_rewards:
                    neg_rewards.append(square.reward)
                elif square.reward > 0 and square.reward not in pos_rewards:
                    pos_rewards.append(square.reward)
        return sorted(neg_rewards), sorted(pos_rewards)

    def _square_coord_to_canvas_coord(self, x, y):
        """
        Convert a square's position on the grid to a pixel position on the 
        canvas.

        The subtraction in the y coordinate is because Tkinter reverses the 
        y-axis from a standard orientation.
        """
        canvas_x = x * self.SQUARE_WIDTH + self.SQUARE_BORDER * 2
        canvas_y = (self.height - self.SQUARE_BORDER * 2 - 
                        (y + 1) * self.SQUARE_WIDTH)
        return canvas_x, canvas_y

    def _draw_square(self, square, pos, agent):
        """
        Draw a colored square at the given location.

        If the agent is in that location, draw a smaller box to show it.
        """
        square_color = self._determine_square_color(square)
        self.canvas.create_rectangle(pos[0], pos[1], 
                                    pos[0] + self.SQUARE_WIDTH, 
                                    pos[1] + self.SQUARE_WIDTH,
                                    fill = square_color,
                                    outline = self.SQUARE_BORDER_COLOR,
                                    width = self.SQUARE_BORDER)

        if agent:
            a_x1 = pos[0] + self.SQUARE_WIDTH // 4 - self.SQUARE_BORDER
            a_y1 = pos[1] + self.SQUARE_WIDTH // 4 - self.SQUARE_BORDER
            a_x2 = pos[0] + self.SQUARE_WIDTH * 3 // 4
            a_y2 = pos[1] + self.SQUARE_WIDTH * 3 // 4
            self.canvas.create_rectangle(a_x1, a_y1, a_x2, a_y2,
                                        fill = self.SQUARE_AGENT_COLOR)

    def _determine_square_color(self, square):
        """
        Determine which color the square should be based on whether it is 
        terminal and its reward.
        """
        color = self.SQUARE_EMPTY_COLOR
        if square.terminal:
            color = self.lookup[square.reward]
        return color

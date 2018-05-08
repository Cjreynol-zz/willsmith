from games.gridworld.grid import Grid
from games.gridworld.gridworld import Gridworld
from games.gridworld.gridworld_direction import GridworldDirection


#####################################
#   Example transition functions
#####################################

def deterministic_transition(action):
    return [action], [1.0]

def shaky_transition(action):
    dirs = list(GridworldDirection)
    action_index = dirs.index(action)

    side1 = dirs[action_index - 1]
    side2 = dirs[action_index + 1]

    return [action, side1, side2], [0.6, 0.2, 0.2]


#####################################
#   Example Gridworlds
#####################################

simple_terminals = {(3, 1) : -50, (3, 2) : 500}
simple_living_reward = -1
simple_walls = {(1, 1)}
simple_grid_size = (4,3)
simple_start = (0, 0)

simple_grid = Grid(simple_terminals, simple_walls, simple_grid_size)

bridge_crossing_terminals = {(x, y) : -100
                                for x in range(1, 6) for y in [0, 2]}
bridge_crossing_terminals.update({(6, 1) : 10})
bridge_crossing_walls = {(0, 0), (0, 2), (6, 0), (6, 2)}
bridge_crossing_size = (7, 3)
bridge_crossing_start = (0, 1)

bridge_crossing_grid = Grid(bridge_crossing_terminals, bridge_crossing_walls, 
                                bridge_crossing_size)

def make_simple_gridworld(use_display):
    return Gridworld(simple_grid, simple_living_reward, deterministic_transition, 
                        simple_start, use_display)

def make_classic_gridworld(use_display):
    return Gridworld(simple_grid, simple_living_reward, shaky_transition, 
                        simple_start, use_display)

def make_bridge_crossing_gridworld(use_display):
    return Gridworld(bridge_crossing_grid, simple_living_reward, 
                        shaky_transition,bridge_crossing_start, use_display)

"""
The main executable that instantiates the simulator with the proper game and 
agents, as specified by the command-line arguments.

Run python main.py -h for the explanation of command-line arguments.

See README.md for a full-project breakdown.
"""


from argparse import ArgumentParser
from logging import FileHandler, Formatter, StreamHandler, DEBUG, INFO, getLogger

from agents.game.human_agent import HumanAgent
from agents.game.mcts_agent import MCTSAgent
from agents.game.random_agent import RandomAgent
from agents.mdp.gridworld_approx_qlearning_agent import GridworldApproxQLearningAgent

from games.havannah.havannah import Havannah
from games.ttt.nested_ttt import NestedTTT
from mdps.gridworld.gridworld_examples import make_simple_gridworld

from willsmith.simulator import Simulator


__version__ = "0.6.0"

HAVANNAH_LABELS = ["Havannah", "hav"]
NESTEDTTT_LABELS = ["NestedTTT", "ttt"]
GAME_AGENT_LABELS = ["mcts", "rand", "human"]
DEFAULT_GAME_AGENTS = ["mcts", "rand"]
DEFAULT_TIME_ALLOTTED = 0.5
DEFAULT_NUM_GAMES = 1

GRIDWORLD_LABELS = ["Gridworld", "grid"]
MDP_AGENT_LABELS = ["approxql"]
DEFAULT_TRIALS = 200
DEFAULT_MDP_AGENT = "approxql"

LOG_FILENAME = "delspooner.log"
FILE_LOG_FORMAT = "%(asctime)s::%(name)s::%(levelname)s::%(message)s"
STDERR_LOG_FORMAT = "%(message)s"
TIMESTAMP_FORMAT = "%H:%M:%S"

def create_logger(debug):
    """
    Create and configure the root logger that other loggers refer to.
    
    There are two handlers, one to write out to the file and one to write 
    to console.  
    
    The console handler is always set to only display INFO level messages to 
    prevent clutter.  The file handler can be optionally set to DEBUG level 
    to capture additional information.
    """
    stderr_level = INFO
    file_level = INFO
    if debug:
        file_level = DEBUG

    logger = getLogger()
    logger.setLevel(file_level)

    file_handler = FileHandler(LOG_FILENAME, mode = 'w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(Formatter(FILE_LOG_FORMAT, TIMESTAMP_FORMAT))

    stderr_handler = StreamHandler()
    stderr_handler.setLevel(stderr_level)
    stderr_handler.setFormatter(Formatter(STDERR_LOG_FORMAT, TIMESTAMP_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)

def create_parser():
    """
    """
    parser = ArgumentParser(description = "Run agents through simulations")

    parser.add_argument("-c", "--console_display", action = "store_true",
                        default = False,
                        help = "Render the game on the command-line")
    parser.add_argument("-d", "--debug", action = "store_true",
                        default = False,
                        help = "Turn on debug-level logging")
    parser.add_argument("-r", "--no_display", action = "store_true", 
                        default = False,
                        help = "Do not display the game on each turn.")
    parser.add_argument("-v", "--version", action = "version", 
                        version = "willsmith " + __version__)

    subparser = parser.add_subparsers(dest = "sim_type",
                                        help = "Choose the type of simulation to run")

    game_parser = subparser.add_parser("game",
                                        help = "Simulate an adversarial game")
    game_parser.add_argument("game_choice", type = str,
                                choices = NESTEDTTT_LABELS + HAVANNAH_LABELS,
                                help = "The game for the agents to play")
    game_parser.add_argument("-a", "--agents", nargs = '*',
                                default = DEFAULT_GAME_AGENTS,
                                choices = GAME_AGENT_LABELS,
                                help = "Agent types")
    game_parser.add_argument("-n", "--num_games", type = int,
                                default = DEFAULT_NUM_GAMES,
                                help = "Number of successive game simulations to run.")
    game_parser.add_argument("-t", "--time_allotted", type = float,
                                default = DEFAULT_TIME_ALLOTTED,
                                help = "Time allotted for agent moves")

    mdp_parser = subparser.add_parser("mdp", help = "Simulate an MDP")
    mdp_parser.add_argument("mdp_choice", type = str,
                            choices = GRIDWORLD_LABELS,
                            help = "The MDP for the agent to learn")
    mdp_parser.add_argument("-a", "--agent_choice", type = str,
                            default = DEFAULT_MDP_AGENT,
                            choices = MDP_AGENT_LABELS,
                            help = "Agent types")
    mdp_parser.add_argument("-n", "--num_trials", type = int,
                            default = DEFAULT_TRIALS,
                            help = "The number of trials to simulate")
    mdp_parser.add_argument("-l", "--learning_rate", type = float,
                            help = "The initial learning rate of the agent")
    mdp_parser.add_argument("-e", "--exploration_rate", type = float,
                            help = "The initial exploration rate of the agent")
    mdp_parser.add_argument("-g", "--discount", type = float,
                            help = "The discount applied to future rewards")

    return parser

def lookup_agent(num, agent_str):
    lookup = {"mcts" : MCTSAgent, "rand" : RandomAgent, "human" : HumanAgent}
    try:
        agent_class = lookup[agent_str]
    except KeyError:
        raise RuntimeError("Unexpected agent_class string type.")

    getLogger().debug("Agent {} is {}".format(num, agent_class.__name__))
    return agent_class

def lookup_mdp_agent(agent_str, mdp):
    lookup = {("approxql", "Gridworld") : GridworldApproxQLearningAgent}
    try:
        agent_class = lookup[(agent_str, mdp.__class__.__name__)]
    except KeyError:
        raise RuntimeError("Unexpected agent string type.")
    return agent_class

def lookup_game(game_str):
    if game_str in NESTEDTTT_LABELS:
        game_class = NestedTTT
    elif game_str in HAVANNAH_LABELS:
        game_class = Havannah
    else:
        raise RuntimeError("Unexpected game type.")

    getLogger().info("Game is {}".format(game_class.__name__))
    return game_class

def lookup_mdp(mdp_str):
    if mdp_str in GRIDWORLD_LABELS:
        mdp_gen = make_simple_gridworld
    else:
        raise RuntimeError("Unexpected mdp type.")

    getLogger().info("MDP generator is {}".format(mdp_gen.__name__))
    return mdp_gen

def determine_display(no_display, console_display):
    gui = True
    if no_display:
        gui = None
        getLogger().debug("Non-display option chosen")
    elif console_display:
        gui = False
        getLogger().debug("Console display option chosen")
    return gui

def process_game_args(args, use_gui):
    """
    """
    game = lookup_game(args.game_choice)(use_gui)
    agent_classes = [lookup_agent(i, agent_str) 
                        for i, agent_str in enumerate(args.agents)]
    agents = [(agent(i, use_gui) 
                            if agent is not HumanAgent 
                            else agent(i, use_gui, game.ACTION))
                            for i, agent in enumerate(agent_classes)]

    getLogger().debug("Agents have {} seconds per turn".format(args.time_allotted))
    getLogger().debug("{} game(s) will be played".format(args.num_games))
    return [game, agents, args.time_allotted, args.num_games]

def process_mdp_args(args, use_gui):
    """
    """
    mdp = lookup_mdp(args.mdp_choice)(use_gui)
    agent = lookup_mdp_agent(args.agent_choice, mdp)(mdp.get_action_space(), 
                                                        args.learning_rate,
                                                        args.discount,
                                                        args.exploration_rate)

    getLogger().debug("Agents will start with:  learning rate = {}; discount = {}; exploration_rate = {}".format(args.learning_rate, args.discount, args.exploration_rate))
    getLogger().debug("{} trial(s) will be run".format(args.num_trials))
    return [mdp, agent, args.num_trials]

def process_args(args):
    """
    """
    use_gui = determine_display(args.no_display, args.console_display)

    if args.sim_type == "game":
        sim_args = process_game_args(args, use_gui)
    elif args.sim_type == "mdp":
        sim_args = process_mdp_args(args, use_gui)
    else:
        raise RuntimeError("Unexpected simulation type.")

    return sim_args

def main():
    """
    """
    parser = create_parser()
    args = parser.parse_args()
    create_logger(args.debug)

    sim_args = process_args(args)
    if args.sim_type == "game":
        Simulator.run_games(*sim_args)
    elif args.sim_type == "mdp":
        Simulator.run_mdp(*sim_args)

if __name__ == "__main__":
    main()

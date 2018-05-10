from math import log, sqrt
from random import choice
from time import time

from agents.displays.mcts_display import MCTSDisplay

from willsmith.game_agent import GameAgent


class MCTSAgent(GameAgent):
    """
    Agent that learns action values based on Monte Carlo Tree Search.

    Computes as many runs as possible in the time allotted, where a run
    consists of the following stages:

        Selection       -   Starting at root, choose actions until a new leaf 
                            can be created or the end of the tree is reached
        Expansion       -   Create new leaf node in game tree, unless at 
                            the end of the tree
        Simulation      -   Play out game until reaching a terminal state
        Backpropagation -   Update win/trial counters from the start of the 
                            playout up to the rood

    Then the most played action is chosen.

    The agents internal game state is stored as a tree of nodes, where the
    edges are actions and the nodes are the wins/total trials from the
    perspective of the agent they represent.
    """

    GUI_DISPLAY = MCTSDisplay
    EXPLORATION_PARAM = sqrt(2)

    def __init__(self, agent_id, use_gui, tree_policy = None,
                    simulation_policy = None):
        """
        Put the agent in an initial state and assign its policies.
        """
        super().__init__(agent_id, use_gui)
        self._reset()

        self._tree_policy = tree_policy
        if self._tree_policy is None:
            self._tree_policy = self._UCT

        self._simulation_policy = simulation_policy
        if self._simulation_policy is None:
            self._simulation_policy = self._random_simulation

    def _reset(self):
        """
        Initialize the node tree and the debug attributes.
        """
        self.root = self.Node(None, None)

        self.playout_total = 0
        self.action_node = None

    def search(self, state, allotted_time):
        """
        Run as many playouts as possible in the allotted time and return the 
        most played action.
        """
        playouts = 0

        start_time = time()
        while time() - start_time < allotted_time:
            current_state = state.copy()
            selected_node = self._selection(current_state)
            new_node = self._expansion(current_state, selected_node)
            winning_id = self._simulation(current_state)
            self._backpropagation(winning_id, new_node)
            playouts += 1

        max_action = self.root.max_trials()
        # debug info
        self.playout_total = playouts
        self.action_node = self.root.get_child(max_action)

        return max_action

    def _take_action(self, action):
        """
        Traverse the tree along the edge represented by the action, moving 
        the root reference.

        If that action has not been expanded, then the tree is reset to an 
        initial state.  This happens when an adversary takes an action that 
        has not yet been explored.
        """
        try:
            self.root = self.root.get_child(action)
        except KeyError:
            self.root = self.Node(None, None)

    def _selection(self, state):
        """
        Progress through the tree of Nodes, starting at the root, until 
        either a leaf is found or there are unexplored actions at the current 
        level.
        """
        node = self.root
        unexplored_actions = len(state.get_legal_actions()) > len(node.children.keys())

        while not unexplored_actions and node.has_children():
            action = self._tree_policy(node, state)
            node = node.get_child(action)

            state.take_action(action)
            unexplored_actions = len(state.get_legal_actions()) > len(node.children.keys())

        return node

    def _UCT(self, node, state):
        """
        Choose an action based on an exploitation vs exploration function
        that expresses node value as:

        (exploitation)
        num wins at node / num trials node
        +
        (exploration)
        exploration parameter 
            * sqrt(ln(num trials at node) / num trials at child)
        """
        valid_actions = state.get_legal_actions()

        results = {}
        for action in valid_actions:
            child_node = node.get_child(action)
            value_estimate = child_node.value_estimate()
            exploration_estimate = self.EXPLORATION_PARAM * sqrt(log(node.trials) / child_node.trials)
            results[action] = value_estimate + exploration_estimate
        return max(results.keys(), key=results.get)

    def _expansion(self, state, node):
        """
        Create a new leaf node in the Node tree, provided there are still 
        legal actions.

        Makes a random choice of the unexplored actions, adds that Node to
        the tree, and syncs up the current game state with the node tree.

        In the case where there are no more legal actions left, this method 
        simply returns the given node.
        """
        possible_actions = [action for action in state.get_legal_actions() if action not in node.children]

        if possible_actions:
            action = choice(possible_actions)
            new_child = self.Node(node, state.current_agent_id)
            node.add_child(action, new_child)
            state.take_action(action)
        else:
            new_child = node

        return new_child

    def _simulation(self, state):
        """
        Play out the game until it is terminal  and returns the winning agent 
        id.
        """
        while not state.is_terminal():
            action = self._simulation_policy(state)
            state.take_action(action)
        return state.get_winning_id()

    def _random_simulation(self, state):
        """
        Make random choices for actions regardless of the state.

        This simulation strategy is sometimes called a "light playout".  It 
        has no computational overhead, but does not take advantage of any 
        domain knowledge about the game.
        """
        action = state.generate_random_action()
        return action

    def _backpropagation(self, winning_id, node):
        """
        Update from the given node until the root with the simulation result.
        """
        while node is not None:
            node.update_node(winning_id)
            node = node.parent

    def __str__(self):
        return "playouts {} | node {} | tree max depth {}".format(self.playout_total, self.action_node, self.root.depth())


    class Node:
        """
        Used internally by the agent to model the game tree.

        Each node stores the win percentage of the given agent_id from its 
        position in the game tree.
        """

        def __init__(self, parent, agent_id):
            self.parent = parent
            self.children = dict()  # action : Node
            self.agent_id = agent_id

            self.wins = 0
            self.trials = 0

        def update_node(self, winning_id):
            """
            Update node using simulation result during backpropagation step.
            """
            if self.agent_id is not None and winning_id == self.agent_id:
                self.wins += 1
            self.trials += 1

        def value_estimate(self):
            return self.wins / self.trials

        def max_trials(self):
            """
            Return the child node with the maximum number of trials.
            """
            value_func = lambda x: self.get_child(x).trials
            return max(self.children, key=value_func)

        def add_child(self, action, node):
            self.children[action] = node

        def get_child(self, action):
            return self.children[action]

        def has_children(self):
            return bool(self.children)

        def depth(self):
            depth = 0
            if self.has_children():
                depth = 1 + max([child.depth() for child in self.children.values()])
            return depth

        def __str__(self):
            return "ID{},{}/{}".format(self.agent_id, self.wins, self.trials)

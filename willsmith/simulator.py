"""
This simulator module contains the functions responsible for managing agents 
while they play games and learn MDPs.  
"""


from logging import getLogger


def run_games(game, agents, time_allowed, num_games):
    """
    Play the agents through the given game.
    """
    if len(agents) != game.NUM_PLAYERS:
        raise RuntimeError("Incorrect number of agents for game type.")

    for i in range(num_games):
        getLogger(__name__).info("Game {}/{}".format(i + 1, num_games))
        _run_game(game, agents, time_allowed)
    getLogger(__name__).info("Games complete")
    input("\nPress enter key to end.")

def run_mdp(mdp, agent, num_trials):
    """
    Run the agent through a number of trials on the given MDP.
    """
    for i in range(num_trials):
        getLogger(__name__).info("Trial {}/{}".format(i + 1, num_trials))
        num_steps = _run_trial(mdp, agent)
        getLogger(__name__).debug("reward:{}; timesteps:{}".format(mdp.total_reward, mdp.timesteps))

    getLogger(__name__).debug("Final agent weights: {}".format(agent.weights))
    getLogger(__name__).info("Trials complete.")
    input("\nPress enter key to end.")

def _run_game(game, agents, time_allowed):
    """
    """
    game.reset()
    for agent in agents:
        agent.reset()
        getLogger(__name__).debug("Agent {} start {}".format(agent.agent_id, agent))

    while not game.is_terminal():
        current_agent = agents[game.current_agent_id]
        action = current_agent.search(game.copy(), time_allowed)
        getLogger(__name__).debug("Agent {} {}".format(current_agent.agent_id, current_agent))
        _advance_by_action(game, agents, action)

    getLogger(__name__).info("Winning agent is {}".format(game.get_winning_id()))
    getLogger(__name__).debug("Final state\n{}".format(game))

def _advance_by_action(game, agents, action):
    """
    Update the game and agents with the given action.
    """
    getLogger(__name__).debug("Agent {} action {}".format(game.current_agent_id, action))
    agent_id_for_action = game.current_agent_id

    game.take_action(action)
    for agent in agents:
        agent.take_action(action, agent.agent_id == agent_id_for_action)

def _run_trial(mdp, agent):
    """
    """
    mdp.reset()
    prev_state = None

    while not mdp.is_terminal():
        action = agent.get_next_action(mdp.copy())

        prev_state = mdp.copy()
        reward, terminal = mdp.step(action)

        agent.update(prev_state, mdp, reward, action, terminal)

    return mdp.timesteps

# Willsmith

A framework for creating, testing, and comparing AI agents in different 
domains.  The goals of the project are:   
- to provide a convenient scaffolding for quickly developing new agents and 
testing environments
- to provide reference implementations of agents to compare new 
implemenations against
- to learn about more types of AI agents, games, etc by implementing them

Currently, the project supports adversarial games and Markov Decision 
processes and their agents.  See the abstract base classes for these in the 
`willsmith` directory for the API expected for new implementations.  Provided 
example agents, games, and mdps can be found in their respective directories.  

Work is being done with [John Bourassa](https://github.com/johink).  

## To Run

Tested on Python version 3.6, but it should work on any recent version of 
Python 3.  

Run `python main.py -h`, `python main.py game -h`, and 
`python main.py mdp -h` to see the full help documentation for running a 
simulation and what options are available.  

An example simulation of our Monte Carlo Tree Search agent playing against a 
Random agent in a game of Nested Tic-Tac-Toe can be seen by running 
`python main.py game ttt`.  

## Currently included

### Games:
- NestedTTT
> A game of Nested Tic-Tac-Toe, where each square on the outer board 
> is another Tic-Tac-Toe board.
> 
> The game is played as standard Tic-Tac-Toe, except there are 9 inner 
> boards to make moves on.  Winning an inner board claims that space for 
> the winner on the outer board.  Draws result in a square that does not 
> count for either side.  Winning the game requires winning 3 inner boards 
> in a row, forming a win on the outer board.

- Havannah
> The game of Havannah is played on a hex board that is typically 10 hexes to 
> a side.
> 
> Players alternate turns placing stones, or coloring hexes in our 
> case, in previously unchosen hexes.  Play continues until one player has 
> formed one of three different winning configurations:  a ring, fork, or 
> bridge.

### Game agents:
- MCTSAgent
> An agent that learns action values based on Monte Carlo Tree Search.
> 
> Computes as many runs as possible in the time allotted, where a run
> consists of the following stages:
> 
> > Selection       -   Starting at root, choose actions until a new leaf 
> >                     can be created or the end of the tree is reached
> > Expansion       -   Create a new leaf node in game tree, unless the 
> >                     tree is already at a terminal state
> > Simulation      -   Play out game until reaching a terminal state
> > Backpropagation -   Update win/trial counters from the start of the 
> >                     playout up to the rood
> 
> Then the most played action is chosen.

- RandomAgent
> Agent that chooses random actions regardless of the game state.

- HumanAgent
> Agent that relies on user input to make action choices.

### MDPs:
- Gridworld
> The standard example of a Markov Decision Process, where an agent moves 
> around in different squares, gathering living rewards until a terminal 
> state is reached.

### MDP agents:
- ApproxQLearningAgent
> An agent that relies on appoximate Q-learning to learn a policy.
> 
> This agent utilizes given feature functions to determine values for the 
> states of the MDP.  These functions are weighted, and the weights are 
> updated after each action.  
> 
> Given the proper parameters and enough trials of the MDP, this agent's 
> weights will converge on a set of values that cause the agent to act in 
> accordance with the MDP's optimal policy.

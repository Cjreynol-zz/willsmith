from nested_ttt import NestedTTT
from mcts_agent import MCTSAgent


game = NestedTTT([0,1])
p1 = MCTSAgent(game, 0)

while not game.is_terminal():
    action = p1.search()
    p1.take_action(action)
    game.take_action(action)
    
    if not game.is_terminal():
        action = game.generate_random_action()
        p1.take_action(action)
        game.take_action(action)


    print(chr(27) + "[2J")      # clears the terminal
    print(game)

    input()

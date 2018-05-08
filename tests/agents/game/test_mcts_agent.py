from unittest import TestCase
from unittest.mock import patch

from agents.game.mcts_agent import MCTSAgent


class TestMCTSAgent(TestCase):

    def setUp(self):
        self.agent = MCTSAgent(0, False)

    @patch("willsmith.game.Game")
    def test_expansion_in_terminal_state_noop(self, game):
        game.is_terminal.return_value = True
        current_node = None
        returned_node = self.agent._expansion(game, current_node)
        self.assertEqual(current_node, returned_node)

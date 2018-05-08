from unittest import TestCase


class MDPTestCase(TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_mdp_equality(self):
        other_mdp = self.mdp.copy()
        self.assertEqual(self.mdp, other_mdp)

        self.mdp.step(self.test_action)
        self.assertNotEqual(self.mdp, other_mdp)

        other_mdp.step(self.test_action)
        self.assertEqual(self.mdp, other_mdp)

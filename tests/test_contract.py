import unittest

from dcss_env import DCSSEnv, VARIANTS
from dcss_gym import scenarios


class _Screen:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class ContractTests(unittest.TestCase):
    def test_gym_observations_are_player_legal(self):
        for case in scenarios("b"):
            obs = case.observation
            self.assertEqual(len(obs.action_names), len(obs.action_mask))
            self.assertTrue(any(obs.action_mask))
            self.assertTrue(obs.action_mask[obs.action_names.index(case.expected_action)])

    def test_picks_are_masked_outside_a_menu(self):
        env = object.__new__(DCSSEnv)
        env.spec = VARIANTS["b"]
        env.n_actions = len(env.spec)
        env.menu_open = None
        env.c = _Screen("")
        mask = env.action_mask()
        for i, (name, _key) in enumerate(env.spec):
            is_menu_pick = name.startswith("pick") and name[-1].isdigit()
            self.assertEqual(mask[i], not is_menu_pick)

    def test_menu_allows_only_existing_choices_or_cancel(self):
        env = object.__new__(DCSSEnv)
        env.spec = VARIANTS["b"]
        env.n_actions = len(env.spec)
        env.menu_open = "wield"
        env.c = _Screen("a - a +2 war axe\nb - a +0 club\n")
        mask = env.action_mask()
        legal = [name for (name, _key), ok in zip(env.spec, mask) if ok]
        self.assertEqual(legal, ["escape", "pick1", "pick2"])


if __name__ == "__main__":
    unittest.main()

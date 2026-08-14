import random
import unittest

import torch

from r2d2 import PrioritizedEpisodeReplay, RecurrentQ, Transition
from train_rl import OBS_LEN, Policy


def transition(value, done=False):
    obs = torch.full((OBS_LEN,), value, dtype=torch.long)
    mask = torch.ones(3, dtype=torch.bool)
    return Transition(obs, -1, 0, 0.0, done, mask, obs.clone(), mask.clone())


class R2D2Tests(unittest.TestCase):
    def test_recurrent_q_shapes_and_legal_mask(self):
        model = RecurrentQ(3)
        obs = torch.full((2, OBS_LEN), 32, dtype=torch.long)
        hidden = model.initial_state(2)
        mask = torch.tensor([[True, False, True], [False, True, True]])
        q, next_hidden = model.step(
            obs, torch.tensor([-1, 1]), hidden, mask)
        self.assertEqual(q.shape, (2, 3))
        self.assertEqual(next_hidden.shape, hidden.shape)
        self.assertLess(q[0, 1], -1e20)
        self.assertLess(q[1, 0], -1e20)

    def test_spatial_warm_start_preserves_action_probabilities(self):
        torch.manual_seed(5)
        spatial = Policy(3)
        recurrent = RecurrentQ(3)
        recurrent.warm_start_spatial(spatial.state_dict())
        observation = torch.full((2, OBS_LEN), 32, dtype=torch.long)
        with torch.no_grad():
            logits, _value = spatial(observation)
            q, _hidden = recurrent.step(
                observation, torch.tensor([-1, -1]),
                recurrent.initial_state(2))
        self.assertTrue(torch.allclose(
            torch.softmax(logits, -1), torch.softmax(q, -1), atol=1e-6))

    def test_episode_replay_supplies_burn_in_context(self):
        random.seed(2)
        replay = PrioritizedEpisodeReplay(capacity_steps=20)
        replay.add([transition(i, i == 7) for i in range(8)])
        samples = replay.sample(4, burn_in=3, unroll=2)
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(0 <= sample.burn_in <= 3 for sample in samples))
        self.assertTrue(all(sample.transitions for sample in samples))

    def test_replay_capacity_drops_whole_old_episodes(self):
        replay = PrioritizedEpisodeReplay(capacity_steps=5)
        replay.add([transition(1) for _ in range(4)])
        replay.add([transition(2) for _ in range(4)])
        self.assertEqual(len(replay.episodes), 1)
        self.assertEqual(len(replay), 4)


if __name__ == "__main__":
    unittest.main()

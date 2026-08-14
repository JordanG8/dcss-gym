import unittest

from async_actors import InferenceRequest


class AsyncActorTests(unittest.TestCase):
    def test_request_carries_only_public_observation_contract(self):
        request = InferenceRequest(
            actor=3, observation="visible screen", action_mask=[True, False],
            previous_action=0, reset=False)
        self.assertEqual(request.observation, "visible screen")
        self.assertEqual(request.action_mask, [True, False])
        self.assertFalse(hasattr(request, "game_state"))


if __name__ == "__main__":
    unittest.main()

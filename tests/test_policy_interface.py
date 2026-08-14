import unittest

import torch

from dcss_env import BASE, MOBILITY, VARIANTS
from train_rl import BASE_VOCAB, CROP, Policy, encode, load_policy_state
from webtiles_policy_agent import PolicyView, visible_action_mask


class PolicyInterfaceTests(unittest.TestCase):
    def test_variant_c_appends_mobility_without_reordering_old_actions(self):
        self.assertEqual(VARIANTS["c"][:len(BASE)], BASE)
        self.assertEqual([name for name, _ in MOBILITY], [
            "move_n", "move_ne", "move_e", "move_se", "move_s",
            "move_sw", "move_w", "move_nw", "wait",
        ])

    def test_explicit_hostile_cell_uses_second_vocabulary_plane(self):
        rows = [list(" " * 80) for _ in range(24)]
        rows[8][20] = "@"
        rows[8][21] = "g"
        screen = "\n".join("".join(row) for row in rows)
        encoded = encode(screen, hostile_cells={(21, 8)})
        # Full terminal grid follows the 15x15 egocentric crop.
        self.assertEqual(
            encoded[CROP * CROP + 8 * 80 + 21], BASE_VOCAB + ord("g"))
        self.assertEqual(encoded[CROP * CROP + 8 * 80 + 20], ord("@"))

    def test_checkpoint_migration_preserves_old_actor_and_embeddings(self):
        torch.manual_seed(3)
        old = Policy(7)
        new = Policy(16)
        legacy = old.state_dict()
        legacy["emb.weight"] = legacy["emb.weight"][:BASE_VOCAB].clone()
        report = load_policy_state(new, legacy)
        self.assertIn("actor.weight", report["expanded"])
        self.assertIn("emb.weight", report["expanded"])
        self.assertTrue(torch.equal(new.actor.weight[:7], old.actor.weight))
        self.assertTrue(torch.equal(new.actor.bias[:7], old.actor.bias))
        self.assertTrue(torch.equal(
            new.emb.weight[:BASE_VOCAB], old.emb.weight[:BASE_VOCAB]))
        self.assertTrue(torch.equal(
            new.emb.weight[BASE_VOCAB:], old.emb.weight[:BASE_VOCAB]))
        self.assertTrue(torch.all(
            new.actor.bias[7:] < old.actor.bias.min()).item())

    def test_id_only_monster_delta_keeps_visible_hostile_attitude(self):
        view = PolicyView()
        view.apply_map({"cells": [{"x": 4, "y": 5, "g": "g",
                                    "mon": {"id": 17, "att": 0}}]})
        self.assertIn((4, 5), view.hostiles)
        view.apply_map({"cells": [{"x": 4, "y": 5,
                                    "mon": {"id": 17}}]})
        self.assertIn((4, 5), view.hostiles)

    def test_empty_glyph_delta_erases_stale_monster(self):
        view = PolicyView()
        view.player = {"pos": {"x": 4, "y": 5}}
        view.apply_map({"cells": [{"x": 5, "y": 5, "g": "g",
                                    "mon": {"id": 17, "att": 0}}]})
        self.assertEqual(view.monsters_near(), 1)
        view.apply_map({"cells": [{"x": 5, "y": 5, "g": ""}]})
        self.assertNotIn((5, 5), view.grid)
        self.assertNotIn((5, 5), view.hostiles)
        self.assertEqual(view.monsters_near(), 0)

    def test_null_monster_delta_erases_stale_monster_without_glyph(self):
        view = PolicyView()
        view.player = {"pos": {"x": 5, "y": 5}}
        view.apply_map({"cells": [{"x": 5, "y": 5, "g": "g",
                                    "mon": {"id": 7, "att": 0}}]})
        self.assertEqual(view.monsters_near(), 1)

        view.apply_map({"cells": [{"x": 5, "y": 5, "mon": None}]})

        self.assertNotIn((5, 5), view.grid)
        self.assertNotIn((5, 5), view.hostiles)
        self.assertEqual(view.monsters_near(), 0)

    def test_nonhostile_monster_metadata_overrides_letter_fallback(self):
        view = PolicyView()
        view.player = {"pos": {"x": 5, "y": 5}}
        view.apply_map({"cells": [{"x": 5, "y": 5, "g": "g",
                                    "mon": {"id": 9, "att": 4}}]})

        self.assertEqual(view.monsters_near(), 0)

    def test_mobility_prevents_all_false_visible_mask(self):
        view = PolicyView()
        view.player = {"pos": {"x": 1, "y": 1}, "hp": 5, "hp_max": 10,
                       "status": [{"text": "Berserk"}]}
        view.hostiles.add((2, 1))
        names = [name for name, _ in VARIANTS["c"]]
        mask, _ = visible_action_mask(names, view, (1, 1, 1, 1), {})
        self.assertTrue(any(mask))
        self.assertTrue(all(mask[names.index(name)]
                            for name, _ in MOBILITY))

    def test_rejected_movement_is_scoped_to_unchanged_visible_state(self):
        view = PolicyView()
        view.player = {"depth": 1, "turn": 8, "hp": 20, "hp_max": 20,
                       "pos": {"x": 3, "y": 4}, "status": []}
        names = [name for name, _ in VARIANTS["c"]]
        signature = (1, 8, 3, 4)
        mask, _ = visible_action_mask(
            names, view, signature, {"move_n": signature})
        self.assertFalse(mask[names.index("move_n")])
        self.assertTrue(mask[names.index("move_e")])
        mask, _ = visible_action_mask(
            names, view, signature, {"autofight": signature})
        self.assertFalse(mask[names.index("autofight")])
        self.assertTrue(mask[names.index("move_e")])
        changed = (1, 9, 3, 4)
        mask, _ = visible_action_mask(
            names, view, changed, {"move_n": signature})
        self.assertTrue(mask[names.index("move_n")])


if __name__ == "__main__":
    unittest.main()

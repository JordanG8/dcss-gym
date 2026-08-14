import tempfile
import unittest
from pathlib import Path

import torch

from checkpointing import (atomic_torch_save, manifest_path,
                           publish_manifest, read_manifest, sha256_file)


class CheckpointingTests(unittest.TestCase):
    def test_checkpoint_and_manifest_are_complete_and_hashed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = atomic_torch_save(
                {"weight": torch.tensor([1.0, 2.0])}, root / "policy.pt")
            manifest = manifest_path(root, "c", "candidate")
            published = publish_manifest(
                checkpoint, manifest, variant="c", channel="candidate",
                update=12, action_names=["wait"], metrics={"depth": 2})

            loaded = read_manifest(manifest)
            self.assertEqual(loaded["sha256"], sha256_file(checkpoint))
            self.assertEqual(loaded["sha256"], published["sha256"])
            self.assertEqual(loaded["update"], 12)
            self.assertEqual(loaded["metrics"]["depth"], 2)
            self.assertTrue(torch.equal(
                torch.load(checkpoint, weights_only=True)["weight"],
                torch.tensor([1.0, 2.0])))

    def test_unknown_channel_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                manifest_path(Path(folder), "c", "nightly")


if __name__ == "__main__":
    unittest.main()

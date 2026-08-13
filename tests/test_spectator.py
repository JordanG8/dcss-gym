import json
import tempfile
import unittest
from pathlib import Path

from spectator import load_frames, replay_files


class SpectatorTests(unittest.TestCase):
    def test_replays_are_listed_and_loaded_by_safe_stem(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / "rl_replays_b"
            folder.mkdir()
            path = folder / "episode-01.jsonl"
            path.write_text(json.dumps({"screen": "first"}) + "\nnot-json\n" +
                            json.dumps({"screen": "second"}) + "\n",
                            encoding="utf-8")
            self.assertEqual(replay_files(root), [path])
            self.assertEqual([f["screen"] for f in load_frames(root, "episode-01")],
                             ["first", "second"])
            self.assertEqual(load_frames(root, "../episode-01"), [])


if __name__ == "__main__":
    unittest.main()

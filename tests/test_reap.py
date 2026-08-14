import unittest

from reap import is_owner_command


class ReapTests(unittest.TestCase):
    def test_training_and_webtiles_parents_own_their_crawl_processes(self):
        self.assertTrue(is_owner_command(
            "/root/pty-venv/bin/python train_r2d2.py --envs 32"))
        self.assertTrue(is_owner_command(
            "/root/pty-venv/bin/python train_async_r2d2.py --envs 24"))
        self.assertTrue(is_owner_command(
            "/root/webtiles-venv/bin/python webserver/server.py"))

    def test_init_or_unrelated_python_does_not_own_crawl(self):
        self.assertFalse(is_owner_command("/init"))
        self.assertFalse(is_owner_command("python dashboard.py"))


if __name__ == "__main__":
    unittest.main()

"""Install and smoke-test the native DCSS Gym Sprint map.

The map file lives in this repository.  Installation is explicit because it
adds a single `.des` file to a local Crawl source checkout and rebuilds Crawl's
map cache; neither operation is needed for ordinary PPO/Gym fixture tests.

    /root/pty-venv/bin/python native_gym.py --install
    /root/pty-venv/bin/python native_gym.py --smoke
"""
import argparse
import shutil
import subprocess
from pathlib import Path

from dcss_env import CRAWL_DIR, DCSSEnv


HERE = Path(__file__).parent
MAP_NAME = "dcss_gym_equipment"
MAP_SOURCE = HERE / "gym_maps" / f"{MAP_NAME}.des"


def install(crawl_dir=CRAWL_DIR):
    target = Path(crawl_dir) / "dat" / "des" / "sprint" / MAP_SOURCE.name
    if not MAP_SOURCE.exists():
        raise FileNotFoundError(MAP_SOURCE)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MAP_SOURCE, target)
    subprocess.run(["./crawl", "-builddb"], cwd=crawl_dir, check=True)
    print(f"installed {MAP_NAME} -> {target}")


def smoke():
    env = DCSSEnv(env_id=98_765, variant="b", seed=424242,
                  max_steps=20, sprint_map=MAP_NAME)
    try:
        screen = env.reset()
        public = env.public_observation()
        print(screen)
        print("\npublic observation:")
        print({"status": dict(public.status), "mask": list(public.action_mask),
               "actions": list(public.action_names)})
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.install and not args.smoke:
        ap.error("choose --install and/or --smoke")
    if args.install:
        install()
    if args.smoke:
        smoke()


if __name__ == "__main__":
    main()

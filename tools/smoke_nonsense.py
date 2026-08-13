"""Check the nonsense detector fires on real mistakes and not on good play.

A penalty that mislabels sensible actions is worse than no penalty: it teaches
the policy to avoid the RIGHT move. So drive deliberate mistakes and confirm
each one is caught, then confirm a legitimate rest is NOT caught.

    /root/pty-venv/bin/python tools/smoke_nonsense.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcss_env import VARIANTS, DCSSEnv          # noqa: E402


def main():
    names = [n for n, _ in VARIANTS["a"]]
    idx = {n: i for i, n in enumerate(names)}
    env = DCSSEnv(env_id=98, max_steps=400, variant="a")
    env.reset()

    results = []

    def do(action, label):
        before = env.nonsense
        scr, r, _, info = env.step(idx[action])
        msg = [l.strip() for l in scr.split("\n")[17:] if l.strip()]
        results.append((label, round(r, 3), env.nonsense > before,
                        env.monsters_visible(scr),
                        (msg[0] if msg else "")[:52]))

    # Deliberate mistakes, before exploring: nothing to fight, no stairs known.
    do("autofight", "autofight, no enemy")
    do("descend", "descend, not on stairs")
    do("pickup", "pickup, nothing here")
    do("wear", "wear, nothing to wear")

    # Explore to completion, then explore again — that is a wasted key.
    for _ in range(60):
        env.step(idx["explore"])
    do("explore", "explore when done")

    # A rest with no monster around is a legitimate move and must NOT be
    # flagged; a rest with one visible must be.
    do("rest", "rest (monster state shown ->)")

    print(f"{'case':30s} {'reward':>8}  {'flagged':>7}  {'mons':>5}  message")
    for label, r, flagged, mons, msg in results:
        print(f"{label:30s} {r:+8.3f}  {str(flagged):>7}  {str(mons):>5}  {msg}")
    print(f"\ntotal nonsense flagged: {env.nonsense}   "
          f"travel refused: {env.travel_refused}   depth: {env.max_depth}")
    env.close()


if __name__ == "__main__":
    main()

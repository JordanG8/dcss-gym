"""Prove the new actions do what their names claim.

Keystroke assumptions are the thing most likely to be silently wrong here — an
action that does nothing looks identical to a bad decision, and the agent would
spend hours learning not to press it. So check the game's own messages.

    /root/pty-venv/bin/python smoke_actions.py
"""
import re

from dcss_env import VARIANTS, DCSSEnv

RE_ANY_BERSERK = re.compile(r"berserk|exhaust", re.I)
RE_ANY_EQUIP = re.compile(r"wear|wield|wielding|You have nothing|not wearing",
                          re.I)


def main():
    names = [n for n, _ in VARIANTS["a"]]
    idx = {n: i for i, n in enumerate(names)}
    env = DCSSEnv(env_id=90, max_steps=200, variant="a")
    env.reset()

    # Explore a little first so there is something to fight and pick up.
    for _ in range(25):
        env.step(idx["explore"])

    checks = {}

    def msgs(scr):
        """Just the message area — rows 17+, where crawl prints what happened."""
        return " | ".join(l.strip() for l in scr.split("\n")[17:] if l.strip())

    scr, r, _, info = env.step(idx["berserk"])
    checks["berserk"] = (round(r, 3), msgs(scr)[:100] or "(no message)")

    scr, r, _, _ = env.step(idx["berserk"])   # immediately again: should be refused
    checks["berserk again"] = (round(r, 3), msgs(scr)[:100] or "(no message)")

    scr, r, _, _ = env.step(idx["pickup"])
    checks["pickup"] = (round(r, 3), msgs(scr)[:100] or "(no message)")

    scr, r, _, _ = env.step(idx["wear"])
    checks["wear"] = (round(r, 3), msgs(scr)[:100] or "(no message)")

    scr, r, _, info = env.step(idx["wield"])
    checks["wield"] = (round(r, 3), msgs(scr)[:100] or "(no message)")

    # The critical regression: the agent must still be holding its axe.
    env.c.send("i")
    env.c.drain(quiet=0.2, timeout=5)
    inv = env.c.text()
    env.c.send("\x1b")
    env.c.drain(quiet=0.1, timeout=3)
    held = [l.strip() for l in inv.split("\n") if "(weapon)" in l.lower()]
    checks["still armed"] = (0.0, held[0][:100] if held else "NOTHING WIELDED")

    for k, (r, line) in checks.items():
        print(f"{k:14s} reward={r:+.3f}  {line}")
    # ac must be a number, not None: if the status panel never parsed, the
    # equipment reward is silently dead again in exactly the old way.
    print(f"\nac={env.ac} (baseline {env.max_ac}, +{env.ac_gained} earned) "
          f"equips={env.equips} berserks={env.berserks} "
          f"wasted={env.berserk_wasted} depth={env.max_depth} hp={env.hp_frac:.2f}")
    if env.ac is None:
        print("FAIL: AC never read off the status panel")
    env.close()


if __name__ == "__main__":
    main()

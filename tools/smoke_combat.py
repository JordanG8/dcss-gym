"""Check the combat reward fires on real fights, and that shops don't wedge.

Both are message-matching, and message-matching is exactly where I have already
guessed wrong twice ("You are too berserk!" vs the phrasing I invented). So run
a real game and print what the reward actually did.

    /root/pty-venv/bin/python smoke_combat.py
"""
from dcss_env import RE_HIT, RE_KILL, RE_SHOP, VARIANTS, DCSSEnv


def main():
    names = [n for n, _ in VARIANTS["a"]]
    idx = {n: i for i, n in enumerate(names)}
    env = DCSSEnv(env_id=91, max_steps=400, variant="a")
    env.reset()

    combat_rewards, samples, shops = [], [], 0
    for t in range(220):
        # Explore to find monsters, then swing at whatever turns up.
        a = idx["autofight"] if (t % 3) else idx["explore"]
        scr, r, done, info = env.step(a)
        msg = "\n".join(scr.split("\n")[17:])
        if RE_HIT.search(msg) or RE_KILL.search(msg):
            combat_rewards.append(r)
            if len(samples) < 6:
                samples.append((round(r, 3),
                                " | ".join(l.strip() for l in msg.split("\n")
                                           if l.strip())[:88]))
        if RE_SHOP.search(scr):
            shops += 1
        if done:
            print(f"episode ended at step {t}: {env.outcome}")
            break

    print(f"\nsteps with combat messages : {len(combat_rewards)}")
    print(f"hits={env.hits} kills={env.kills} "
          f"depth={env.max_depth} xl={env.xl} hp={env.hp_frac:.2f}")
    print(f"shop screens seen (should be 0 after settle): {shops}")
    print(f"total combat reward         : {sum(combat_rewards):+.2f}")
    print("\nsamples:")
    for r, line in samples:
        print(f"  {r:+.3f}  {line}")
    env.close()


if __name__ == "__main__":
    main()

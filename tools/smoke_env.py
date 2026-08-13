"""Smoke-test DCSSEnv with a random policy.

Answers the only two questions that decide whether RL is affordable here:
how many seconds does a step cost, and does a random policy ever see reward?

    /root/pty-venv/bin/python smoke_env.py --envs 4 --steps 60
"""
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

from dcss_env import ACTION_NAMES, N_ACTIONS, DCSSEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    envs = [DCSSEnv(env_id=i, max_steps=args.steps) for i in range(args.envs)]
    pool = ThreadPoolExecutor(max_workers=args.envs)

    t0 = time.time()
    list(pool.map(lambda e: e.reset(), envs))
    print(f"reset {args.envs} envs in {time.time()-t0:.1f}s", flush=True)

    done = [False] * args.envs
    totals = [0.0] * args.envs
    n_steps = 0
    t0 = time.time()

    def run(i):
        if done[i]:
            return None
        a = random.randrange(N_ACTIONS)
        return envs[i].step(a)

    for t in range(args.steps):
        results = list(pool.map(run, range(args.envs)))
        for i, r in enumerate(results):
            if r is None:
                continue
            _, rew, d, info = r
            totals[i] += rew
            n_steps += 1
            if d:
                done[i] = True
                print(f"  env{i} finished t={t} {info}", flush=True)
        if all(done):
            break

    el = time.time() - t0
    print(f"\n{n_steps} steps in {el:.1f}s "
          f"= {n_steps/el:.2f} steps/s across {args.envs} envs")
    for i, e in enumerate(envs):
        print(f"  env{i}: return={totals[i]:+.2f} max_depth={e.max_depth} "
              f"xl={e.xl} turns={e.turns} outcome={e.outcome or 'running'}")
    for e in envs:
        e.close()


if __name__ == "__main__":
    main()

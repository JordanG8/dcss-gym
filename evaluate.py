"""Held-out, deterministic DCSS evaluation.

Training metrics describe what happened while the policy was sampling.  This
script evaluates an immutable seed list with greedy legal actions, records the
exact configuration, and writes a JSON report outside version control.

    /root/pty-venv/bin/python evaluate.py --checkpoint data/rl_policy.b.pt
    /root/pty-venv/bin/python evaluate.py --checkpoint data/rl_policy.b.pt --limit 100
"""
import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from dcss_env import DCSSEnv, VARIANTS
from train_rl import Policy, apply_action_mask, encode


HERE = Path(__file__).parent
DEFAULT_SEEDS = tuple(810_001 + 7919 * i for i in range(100))


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summarize(episodes, target_depth):
    outcomes = Counter(e["outcome"] for e in episodes)
    n = max(1, len(episodes))
    return {
        "episodes": len(episodes),
        "solve_rate": sum(e["max_depth"] >= target_depth for e in episodes) / n,
        "mean_depth": sum(e["max_depth"] for e in episodes) / n,
        "mean_turns": sum(e["turns"] for e in episodes) / n,
        "mean_actions": sum(e["actions"] for e in episodes) / n,
        "outcomes": dict(outcomes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="b", choices=sorted(VARIANTS))
    ap.add_argument("--target-depth", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=20,
                    help="first N fixed held-out seeds; use 100 for a full report")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not 1 <= args.limit <= len(DEFAULT_SEEDS):
        ap.error(f"--limit must be 1..{len(DEFAULT_SEEDS)}")

    action_names = [name for name, _key in VARIANTS[args.variant]]
    model = Policy(len(action_names)).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    episodes = []
    for i, seed in enumerate(DEFAULT_SEEDS[:args.limit]):
        env = DCSSEnv(env_id=10_000 + i, seed=seed, variant=args.variant,
                      target_depth=args.target_depth, max_steps=args.max_steps)
        try:
            obs = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    x = encode(obs).unsqueeze(0).to(args.device)
                    mask = torch.tensor(env.action_mask(), dtype=torch.bool,
                                        device=args.device).unsqueeze(0)
                    logits, _value = model(x)
                    action = int(apply_action_mask(logits, mask).argmax(-1))
                obs, _reward, done, _info = env.step(action)
            episodes.append({
                "seed": seed, "outcome": env.outcome, "max_depth": env.max_depth,
                "turns": env.turns, "actions": env.steps,
            })
            print(f"{i + 1:3d}/{args.limit} seed={seed} {env.outcome:14s} "
                  f"D:{env.max_depth} turns={env.turns}", flush=True)
        finally:
            env.close()

    report = {
        "kind": "held_out_seed_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "variant": args.variant,
        "target_depth": args.target_depth,
        "max_steps": args.max_steps,
        "policy": "greedy argmax over legal actions",
        "seeds": list(DEFAULT_SEEDS[:args.limit]),
        "summary": summarize(episodes, args.target_depth),
        "episodes_detail": episodes,
    }
    out = args.out or HERE / "data" / f"eval-{args.variant}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    s = report["summary"]
    print(f"\nD:{args.target_depth}={s['solve_rate']:.1%} "
          f"mean_depth={s['mean_depth']:.2f}  report={out}")


if __name__ == "__main__":
    main()

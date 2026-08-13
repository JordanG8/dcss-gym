"""Evaluate a checkpoint on the deterministic public-view DCSS Gym.

    /root/pty-venv/bin/python gym_eval.py --checkpoint data/rl_policy.b.pt
"""
import argparse
from collections import defaultdict

import torch

from dcss_gym import scenarios
from train_rl import Policy, apply_action_mask, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="b", choices=("a", "b", "c"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-score", type=float, default=0.95,
                    help="required score for every skill group")
    args = ap.parse_args()

    suite = list(scenarios(args.variant))
    model = Policy(len(suite[0].observation.action_names)).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    if not 0 < args.min_score <= 1:
        ap.error("--min-score must be in (0, 1]")
    passed = 0
    by_skill = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for case in suite:
            obs = case.observation
            x = encode(obs.screen).unsqueeze(0).to(args.device)
            mask = torch.tensor(obs.action_mask, dtype=torch.bool,
                                device=args.device).unsqueeze(0)
            logits, _value = model(x)
            chosen = obs.action_names[int(apply_action_mask(logits, mask).argmax(-1))]
            ok = chosen == case.expected_action
            passed += int(ok)
            by_skill[case.skill][0] += int(ok)
            by_skill[case.skill][1] += 1
            print(f"{'PASS' if ok else 'FAIL'}  {case.name:22s} "
                  f"wanted={case.expected_action:10s} got={chosen}")
    print(f"\nGym score: {passed}/{len(suite)} ({passed / len(suite):.1%})")
    all_groups_pass = True
    for skill, (good, total) in sorted(by_skill.items()):
        rate = good / total
        ok = rate >= args.min_score
        all_groups_pass &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {skill:22s} {good}/{total} ({rate:.1%})")
    return 0 if all_groups_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

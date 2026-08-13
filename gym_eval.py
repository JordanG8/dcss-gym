"""Evaluate a checkpoint on the deterministic public-view DCSS Gym.

    /root/pty-venv/bin/python gym_eval.py --checkpoint data/rl_policy.b.pt
"""
import argparse

import torch

from dcss_gym import scenarios
from train_rl import Policy, apply_action_mask, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="b", choices=("a", "b", "c"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    suite = list(scenarios(args.variant))
    model = Policy(len(suite[0].observation.action_names)).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    passed = 0
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
            print(f"{'PASS' if ok else 'FAIL'}  {case.name:22s} "
                  f"wanted={case.expected_action:10s} got={chosen}")
    print(f"\nGym score: {passed}/{len(suite)}")
    return 0 if passed == len(suite) else 1


if __name__ == "__main__":
    raise SystemExit(main())

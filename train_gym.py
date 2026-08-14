"""Supervised warm-start on the deterministic, player-visible DCSS Gym.

This is deliberately a *curriculum seed*, not a claim of a competent Crawl
agent: it makes the intended behaviour (notice a nearby enemy; prefer an
observable upgrade; cancel a downgrade) executable and regression-testable
before PPO has to discover it from sparse game experience.

    /root/pty-venv/bin/python train_gym.py --variant b
    /root/pty-venv/bin/python train_gym.py --out data/rl_policy.b.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dcss_env import VARIANTS
from dcss_gym import scenarios
from train_rl import Policy, apply_action_mask, encode, load_policy_state


HERE = Path(__file__).parent


def dataset(variant):
    """Return encoded public observations, legal masks, and action labels."""
    cases = list(scenarios(variant))
    names = tuple(name for name, _key in VARIANTS[variant])
    x = torch.stack([encode(case.observation.screen) for case in cases])
    mask = torch.tensor([case.observation.action_mask for case in cases],
                        dtype=torch.bool)
    y = torch.tensor([names.index(case.expected_action) for case in cases])
    return cases, x, mask, y


def score(model, x, mask, y):
    with torch.no_grad():
        logits, _value = model(x)
        pred = apply_action_mask(logits, mask).argmax(-1)
    return int((pred == y).sum()), len(y), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="b")
    # Fixed seed: reaches all six exercises by epoch 200; leave margin so the
    # default command is a trustworthy smoke test rather than a near miss.
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None,
                    help="checkpoint path (default: data/gym_policy.<variant>.pt)")
    ap.add_argument("--resume-from", type=Path,
                    help="warm-start from an existing checkpoint")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="fine-tune only the semantic action head")
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs must be positive")

    torch.manual_seed(args.seed)
    names = tuple(name for name, _key in VARIANTS[args.variant])
    cases, x, mask, y = dataset(args.variant)
    model = Policy(len(names))
    if args.resume_from:
        report = load_policy_state(
            model, torch.load(args.resume_from, map_location="cpu"))
        print(f"warm-started from {args.resume_from}: {report}")
    if args.freeze_trunk:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.actor.parameters():
            parameter.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=0.0)
    cached_features = None
    if args.freeze_trunk:
        # The visual trunk is immutable in this mode.  Computing the exact
        # same transformer activations on every epoch made a tiny 12-example
        # head fit take minutes under eight live evaluators; cache them once.
        model.eval()
        with torch.no_grad():
            cached_features = model.enc(model._tokens(x)).mean(1)
    model.train()
    for epoch in range(args.epochs):
        if cached_features is None:
            logits, _value = model(x)
        else:
            logits = model.actor(cached_features)
        loss = F.cross_entropy(apply_action_mask(logits, mask), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 40 == 0 or epoch + 1 == args.epochs:
            if cached_features is None:
                correct, total, _pred = score(model, x, mask, y)
            else:
                with torch.no_grad():
                    pred = apply_action_mask(
                        model.actor(cached_features), mask).argmax(-1)
                correct, total = int((pred == y).sum()), len(y)
            print(f"epoch={epoch + 1:3d} loss={loss.item():.4f} gym={correct}/{total}")

    correct, total, pred = score(model, x, mask, y)
    for case, action in zip(cases, pred.tolist()):
        print(f"{case.name:24s} -> {names[action]}")
    if correct != total:
        raise SystemExit(f"Gym warm-start did not converge ({correct}/{total})")

    out = args.out or HERE / "data" / f"gym_policy.{args.variant}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved player-view Gym warm-start: {out}")


if __name__ == "__main__":
    main()

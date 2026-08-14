"""Create a reproducible expanded policy checkpoint from an older one."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from dcss_env import VARIANTS
from train_rl import Policy, load_policy_state


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="c")
    args = parser.parse_args()

    torch.manual_seed(0)
    names = [name for name, _ in VARIANTS[args.variant]]
    model = Policy(len(names))
    source_state = torch.load(args.source, map_location="cpu")
    report = load_policy_state(model, source_state)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.destination)
    metadata = {
        "format": "dcss-policy-migration-v1",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "destination_sha256": sha256(args.destination),
        "variant": args.variant,
        "actions": names,
        "migration": report,
        "hostile_channel": "player-visible glyph high bit",
    }
    args.destination.with_suffix(args.destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

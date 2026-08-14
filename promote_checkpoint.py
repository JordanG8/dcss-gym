"""Promote an evaluated candidate checkpoint to the spectator champion lane."""
import argparse
import json
from pathlib import Path

from checkpointing import manifest_path, publish_manifest, read_manifest


HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("a", "b", "c"), default="c")
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="JSON evaluation containing episodes and solve_rate")
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if int(evaluation.get("episodes", 0)) < 20:
        raise SystemExit("refusing promotion: at least 20 evaluation episodes required")
    if "solve_rate" not in evaluation:
        raise SystemExit("refusing promotion: evaluation has no solve_rate")
    candidate = read_manifest(
        manifest_path(args.data_root, args.variant, "candidate"))
    promoted = publish_manifest(
        Path(candidate["checkpoint"]),
        manifest_path(args.data_root, args.variant, "champion"),
        variant=args.variant, channel="champion",
        update=candidate.get("update", 0),
        architecture=candidate.get("architecture", "spatial-v1"),
        action_names=candidate.get("action_names", ()), metrics=evaluation)
    print(json.dumps(promoted, indent=2))


if __name__ == "__main__":
    main()

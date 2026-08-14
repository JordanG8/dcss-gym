"""Production asynchronous recurrent DCSS trainer."""
import argparse
import json
import random
import time
from collections import deque

import torch

from async_actors import AsyncActorPool, BatchedInferenceServer
from checkpointing import (atomic_json_write, atomic_torch_save, manifest_path,
                           publish_manifest, read_manifest)
from dcss_env import VARIANTS
from r2d2 import PrioritizedEpisodeReplay, RecurrentQ
from train_r2d2 import epsilon_ladder, learn
from train_rl import DATA, load_policy_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("a", "b", "c"), default="c")
    parser.add_argument("--envs", type=int, default=24)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument(
        "--learner-every", type=int, default=24,
        help="Run one learner update per this many fresh actor steps.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=10)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--replay-steps", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--target-sync", type=int, default=250)
    parser.add_argument("--actor-sync", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-depth", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_names = [name for name, _ in VARIANTS[args.variant]]
    online = RecurrentQ(len(action_names))
    candidate_manifest = manifest_path(DATA, args.variant, "candidate")
    update_offset = 0
    if args.resume and candidate_manifest.exists():
        previous = read_manifest(candidate_manifest)
        update_offset = int(previous.get("update", 0) or 0)
        state = torch.load(previous["checkpoint"], map_location="cpu")
        if previous.get("architecture") == "r2d2-v1":
            online.load_state_dict(state)
        else:
            report = online.warm_start_spatial(state)
            print(f"warm-started recurrent spatial encoder: {report}", flush=True)
    online.to(device)
    target = RecurrentQ(len(action_names)).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr, eps=1e-5)
    replay = PrioritizedEpisodeReplay(args.replay_steps)

    inference = BatchedInferenceServer(
        online, args.envs, device, batch_size=min(32, args.envs),
        epsilons=epsilon_ladder(args.envs))
    actors = AsyncActorPool(
        args.envs, args.variant, args.max_steps,
        args.target_depth, inference)
    episodes = [[] for _ in range(args.envs)]
    returns = [0.0] * args.envs
    recent_depths, recent_returns = deque(maxlen=100), deque(maxlen=100)
    env_rows = [{} for _ in range(args.envs)]
    total_actor_steps = learner_updates = 0
    started = time.time()
    checkpoint = DATA / f"rl_policy.{args.variant}.r2d2.pt"
    log_path = DATA / f"r2d2_log.{args.variant}.jsonl"
    actors.start()
    try:
        while learner_updates < args.updates:
            event = actors.events.get(timeout=30)
            total_actor_steps += 1
            episodes[event.actor].append(event.transition)
            returns[event.actor] += event.transition.reward
            info = event.info
            env_rows[event.actor] = {
                "env": event.actor, "depth": info.get("depth", 0),
                "xl": info.get("xl", 1), "turns": info.get("turns", 0),
                "hp": info.get("hp_frac", 0), "step": info.get("steps", 0),
                "action": info.get("action", ""),
            }
            if event.transition.done:
                replay.add(episodes[event.actor])
                episodes[event.actor] = []
                recent_depths.append(info.get("max_depth", info.get("depth", 1)))
                recent_returns.append(returns[event.actor])
                returns[event.actor] = 0.0
            if total_actor_steps % max(1, args.envs // 2) == 0:
                atomic_json_write(
                    [row for row in env_rows if row],
                    DATA / f"rl_envs.{args.variant}.json")
            try:
                watched = int(
                    (DATA / f"rl_view.{args.variant}.txt").read_text())
            except (OSError, ValueError):
                watched = 0
            if event.actor == max(0, min(args.envs - 1, watched)):
                atomic_json_write({
                    "architecture": "r2d2-v1", "variant": args.variant,
                    "env": event.actor, "step": info.get("steps", 0),
                    "action": info.get("action", ""),
                    "screen": event.screen, "colors": event.colors,
                    "names": action_names, "probs": event.probabilities,
                }, DATA / f"rl_live.{args.variant}.json")

            if (len(replay) < args.learning_starts or
                    total_actor_steps % args.learner_every != 0):
                continue
            beta = min(1.0, 0.4 + 0.6 * learner_updates / args.updates)
            loss = learn(online, target, optimizer, replay, args, device, beta)
            learner_updates += 1
            published_update = update_offset + learner_updates
            if learner_updates % args.actor_sync == 0:
                inference.sync_from(online)
            if learner_updates % args.target_sync == 0:
                target.load_state_dict(online.state_dict())
            mean_depth = (sum(recent_depths) / len(recent_depths)
                          if recent_depths else 0.0)
            solve_rate = (sum(depth >= args.target_depth
                              for depth in recent_depths) / len(recent_depths)
                          if recent_depths else 0.0)
            row = {
                "architecture": "r2d2-v1-async", "variant": args.variant,
                "update": published_update, "run_update": learner_updates,
                "actor_steps": total_actor_steps,
                "replay_steps": len(replay), "episodes": len(recent_depths),
                "mean_depth": round(mean_depth, 3),
                "best_depth": max(recent_depths, default=0),
                "solve_rate": round(solve_rate, 4),
                "mean_return": round(
                    sum(recent_returns) / len(recent_returns), 3)
                    if recent_returns else 0.0,
                "loss": round(loss, 5),
                "steps_per_second": round(
                    total_actor_steps / (time.time() - started), 2),
                "queue_depth": actors.events.qsize(),
            }
            atomic_json_write(row, DATA / f"r2d2_live.{args.variant}.json")
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if learner_updates % 10 == 0:
                print(json.dumps(row), flush=True)
            if learner_updates % args.checkpoint_every == 0:
                atomic_torch_save(online.state_dict(), checkpoint)
                publish_manifest(
                    checkpoint, candidate_manifest, variant=args.variant,
                    channel="candidate", update=published_update,
                    architecture="r2d2-v1", action_names=action_names,
                    metrics=row)
    finally:
        actors.stop()

    atomic_torch_save(online.state_dict(), checkpoint)
    publish_manifest(
        checkpoint, candidate_manifest, variant=args.variant,
        channel="candidate", update=update_offset + learner_updates,
        architecture="r2d2-v1", action_names=action_names,
        metrics={"complete": True, "actor_steps": total_actor_steps})


if __name__ == "__main__":
    main()

"""Recurrent prioritized-replay learner for player-visible DCSS.

This is a single-machine R2D2-style implementation: many Crawl actors share
one batched GPU policy, while rare episode sequences remain available for
replay instead of being discarded after one PPO update.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from checkpointing import (atomic_json_write, atomic_torch_save, manifest_path,
                           publish_manifest, read_manifest)
from dcss_env import DCSSEnv, VARIANTS
from r2d2 import PrioritizedEpisodeReplay, RecurrentQ, Transition
from train_rl import DATA, encode, load_policy_state


HERE = Path(__file__).parent


def legal_random(mask):
    choices = [index for index, allowed in enumerate(mask) if allowed]
    if not choices:
        raise ValueError("actor received an action mask with no legal action")
    return random.choice(choices)


def epsilon_ladder(count, low=0.01, high=0.40):
    if count == 1:
        return [low]
    return [low * (high / low) ** (index / (count - 1))
            for index in range(count)]


def sequence_loss(online, target, sample, gamma, device):
    hidden = online.initial_state(1, device)
    target_hidden = target.initial_state(1, device)
    losses, errors = [], []
    for index, transition in enumerate(sample.transitions):
        observation = transition.observation.to(
            device=device, dtype=torch.long).unsqueeze(0)
        action_mask = transition.action_mask.to(device).unsqueeze(0)
        previous_action = torch.tensor(
            [transition.previous_action], device=device)
        q, next_hidden = online.step(
            observation, previous_action, hidden, action_mask)
        with torch.no_grad():
            _target_q, next_target_hidden = target.step(
                observation, previous_action, target_hidden, action_mask)
            next_observation = transition.next_observation.to(
                device=device, dtype=torch.long).unsqueeze(0)
            next_mask = transition.next_action_mask.to(device).unsqueeze(0)
            current_action = torch.tensor([transition.action], device=device)
            next_online_q, _ = online.step(
                next_observation, current_action, next_hidden.detach(), next_mask)
            next_target_q, _ = target.step(
                next_observation, current_action, next_target_hidden, next_mask)
            next_action = next_online_q.argmax(dim=-1)
            bootstrap = next_target_q.gather(1, next_action[:, None]).squeeze(1)
            expected = torch.tensor([transition.reward], device=device)
            if not transition.done:
                expected = expected + gamma * bootstrap
        prediction = q[0, transition.action]
        error = prediction - expected[0]
        if index >= sample.burn_in:
            losses.append(F.smooth_l1_loss(prediction, expected[0]))
            errors.append(float(error.detach().abs()))
        hidden = next_hidden
        target_hidden = next_target_hidden
        if index + 1 == sample.burn_in:
            # Burn-in reconstructs memory but is not part of the gradient
            # horizon, preventing very long episode graphs.
            hidden = hidden.detach()
    if not losses:
        return None, 0.0
    return torch.stack(losses).mean() * sample.weight, max(errors)


def learn(online, target, optimizer, replay, args, device, beta):
    samples = replay.sample(
        args.batch_size, burn_in=args.burn_in,
        unroll=args.unroll, beta=beta)
    weighted, priorities = [], []
    for sample in samples:
        loss, priority = sequence_loss(
            online, target, sample, args.gamma, device)
        if loss is not None:
            weighted.append(loss)
            priorities.append((sample.replay_index, priority))
    if not weighted:
        return 0.0
    loss = torch.stack(weighted).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    replay.update_priorities(
        [index for index, _ in priorities],
        [priority for _, priority in priorities])
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("a", "b", "c"), default="c")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=10)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--replay-steps", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--target-sync", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-depth", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_names = [name for name, _ in VARIANTS[args.variant]]
    online = RecurrentQ(len(action_names))
    candidate_manifest = manifest_path(DATA, args.variant, "candidate")
    if args.resume and candidate_manifest.exists():
        previous = read_manifest(candidate_manifest)
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

    envs = [DCSSEnv(i, target_depth=args.target_depth,
                    max_steps=args.max_steps, variant=args.variant)
            for i in range(args.envs)]
    pool = ThreadPoolExecutor(max_workers=args.envs)
    observations = list(pool.map(lambda env: env.reset(), envs))
    hidden = online.initial_state(args.envs, device)
    previous_actions = torch.full(
        (args.envs,), -1, dtype=torch.long, device=device)
    episodes = [[] for _ in envs]
    epsilons = epsilon_ladder(args.envs)
    recent_depths, recent_returns = deque(maxlen=100), deque(maxlen=100)
    returns = [0.0] * args.envs
    total_actor_steps = learner_updates = 0
    started = time.time()
    checkpoint = DATA / f"rl_policy.{args.variant}.r2d2.pt"
    log_path = DATA / f"r2d2_log.{args.variant}.jsonl"

    try:
        while learner_updates < args.updates:
            decision_screens = observations
            decision_colors = [env.color_text() for env in envs]
            encoded = torch.stack([encode(text) for text in observations])
            masks = torch.tensor(
                [env.action_mask() for env in envs], dtype=torch.bool)
            with torch.no_grad():
                q, next_hidden = online.step(
                    encoded.to(device), previous_actions,
                    hidden, masks.to(device))
                greedy = q.argmax(dim=-1).cpu().tolist()
                probabilities = torch.softmax(q, dim=-1).cpu()
            actions = [
                legal_random(masks[index].tolist())
                if random.random() < epsilons[index] else greedy[index]
                for index in range(args.envs)]
            results = list(pool.map(
                lambda pair: pair[0].step(pair[1]), zip(envs, actions)))

            next_observations = []
            env_rows = []
            for index, (next_text, reward, done, info) in enumerate(results):
                next_encoded = encode(next_text)
                next_mask = torch.tensor(
                    info.get("action_mask", envs[index].action_mask()),
                    dtype=torch.bool)
                episodes[index].append(Transition(
                    encoded[index].to(torch.uint8),
                    int(previous_actions[index]), actions[index], float(reward),
                    bool(done), masks[index].clone(),
                    next_encoded.to(torch.uint8), next_mask))
                returns[index] += reward
                total_actor_steps += 1
                if done:
                    replay.add(episodes[index])
                    episodes[index] = []
                    recent_depths.append(envs[index].max_depth)
                    recent_returns.append(returns[index])
                    returns[index] = 0.0
                    next_text = envs[index].reset()
                    next_hidden[index].zero_()
                    previous_actions[index] = -1
                else:
                    previous_actions[index] = actions[index]
                next_observations.append(next_text)
                env_rows.append({
                    "env": index, "depth": info.get("depth", 0),
                    "xl": info.get("xl", 1), "turns": info.get("turns", 0),
                    "hp": info.get("hp_frac", 0), "step": info.get("steps", 0),
                    "action": info.get("action", action_names[actions[index]]),
                    "epsilon": round(epsilons[index], 4),
                })
            observations = next_observations
            hidden = next_hidden.detach()
            atomic_json_write(env_rows, DATA / f"rl_envs.{args.variant}.json")
            try:
                watched = int((DATA / f"rl_view.{args.variant}.txt").read_text())
            except (OSError, ValueError):
                watched = 0
            watched = max(0, min(args.envs - 1, watched))
            atomic_json_write({
                "architecture": "r2d2-v1", "variant": args.variant,
                "env": watched, "step": env_rows[watched]["step"],
                "action": action_names[actions[watched]],
                "screen": decision_screens[watched],
                "colors": decision_colors[watched],
                "names": action_names,
                "probs": [round(float(value), 6)
                          for value in probabilities[watched]],
                "value": round(float(q[watched].max()), 4),
            }, DATA / f"rl_live.{args.variant}.json")

            if len(replay) >= args.learning_starts:
                beta = min(1.0, 0.4 + 0.6 * learner_updates / args.updates)
                loss = learn(online, target, optimizer, replay, args, device, beta)
                learner_updates += 1
                if learner_updates % args.target_sync == 0:
                    target.load_state_dict(online.state_dict())
                mean_depth = (sum(recent_depths) / len(recent_depths)
                              if recent_depths else 0.0)
                solve_rate = (sum(d >= args.target_depth for d in recent_depths)
                              / len(recent_depths) if recent_depths else 0.0)
                row = {
                    "architecture": "r2d2-v1", "variant": args.variant,
                    "update": learner_updates, "actor_steps": total_actor_steps,
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
                        channel="candidate", update=learner_updates,
                        architecture="r2d2-v1", action_names=action_names,
                        metrics=row)
    finally:
        for env in envs:
            env.close()
        pool.shutdown(wait=True)

    atomic_torch_save(online.state_dict(), checkpoint)
    publish_manifest(
        checkpoint, candidate_manifest, variant=args.variant,
        channel="candidate", update=learner_updates,
        architecture="r2d2-v1", action_names=action_names,
        metrics={"complete": True, "actor_steps": total_actor_steps})


if __name__ == "__main__":
    main()

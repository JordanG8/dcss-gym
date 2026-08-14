"""Measure centralized neural inference plus real Crawl actor throughput."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from dcss_env import DCSSEnv, VARIANTS
from r2d2 import RecurrentQ
from train_rl import encode


def available_gib():
    fields = {}
    with open("/proc/meminfo", encoding="utf-8") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            fields[key] = int(value.strip().split()[0])
    return round(fields["MemAvailable"] / 1024 / 1024, 2)


def run(count, batches, variant, threads):
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [name for name, _ in VARIANTS[variant]]
    model = RecurrentQ(len(names)).to(device).eval()
    envs = [DCSSEnv(index, max_steps=batches + 20, variant=variant)
            for index in range(count)]
    pool = ThreadPoolExecutor(max_workers=count)
    started = time.time()
    try:
        observations = list(pool.map(lambda env: env.reset(), envs))
        startup = time.time() - started
        hidden = model.initial_state(count, device)
        previous = torch.full((count,), -1, dtype=torch.long, device=device)
        interaction_started = time.time()
        for _ in range(batches):
            encoded = torch.stack([encode(text) for text in observations])
            masks = torch.tensor(
                [env.action_mask() for env in envs], dtype=torch.bool)
            with torch.no_grad():
                q, hidden = model.step(
                    encoded.to(device), previous, hidden, masks.to(device))
                actions = q.argmax(dim=-1).cpu().tolist()
            results = list(pool.map(
                lambda pair: pair[0].step(pair[1]), zip(envs, actions)))
            next_observations = []
            for index, (text, _reward, done, _info) in enumerate(results):
                if done:
                    text = envs[index].reset()
                    hidden[index].zero_()
                    previous[index] = -1
                else:
                    previous[index] = actions[index]
                next_observations.append(text)
            observations = next_observations
            hidden = hidden.detach()
        interaction = time.time() - interaction_started
        return {
            "actors": count, "batches": batches, "steps": count * batches,
            "startup_s": round(startup, 2),
            "interaction_s": round(interaction, 2),
            "steps_per_second": round(count * batches / interaction, 2),
            "wsl_available_gib": available_gib(), "device": str(device),
        }
    finally:
        for env in envs:
            env.close()
        pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default="16,24,32,40,48")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--variant", choices=("a", "b", "c"), default="a")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for count in (int(value) for value in args.counts.split(",")):
        print(json.dumps(run(count, args.batches, args.variant, args.threads)),
              flush=True)


if __name__ == "__main__":
    main()

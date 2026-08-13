"""Watch all three variants and speak only when something matters.

Emits one line per event on stdout, for Monitor. Deliberately quiet: the point
is to stop hand-checking the numbers every few minutes.

    /root/pty-venv/bin/python tools/watch_all.py
"""
import json
import subprocess
import time
from pathlib import Path

DATA = Path("/mnt/c/Users/jorda/dcss-research/data")
VARIANTS = ["a", "b", "c"]
POLL = 90
SOLVE_MARKS = [0.35, 0.45, 0.55]
MIN_EPS = 40          # a solve rate over fewer episodes than this is noise


def rows(v):
    p = DATA / f"rl_log.{v}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not out:
        return []
    run = out[-1].get("run")
    return [r for r in out if r.get("run") == run]


def alive():
    r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
    return sum(1 for l in r.stdout.splitlines()
               if "train_rl.py" in l and "--variant" in l)


def main():
    hit_solve = {v: set() for v in VARIANTS}
    equipped = set()
    last_update = {v: -1 for v in VARIANTS}
    last_change = time.time()
    down_reported = False

    while True:
        time.sleep(POLL)
        moved = False
        for v in VARIANTS:
            rs = rows(v)
            if not rs:
                continue
            r = rs[-1]
            if r["update"] != last_update[v]:
                last_update[v] = r["update"]
                moved = True

            # First real equipment — the thing the whole pickup fix was for.
            if v not in equipped and (r.get("equips") or 0) > 0:
                equipped.add(v)
                print(f"FIRST EQUIP in variant {v}: equips={r['equips']} "
                      f"at update {r['update']} — the equipment branch is live "
                      f"(depth {r['mean_depth']:.2f}, D:5 {100*r['solve_rate']:.0f}%)",
                      flush=True)

            if r.get("episodes", 0) >= MIN_EPS:
                for m in SOLVE_MARKS:
                    if m not in hit_solve[v] and r["solve_rate"] >= m:
                        hit_solve[v].add(m)
                        print(f"*** variant {v}: D:5 solve rate "
                              f"{100*r['solve_rate']:.0f}% at update {r['update']} "
                              f"(depth {r['mean_depth']:.2f}, "
                              f"{r['episodes']} episodes)", flush=True)

        if moved:
            last_change = time.time()
            down_reported = False
        elif not down_reported and time.time() - last_change > 900:
            down_reported = True
            print(f"STALLED: no update from any variant in 15 min. "
                  f"trainers alive: {alive()}/3", flush=True)


if __name__ == "__main__":
    main()

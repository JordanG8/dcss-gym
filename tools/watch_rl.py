"""Milestone watcher for the PPO run. One stdout line per thing worth knowing.

Deliberately quiet: it reports progress records, learning signals, stalls and
crashes — not every update. Run under Monitor.

    /root/pty-venv/bin/python watch_rl.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LOG = HERE / "data" / "rl_log.jsonl"
RUNLOG = HERE / "rl_train.log"

POLL = 60
STALL_S = 900          # no new update for 15 min = wedged
# Quiet on purpose: the ask is "ping me at 30% solve rate". Intermediate depth
# and entropy milestones were useful while diagnosing reward bugs and are just
# noise now. Failure signals stay on — silence should mean "still climbing",
# not "died an hour ago".
DEPTH_MARKS = [4.0]
ENT_MARKS = []
SOLVE_MARKS = [0.30, 0.40, 0.50, 0.60]


def rows():
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def alive():
    r = subprocess.run(["pgrep", "-f", "train_rl.py"], capture_output=True)
    return r.returncode == 0


def emit(s):
    print(s, flush=True)


def main():
    # Episodes terminate at max_depth >= 5, so D:5 is routine now and anything
    # deeper is a SHAFT skipping past the target, not better play. Neither is
    # worth waking anyone for; the solve RATE is the real measure.
    best_depth = 99
    hit_depth, hit_ent, hit_solve = set(), set(), set()
    last_n = 0
    last_change = time.time()
    stall_reported = False
    regressed = set()
    peak_depth = 0.0

    while True:
        time.sleep(POLL)
        rs = rows()

        if len(rs) != last_n:
            last_n, last_change = len(rs), time.time()
            stall_reported = False
        elif not stall_reported and time.time() - last_change > STALL_S:
            stall_reported = True
            emit(f"STUCK: no new PPO update for {int((time.time()-last_change)/60)} min "
                 f"(stopped at update {last_n}). Run alive={alive()}")

        if not rs:
            if not alive():
                emit("CRASHED: train_rl.py is gone and wrote no updates.")
                return 1
            continue

        r = rs[-1]

        if r["best_depth"] > best_depth:
            best_depth = r["best_depth"]
            emit(f"NEW DEPTH RECORD D:{best_depth} at update {r['update']} "
                 f"({r['steps']} steps, {r['elapsed_s']//60}min) — "
                 f"mean_depth={r['mean_depth']} entropy={r['entropy']}")

        # Mean-depth marks need a real sample. The stats window is a 60-episode
        # deque that starts EMPTY on every restart, so at update 4 it held two
        # episodes and fired four "milestones" at once off a mean of 3.5.
        # A milestone computed from 2 episodes is noise wearing a milestone's
        # clothes.
        for m in DEPTH_MARKS:
            if m not in hit_depth and r["episodes"] >= 20 and r["mean_depth"] >= m:
                hit_depth.add(m)
                emit(f"MEAN DEPTH >= {m} at update {r['update']} "
                     f"(mean_depth={r['mean_depth']} over {r['episodes']} eps, "
                     f"entropy={r['entropy']}, return={r['mean_return']})")

        for m in ENT_MARKS:
            if m not in hit_ent and r["entropy"] <= m:
                hit_ent.add(m)
                emit(f"LEARNING SIGNAL: entropy fell below {m} "
                     f"(now {r['entropy']}, max is 1.792) at update {r['update']} — "
                     f"mean_depth={r['mean_depth']} return={r['mean_return']} "
                     f"actions={r['actions']}")

        for m in SOLVE_MARKS:
            if m not in hit_solve and r["episodes"] >= 20 and r["solve_rate"] >= m:
                hit_solve.add(m)
                emit(f"*** D:5 SOLVE RATE {r['solve_rate']*100:.0f}% at update "
                     f"{r['update']} ({r['elapsed_s']//60}min) — "
                     f"mean_depth={r['mean_depth']} return={r['mean_return']}")

        # --- regression alarms ---
        # A watcher that only fires on records treats "slowly getting worse"
        # as silence, and silence is indistinguishable from "still warming up".
        # Both of these are things I would act on, so both must speak.
        if len(rs) >= 3 and r["update"] > 30:
            last3 = rs[-3:]
            if all(x["mean_depth"] < 1.4 for x in last3) and "depth" not in regressed:
                regressed.add("depth")
                emit(f"REGRESSION: mean_depth below 1.4 for 3 updates "
                     f"(now {r['mean_depth']}, peak was {peak_depth}) at update "
                     f"{r['update']} — entropy={r['entropy']} actions={r['actions']}")
            shares = [x["actions"].get("explore", 0) / max(1, sum(x["actions"].values()))
                      for x in last3]
            if all(s < 0.04 for s in shares) and "explore" not in regressed:
                regressed.add("explore")
                emit(f"REGRESSION: explore collapsed below 4% for 3 updates "
                     f"(now {shares[-1]*100:.1f}%) at update {r['update']} — "
                     f"without it the agent cannot find stairs it has not seen. "
                     f"mean_depth={r['mean_depth']} entropy={r['entropy']}")
        peak_depth = max(peak_depth, r["mean_depth"])

        if not alive():
            tail = ""
            if RUNLOG.exists():
                tail = RUNLOG.read_text(encoding="utf-8", errors="replace")[-600:]
            done = r["update"] >= 400
            emit(f"{'RUN COMPLETE' if done else 'RUN DIED'} at update {r['update']} "
                 f"({r['elapsed_s']//60}min): best=D:{r['best_depth']} "
                 f"mean_depth={r['mean_depth']} solve={r['solve_rate']*100:.0f}% "
                 f"entropy={r['entropy']}\n--- tail ---\n{tail}")
            return 0


if __name__ == "__main__":
    sys.exit(main())

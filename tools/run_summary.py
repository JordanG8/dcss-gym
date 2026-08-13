"""Natural-language summary of a training run, from its rl_log.

    python tools/run_summary.py                 # every live variant
    python tools/run_summary.py --variant b
    python tools/run_summary.py --log data/archive/run-.../rl_log.b.jsonl

The point is to answer "what happened and what should I do about it" without
anyone having to eyeball 1200 log lines. Two rules it exists to enforce, both
learned the hard way on 2026-08-13:

1. A counter that is 0 for a whole run is a broken counter until proven
   otherwise. `equips` read 0 for 2449 updates across three variants and the
   conclusion drawn at the time was about the GAME ("nothing reaches the
   pack") when it was about the DETECTOR. Dead branches are now the first
   thing this prints, before any performance number.

2. Outcome mix comes from the trainer's own `outcomes` field, never from
   games.jsonl. games.jsonl records every solve from all 16 envs but only
   env 0's failures, which overstated the solve rate 8x (56.2% against a real
   7.2%). `outcomes` is a rolling window of the last 60 episodes across all
   envs — so its PROPORTIONS are honest but its counts must not be summed
   across updates, which would count each episode ~60 times.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# Counters worth checking for "never moved". Each is (field, what a zero means).
WATCHED = [
    ("equips", "no equipment change ever improved AC or weapon power"),
    ("ac_gained", "the AC reward never paid out"),
    ("wpn_gained", "the weapon-power reward never paid out"),
    ("kills", "nothing was ever killed"),
    ("hits", "no blow ever landed"),
    ("berserks", "berserk was never successfully used"),
    ("ascents", "the agent never went back up a level"),
]
# The starting Minotaur Berserker: animal skin (AC 2), +0 hand axe (power 7).
BASE_AC = 2
BASE_WPN = 7


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        return []
    # `update` restarts at 1 on resume, so keep only the newest run in the file.
    return [r for r in rows if r.get("run") == rows[-1].get("run")]


def trend(rows, key, frac=0.3):
    """(first-window mean, last-window mean, verdict) for a metric over the run."""
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if len(vals) < 6:
        return None
    n = max(2, int(len(vals) * frac))
    a = sum(vals[:n]) / n
    b = sum(vals[-n:]) / n
    if a == 0:
        verdict = "rising" if b > 0 else "flat"
    else:
        change = (b - a) / abs(a)
        verdict = ("rising" if change > 0.15 else
                   "falling" if change < -0.15 else "flat")
    return a, b, verdict


def stuck_for(rows, key):
    """How many updates the metric has been identical for, at the end."""
    vals = [r.get(key) for r in rows]
    n = 0
    for v in reversed(vals):
        if v != vals[-1]:
            break
        n += 1
    return n


def pct(c, k):
    tot = sum(c.values())
    return 100 * c.get(k, 0) / tot if tot else 0.0


def summarise(rows, label):
    out = []
    w = out.append
    last, first = rows[-1], rows[0]
    hrs = last.get("elapsed_s", 0) / 3600
    w(f"=== {label} — run {last.get('run')} ===")
    w(f"{last['update']} updates, {last['steps']:,} steps, {hrs:.1f}h "
      f"({last['steps']/max(1,last.get('elapsed_s',1)):.1f} steps/s)")
    w("")

    # --- 1. dead branches, FIRST, before any performance claim -------------
    dead = [(f, why) for f, why in WATCHED
            if all(not r.get(f) for r in rows) and f in last]
    if dead:
        w("!! NEVER FIRED — suspect the instrument before the agent:")
        for f, why in dead:
            w(f"   {f} = 0 for all {len(rows)} updates — {why}.")
        w("   A counter flat at zero across a whole run is usually a broken")
        w("   detector, not a fact about the game. Verify it can fire at all")
        w("   before reasoning about its size.")
        w("")

    unused = []
    if last.get("actions"):
        seen = Counter()
        for r in rows:
            seen.update(r.get("actions", {}))
        unused = [a for a in last["actions"] if not seen.get(a)]
    if unused:
        w(f"!! actions never once taken: {', '.join(unused)}")
        w("")

    # --- 2. is it learning? ------------------------------------------------
    ts, td = trend(rows, "solve_rate"), trend(rows, "mean_depth")
    if ts and td:
        w(f"SOLVE {100*ts[0]:.1f}% -> {100*ts[1]:.1f}% ({ts[2]}) | "
          f"DEPTH {td[0]:.2f} -> {td[1]:.2f} ({td[2]})")
        flat = stuck_for(rows, "solve_rate")
        if ts[2] == "flat" and td[2] == "flat":
            w("   Verdict: PLATEAU. Neither depth nor solve rate moved across "
              "the run.")
        elif ts[2] == "falling" or td[2] == "falling":
            w("   Verdict: REGRESSING. Worth checking entropy for collapse "
              "onto a degenerate loop.")
        else:
            w("   Verdict: LEARNING.")
        if flat > 40:
            w(f"   solve_rate has been identical for {flat} straight updates.")
    w("")

    # --- 3. outcome mix, from the trainer's own rolling window --------------
    oc = Counter(last.get("outcomes", {}))
    if oc:
        n = sum(oc.values())
        w(f"HOW EPISODES END (last {n}, all envs):")
        for k, v in oc.most_common():
            w(f"   {k:<14} {100*v/n:5.1f}%")
        solved = sum(v for k, v in oc.items() if k.startswith("reached"))
        wedged = oc.get("step limit", 0) + oc.get("stalled", 0)
        if pct(oc, "died") > 40:
            w("   -> Dying is the main blocker. Survival problem: check AC, "
              "rest usage, and XL at death.")
        if 100 * wedged / n > 35:
            w("   -> Burning the clock is the main blocker. Check turns-per-"
              "action: far below 1.0 means keys that consume no game time "
              "(menus, blocked explore, refused travel) — a wedge, not slow "
              "play.")
        if 100 * solved / n > 25:
            w("   -> Solving reliably. Consider raising target depth.")
    w("")

    # --- 4. gear ------------------------------------------------------------
    if last.get("mean_ac") is not None:
        ac, wp = last.get("mean_ac"), last.get("mean_wpn")
        w(f"GEAR: AC {ac} (start {BASE_AC}) | weapon power {wp} "
          f"(start {BASE_WPN})")
        w(f"      earned this update: {last.get('ac_gained',0)} AC pts, "
          f"{last.get('wpn_gained',0)} weapon pts, "
          f"{last.get('equip_refused',0)} refusals")
        if wp is not None and wp < BASE_WPN - 0.05:
            w("   -> weapon power BELOW the starting hand axe: the wield "
              "action is handing away the good weapon again. This was the "
              "pre-2026-08-13 behaviour, 23.5% of frames.")
        if ac is not None and ac <= BASE_AC + 0.05:
            w("   -> AC never rises above the starting animal skin. Either "
              "armour is not reaching the pack or the wear path is dead.")
        ta = trend(rows, "mean_ac")
        if ta and ta[2] == "rising":
            w(f"   -> AC trending up ({ta[0]:.2f} -> {ta[1]:.2f}); the "
              "equipment reward is doing something.")
    w("")

    # --- 5. the codebase's own trigger --------------------------------------
    # Raw, NOT divided by `steps`. `nonsense` is a sum of per-EPISODE counters
    # across the live envs, while `steps` is cumulative for the whole run — so
    # nonsense/steps decays mechanically as the run gets longer and would show
    # a falling rate no matter what the policy did. Raw values are comparable
    # across updates because the env count and rollout length are fixed.
    # An episode runs up to 1000 steps and each update advances every env by
    # `rollout` (32), so it takes ~31 updates before the first episodes even
    # finish. Below roughly two turnovers the in-flight sum is still filling up
    # and rises no matter what the policy does — reporting a verdict there
    # would be reading the warm-up as a regression.
    MIN_UPDATES_FOR_NONSENSE = 60
    tn = trend(rows, "nonsense")
    if tn:
        w(f"NONSENSE (in-flight sum across envs): {tn[0]:.0f} -> {tn[1]:.0f} "
          f"({tn[2]})")
        if len(rows) < MIN_UPDATES_FOR_NONSENSE:
            w(f"   (only {len(rows)} updates — episodes have not turned over "
              "yet, so this rise is warm-up, not behaviour. No verdict.)")
        elif tn[2] != "falling":
            w("   -> Not falling. dcss_env.py's own note says this is the "
              "first number to CUT rather than raise when depth stalls.")
    te = trend(rows, "entropy")
    if te:
        w(f"ENTROPY {te[0]:.2f} -> {te[1]:.2f} ({te[2]})")
        if te[1] < 1.0:
            w("   -> Low. Check whether it has converged onto a degenerate "
              "loop rather than a policy.")
    w("")

    if last.get("actions"):
        mix = Counter(last["actions"]).most_common(5)
        tot = sum(last["actions"].values())
        w("ACTION MIX (latest): " +
          ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in mix))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", help="a, b or c. Default: all live variants.")
    ap.add_argument("--log", help="explicit path, e.g. an archived run")
    args = ap.parse_args()

    if args.log:
        paths = [(Path(args.log).stem, Path(args.log))]
    else:
        vs = [args.variant] if args.variant else ["a", "b", "c"]
        paths = [(f"variant {v}", HERE / "data" / f"rl_log.{v}.jsonl")
                 for v in vs]

    for label, p in paths:
        if not p.exists():
            print(f"=== {label} — no log at {p}\n")
            continue
        rows = load(p)
        if not rows:
            print(f"=== {label} — log is empty (run just started?)\n")
            continue
        print(summarise(rows, label))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What each variant is actually doing right now, and what it is stuck on.

Reads each variant's log plus its live env snapshot, so "stuck" is grounded in
the episode outcomes and the current screens rather than inferred from returns.

    /root/pty-venv/bin/python status.py
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
VARIANTS = {"a": "env picks item", "b": "agent picks item", "c": "env does all"}


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
    return out


def main():
    for v, label in VARIANTS.items():
        rs = rows(v)
        print(f"\n{'='*66}\nVARIANT {v.upper()}  ({label})")
        if not rs:
            print("  no updates yet")
            continue
        r = rs[-1]
        acts = r.get("actions", {})
        tot = max(1, sum(acts.values()))
        out = r.get("outcomes", {})
        eps = max(1, sum(out.values()))

        print(f"  update {r['update']}  steps {r['steps']:,}  "
              f"{r['elapsed_s']//60}min  episodes in window: {r.get('episodes',0)}")
        print(f"  mean depth {r['mean_depth']:.2f}   return {r['mean_return']:+.2f}   "
              f"D:5 {100*r['solve_rate']:.0f}%   entropy {r['entropy']:.3f}")
        print(f"  combat: hits {r.get('hits','-')}  kills {r.get('kills','-')}  "
              f"equips {r.get('equips','-')}  berserk {r.get('berserks','-')} "
              f"(wasted {r.get('berserk_wasted','-')})")

        print("  how episodes END:")
        for k, n in sorted(out.items(), key=lambda kv: -kv[1]):
            print(f"     {k:16s} {n:4d}  {100*n/eps:5.1f}%")

        print("  action mix:")
        for k, n in sorted(acts.items(), key=lambda kv: -kv[1]):
            print(f"     {k:12s} {100*n/tot:5.1f}%")

        live = DATA / f"rl_live.{v}.json"
        if live.exists():
            f = json.loads(live.read_text(errors="replace"))
            msg = [l.strip() for l in f.get("state", "").split("\n")[17:]
                   if l.strip()]
            print(f"  live env {f.get('env')} step {f.get('step')} "
                  f"chose '{f.get('action')}'")
            print(f"     last message: {(msg[0] if msg else '(none)')[:70]}")
    print()


if __name__ == "__main__":
    main()

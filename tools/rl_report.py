"""Print the PPO run as a table: is the action mix improving or collapsing?

    /root/pty-venv/bin/python rl_report.py [every_n]
"""
import json
import sys
from pathlib import Path

ORDER = ["explore", "autofight", "travel", "descend", "rest", "escape"]


def main():
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    p = Path(__file__).parent / "data" / "rl_log.jsonl"
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rs:
        print("no updates yet")
        return

    head = f"{'u':>4} {'depth':>5} {'ret':>6} {'ent':>6} |" + \
        "".join(f"{k[:4]:>6}" for k in ORDER) + " | outcomes"
    print(head)
    print("-" * len(head))
    for r in rs[::every] + ([rs[-1]] if len(rs) % every != 1 else []):
        a = r["actions"]
        tot = max(1, sum(a.values()))
        cells = "".join(f"{100*a.get(k,0)/tot:5.0f}%" for k in ORDER)
        o = ",".join(f"{k}:{v}" for k, v in sorted(r.get("outcomes", {}).items()))
        print(f"{r['update']:4d} {r['mean_depth']:5.2f} {r['mean_return']:+6.2f} "
              f"{r['entropy']:6.3f} |{cells} | {o}")


if __name__ == "__main__":
    main()

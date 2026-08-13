"""
Build the teacher-only training set.

Random-policy traces are unlearnable by construction (the key is not a function
of the screen), so mixing them in poisons the signal. This pulls out only
teacher decisions and writes them to data/traces_teacher.jsonl, which
prepare_dcss.py prefers when it exists.

Sources, in order of trust:
  data/traces.t*.jsonl        teacher shards, still un-merged  (policy known)
  data/traces.jsonl           rows carrying "policy": "teacher"

Rows from the earlier WEBTILES teacher are excluded on purpose: ~80% of those
are the unstick loop mashing enter/space/y/esc, not decisions.

    python build_teacher_set.py
"""
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = DATA / "traces_teacher.jsonl"

# The unstick keys. Present in bulk only in the abandoned webtiles run.
JUNK = {"\r", " ", "y", "\x1b"}


def rows():
    """Every trace explicitly labelled policy=teacher, from shards or canonical.

    Filtering on the label rather than the filename matters: shard names say
    nothing about policy (the dashboard's collectors write shards too, with the
    random policy), and unlabelled rows predate the fix that made the teacher
    able to descend — they are full of `>` spam against "You can't go down
    here!" and are worse than useless as training targets.
    """
    files = [p for p in DATA.glob("traces*.jsonl")
             if p.name != "traces_teacher.jsonl"]
    for p in sorted(files):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Require the `fallback` field to be PRESENT and false. Rows from
            # before that field existed cannot be filtered — they silently
            # carry the unwedge cycle, which is near-uniform noise and swamps
            # the signal. Absent means unknown, and unknown is excluded.
            if (o.get("policy") == "teacher"
                    and "fallback" in o and not o["fallback"]):
                yield o


def main():
    kept, counts = [], Counter()
    for o in rows():
        a = o.get("action")
        if not a or o.get("state") is None:
            continue
        kept.append({"state": o["state"], "action": a,
                     "game": o.get("game", ""), "policy": "teacher"})
        counts[a] += 1

    if not kept:
        print("no teacher traces found yet")
        return 1

    DATA.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    n = len(kept)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    ent = h / math.log(len(counts)) if len(counts) > 1 else 0.0
    junk = sum(v for k, v in counts.items() if k in JUNK)

    print(f"wrote {n} teacher traces -> {OUT}")
    print(f"distinct keys : {len(counts)}")
    print(f"entropy ratio : {ent:.3f}   (1.0 = indistinguishable from random)")
    print(f"prompt/unstick keys: {junk} ({100*junk/n:.1f}%)")
    for k, v in counts.most_common():
        print(f"   {k!r:6} {v:6}  {100*v/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

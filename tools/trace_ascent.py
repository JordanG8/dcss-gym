"""Find where an episode LOSES depth and show the frames around it.

`newly_deep` is computed against max_depth, so going back up costs no reward —
it just silently burns the step budget, which is the dominant failure mode.
Worth knowing exactly how it happens.

    /root/pty-venv/bin/python tools/trace_ascent.py
"""
import glob
import json
import os
import re

DEPTH = re.compile(r"Place:\s*\w+:(\d+)")   # the status panel, NOT loose "D:n"
                                            # in message text, which matched
                                            # "an escape hatch ... (D:1)".


def depth_of(frame):
    m = DEPTH.search(frame.get("state", ""))
    return int(m.group(1)) if m else None


def msg(frame):
    lines = [l.strip() for l in frame.get("state", "").split("\n")[17:] if l.strip()]
    return lines[0] if lines else ""


def main():
    files = sorted(glob.glob("data/rl_replays_*/*.jsonl"), key=os.path.getmtime)
    shown = 0
    for f in files[-14:]:
        frames = [json.loads(l) for l in open(f, errors="replace") if l.strip()]
        prev = None
        for i, fr in enumerate(frames):
            d = depth_of(fr)
            if d is None:
                continue
            if prev is not None and d < prev:
                print(f"\n=== {os.path.basename(f)}  frame {i}: D:{prev} -> D:{d}")
                for j in range(max(0, i - 7), min(len(frames), i + 2)):
                    g = frames[j]
                    dj = depth_of(g)
                    mark = " <<<" if j == i else ""
                    print(f"  f{j:4d} D:{dj if dj else '?'} "
                          f"{str(g.get('action')):10s} | {msg(g)[:70]}{mark}")
                shown += 1
                break
            prev = d
        if shown >= 3:
            break


if __name__ == "__main__":
    main()

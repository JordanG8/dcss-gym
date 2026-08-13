"""
Operations for the DCSS project. Replaces the pile of one-off shell scripts.

    python ops.py webtiles start|stop|status
    python ops.py fleet --workers 6 --games 20 --max-actions 220
    python ops.py merge          # fold parallel shards into games/traces
    python ops.py peek [-n -1]   # print a captured screen
    python ops.py status

Everything that used to live in a .sh file is here, so there is one place to
look and one language to read.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
GAMES = HERE / "games.jsonl"
TRACES = HERE / "data" / "traces.jsonl"
WSL = ["wsl", "-d", "Ubuntu", "--", "bash", "-lc"]

# webtiles forks a child that also holds the listening socket, so killing the
# parent alone leaves :8090 bound and the next server dies with EADDRINUSE
# while the old one keeps serving - a confusing half-broken state.
_KILL_8090 = (
    "for pid in $(ss -ltnp 2>/dev/null | grep ':8090' "
    "| grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill -9 $pid; done; "
    "pkill -9 -f 'webserver/server.py'; true"
)


def wsl(cmd, timeout=120, capture=True):
    return subprocess.run(WSL + [cmd], capture_output=capture, text=True,
                          timeout=timeout)


def webtiles_start():
    wsl(_KILL_8090)
    for _ in range(20):
        if wsl("ss -ltn | grep -q ':8090' && echo held || echo free"
               ).stdout.strip() == "free":
            break
        time.sleep(1)
    subprocess.Popen(WSL + [
        "cd /root/crawl/crawl-ref/source && source ~/webtiles-venv/bin/activate "
        "&& nohup python webserver/server.py >> /tmp/webtiles.log 2>&1 &"])
    for _ in range(30):
        time.sleep(1)
        if "held" in wsl("ss -ltn | grep -q ':8090' && echo held").stdout:
            print("webtiles up on http://localhost:8090")
            return 0
    print("webtiles failed to start; see /tmp/webtiles.log", file=sys.stderr)
    return 1


def webtiles_stop():
    wsl(_KILL_8090)
    print("webtiles stopped")
    return 0


def webtiles_status():
    held = "held" in wsl("ss -ltn | grep -q ':8090' && echo held").stdout
    print(f"webtiles: {'running' if held else 'stopped'}")
    return 0


def fleet(args):
    """Run parallel data-collection workers inside WSL."""
    # Each worker needs its OWN crawl -name: sharing one means sharing one save
    # file, and the games come back at turn 0 having stomped each other.
    # Distinct --tag keeps concurrent appends off the same shard file.
    parts = []
    for i in range(1, args.workers + 1):
        w = f"{args.prefix}{i}"
        parts.append(
            f"nohup /root/pty-venv/bin/python pty_agent.py "
            f"--games {args.games} --max-actions {args.max_actions} "
            f"--policy {args.policy} --target-depth {args.target_depth} "
            f"--name {w} --tag {w} --no-ttyrec >> fleet.log 2>&1 &")
    cmd = ("cd /mnt/c/Users/jorda/dcss-research && " + " ".join(parts)
           + " sleep 1; ps -eo args | grep -c '[p]ty_agent.py'")
    r = wsl(cmd, timeout=180)
    print(f"launched {args.workers} workers x {args.games} games "
          f"({args.workers * args.games} total)")
    print(r.stdout.strip())
    return 0


def wait(timeout=900):
    """Block until no collector workers remain, then report."""
    start = time.time()
    while time.time() - start < timeout:
        n = wsl("ps -eo args | grep -c '[p]ty_agent.py'").stdout.strip() or "0"
        try:
            n = int(n)
        except ValueError:
            n = 0
        if n == 0:
            print(f"workers finished after {int(time.time()-start)}s")
            return 0
        time.sleep(5)
    print("timed out waiting for workers")
    return 1


SKIP_SHARDS = {"games.jsonl", "games.pruned.jsonl", "traces.jsonl"}


def merge():
    """Fold per-worker shards into the canonical files, then delete them.

    Matches ANY suffix (games.w1, games.v3, games.ui...) - an earlier version
    globbed only `w*` and would have silently dropped a second wave launched
    with a different prefix.
    """
    moved_g = moved_t = 0

    for shard in sorted(p for p in HERE.glob("games.*.jsonl")
                        if p.name not in SKIP_SHARDS):
        lines = [l for l in shard.read_text(encoding="utf-8",
                                            errors="replace").splitlines()
                 if l.strip()]
        if lines:
            with open(GAMES, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            moved_g += len(lines)
        shard.unlink()

    tdir = TRACES.parent
    tdir.mkdir(parents=True, exist_ok=True)
    for shard in sorted(p for p in tdir.glob("traces.*.jsonl")
                        if p.name not in SKIP_SHARDS):
        lines = [l for l in shard.read_text(encoding="utf-8",
                                            errors="replace").splitlines()
                 if l.strip()]
        if lines:
            with open(TRACES, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            moved_t += len(lines)
        shard.unlink()

    print(f"merged {moved_g} game rows, {moved_t} traces")
    return 0


def peek(n):
    rows = [json.loads(l) for l in
            TRACES.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]
    if not rows:
        print("no traces")
        return 1
    r = rows[n]
    print(f"--- {len(rows)} rows; index {n}; action={r.get('action')!r} "
          f"source={r.get('source','pty')} ---")
    print(r["state"])
    return 0


def status():
    ng = 0
    if GAMES.exists():
        ng = sum(1 for l in GAMES.read_text(encoding="utf-8",
                                            errors="replace").splitlines()
                 if l.strip())
    nt = 0
    if TRACES.exists():
        nt = sum(1 for l in TRACES.read_text(encoding="utf-8",
                                             errors="replace").splitlines()
                 if l.strip())
    shards = len([p for p in HERE.glob("games.*.jsonl")
                  if p.name not in SKIP_SHARDS])
    running = wsl("ps -eo args | grep -c '[p]ty_agent.py'").stdout.strip()
    print(f"games logged : {ng}")
    print(f"traces       : {nt}")
    print(f"open shards  : {shards}")
    print(f"workers busy : {running}")
    webtiles_status()
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("webtiles")
    w.add_argument("action", choices=["start", "stop", "status"])

    f = sub.add_parser("fleet")
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--games", type=int, default=20)
    f.add_argument("--max-actions", type=int, default=220)
    f.add_argument("--policy", choices=["random","teacher"], default="teacher")
    f.add_argument("--target-depth", type=int, default=99)
    f.add_argument("--prefix", default="w",
                   help="worker name prefix; use a fresh one for a second wave "
                        "so it can't append to a running wave's shard")

    sub.add_parser("merge")
    sub.add_parser("wait")
    sub.add_parser("status")
    p = sub.add_parser("peek")
    p.add_argument("-n", type=int, default=-1)

    a = ap.parse_args()
    if a.cmd == "webtiles":
        return {"start": webtiles_start, "stop": webtiles_stop,
                "status": webtiles_status}[a.action]()
    if a.cmd == "fleet":
        return fleet(a)
    if a.cmd == "wait":
        return wait()
    if a.cmd == "merge":
        return merge()
    if a.cmd == "peek":
        return peek(a.n)
    return status()


if __name__ == "__main__":
    sys.exit(main())

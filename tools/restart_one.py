"""Restart a single variant, optionally discarding its policy.

Shell one-liners keep losing $VAR and quoting to the WSL interop layer, and a
half-executed kill is how three trainers once survived a restart. Do it in
Python where the pid is a variable, not a string being reparsed twice.

    python tools/restart_one.py a --fresh    # drop the checkpoint, start over
    python tools/restart_one.py a            # keep weights, just restart
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"


def pid_of(variant):
    r = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "train_rl.py" in line and f"--variant {variant}" in line:
            try:
                return int(line.split()[0])
            except (ValueError, IndexError):
                pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--fresh", action="store_true",
                    help="archive the checkpoint so training restarts from scratch")
    args = ap.parse_args()
    v = args.variant

    pid = pid_of(v)
    if pid:
        print(f"killing variant {v} (pid {pid})")
        os.kill(pid, signal.SIGKILL)
        for _ in range(10):
            time.sleep(1)
            if pid_of(v) is None:
                break
    else:
        print(f"variant {v} not running")

    if args.fresh:
        arch = DATA / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        for name in (f"rl_policy.{v}.pt", f"rl_log.{v}.jsonl"):
            src = DATA / name
            if src.exists():
                dst = arch / f"{name}.{stamp}"
                src.rename(dst)
                print(f"archived {name} -> archive/{dst.name}")

    subprocess.run([sys.executable, str(HERE / "reap.py"), "--kill"])
    print("done — the supervisor will start it again within 30s")


if __name__ == "__main__":
    main()

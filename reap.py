"""Kill orphaned crawl processes — games whose trainer is gone.

Killing a trainer does NOT kill its games: they are reparented to init and
keep running at ~100% of a core each, invisibly. This has bitten this project
repeatedly (once for about an hour). An orphan is any `crawl` whose parent is not a live agent process.

Checking `ppid == 1` is NOT enough: WSL reparents orphans to its own `/init`
shim (an arbitrary pid), so a ppid==1 test reported "0 orphaned" while 12
abandoned games were running. Identify the parent by its command line instead.

    python reap.py          # report only
    python reap.py --kill
"""
import argparse
import os
import signal
from pathlib import Path


def crawl_procs():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / "comm").read_text().strip() != "crawl":
                continue
            stat = (p / "stat").read_text().rsplit(") ", 1)[1].split()
            yield int(p.name), int(stat[1])      # pid, ppid
        except (OSError, IndexError, ValueError):
            continue


OWNERS = ("train_rl.py", "train_r2d2.py", "train_async_r2d2.py", "pty_agent.py",
          "webtiles_agent.py", "webserver/server.py")


def is_owner_command(command):
    return any(owner in command for owner in OWNERS)


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(
            errors="replace").replace("\0", " ")
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill", action="store_true")
    args = ap.parse_args()

    procs = list(crawl_procs())
    orphans = [pid for pid, ppid in procs
               if not is_owner_command(cmdline(ppid))]
    print(f"crawl processes: {len(procs)}   orphaned: {len(orphans)}")
    if not args.kill or not orphans:
        return
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    print(f"killed {len(orphans)} orphan(s)")


if __name__ == "__main__":
    main()

"""Keep the whole stack alive without a human in the loop.

Everything in this project has, at some point, needed me to restart it by hand:
the dashboard died from inherited console handles, trainers died when the PC
slept, and once three trainers survived a "restart" and kept writing stale
numbers for four hours. That is a supervision problem, not a training problem.

This process is the only thing that should ever need starting. It:
  * keeps the dashboard serving on 8099 (Windows side)
  * keeps one PPO trainer alive per variant (inside WSL)
  * reaps orphaned crawl games, which otherwise burn a core each forever
  * writes supervisor.log so a crash loop is visible after the fact

It is deliberately dumb: check, restart what is missing, sleep, repeat. It does
not tune anything and it never edits training config.

    python supervisor.py            # run in foreground
    python supervisor.py --status   # one-shot report, no changes
    python supervisor.py --stop     # stop trainers and dashboard
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
LOG = HERE / "supervisor.log"
WSL_PY = "/root/pty-venv/bin/python"
PROJ = "/mnt/c/Users/jorda/dcss-research"
VARIANTS = ["a", "b", "c"]

PORT = 8099
ENVS_PER_VARIANT = 16
THREADS = 3
CHECK_EVERY = 30          # seconds
GRACE = 90                # after starting something, don't judge it for this long


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def wsl(cmd, timeout=60):
    """Run a command inside WSL. Passed as a list so neither the Windows shell
    nor WSL's interop can mangle quoting — which it does, repeatedly, to
    anything containing $VAR, // or a heredoc."""
    try:
        r = subprocess.run(["wsl", "-u", "root", "bash", "-lc", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        log(f"wsl call failed: {e}")
        return ""


def port_open(p):
    try:
        socket.create_connection(("127.0.0.1", p), timeout=2).close()
        return True
    except OSError:
        return False


def trainers():
    """variant -> pid, for trainers currently alive."""
    out = {}
    for line in wsl("ps -eo pid,args").splitlines():
        if "train_rl.py" in line and "--variant" in line and "bash -lc" not in line:
            parts = line.split()
            try:
                out[parts[parts.index("--variant") + 1]] = int(parts[0])
            except (ValueError, IndexError):
                pass
    return out


def start_trainer(v):
    log(f"starting trainer {v}")
    # The trailing `sleep 3` is load-bearing. With a bare `&` the shell exits
    # immediately and the child is killed before it ever execs — this is
    # already written up in FINDINGS ("collect games silently did nothing"),
    # and I reproduced it here anyway. Keep the sleep.
    cmd = (f"cd {PROJ} && nohup setsid {WSL_PY} -u train_rl.py "
           f"--variant {v} --envs {ENVS_PER_VARIANT} --rollout 32 --lam 0.97 "
           f"--updates 100000 --max-steps 1000 --target-depth 5 "
           f"--threads {THREADS} --resume "
           f">> rl_train.{v}.log 2>&1 < /dev/null & sleep 3")
    wsl(cmd, timeout=45)
    if v in trainers():
        log(f"trainer {v} up")
    else:
        log(f"trainer {v} FAILED to start — see rl_train.{v}.log")


def dashboard_alive():
    return port_open(PORT)


def start_dashboard():
    log("starting dashboard")
    # Output MUST be redirected: inheriting console handles is what silently
    # killed this server before.
    subprocess.Popen(
        [sys.executable, str(HERE / "project.py"), "--serve"],
        cwd=str(HERE),
        stdout=open(HERE / "panel.out.log", "ab"),
        stderr=open(HERE / "panel.err.log", "ab"),
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def reap():
    out = wsl(f"cd {PROJ} && {WSL_PY} reap.py --kill", timeout=60)
    for line in out.splitlines():
        if "killed" in line:
            log(f"reaper: {line.strip()}")


def status():
    tr = trainers()
    print(f"dashboard :8099   {'up' if dashboard_alive() else 'DOWN'}")
    for v in VARIANTS:
        print(f"trainer {v}        {'pid ' + str(tr[v]) if v in tr else 'DOWN'}")
    games = wsl("pgrep -cx crawl").strip() or "0"
    print(f"crawl games       {games}")
    for v in VARIANTS:
        p = HERE / "data" / f"rl_log.{v}.jsonl"
        if not p.exists():
            continue
        rows = [l for l in p.read_text(errors="replace").splitlines() if l.strip()]
        if rows:
            r = json.loads(rows[-1])
            print(f"  {v}: update {r['update']:5d}  depth {r['mean_depth']:.2f}  "
                  f"D:5 {100*r['solve_rate']:3.0f}%  return {r['mean_return']:+.2f}")


def stop():
    for v, pid in trainers().items():
        log(f"stopping trainer {v} (pid {pid})")
        wsl(f"kill -TERM {pid}")
    time.sleep(8)
    left = trainers()
    for v, pid in left.items():
        wsl(f"kill -9 {pid}")
    reap()
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                    "Where-Object { $_.CommandLine -like '*project.py*' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   capture_output=True)
    log("stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    args = ap.parse_args()

    if args.status:
        status()
        return 0
    if args.stop:
        stop()
        return 0

    log("supervisor started")
    started_at = {}
    while True:
        try:
            if not dashboard_alive():
                if time.time() - started_at.get("dash", 0) > GRACE:
                    start_dashboard()
                    started_at["dash"] = time.time()

            alive = trainers()
            for v in VARIANTS:
                if v not in alive and time.time() - started_at.get(v, 0) > GRACE:
                    start_trainer(v)
                    started_at[v] = time.time()

            # Orphans only appear when something died, so this is cheap and
            # only does work when there is work to do.
            if len(alive) < len(VARIANTS):
                reap()
        except Exception as e:                    # never let the supervisor die
            log(f"check failed: {type(e).__name__}: {e}")
        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    raise SystemExit(main())

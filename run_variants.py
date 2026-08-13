"""Launch the three equipment variants side by side, splitting the machine.

Each variant is an independent PPO run with its own envs, save directories,
checkpoint, log and live feed. They differ ONLY in how equipment is handled
(see dcss_env.VARIANTS), so the comparison is meaningful.

    /root/pty-venv/bin/python run_variants.py --envs 16
    /root/pty-venv/bin/python run_variants.py --stop
"""
import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = "/root/pty-venv/bin/python"
VARIANTS = ["a", "b", "c"]


def running():
    """pid -> variant, for trainers we started."""
    out = {}
    r = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "train_rl.py" in line and "--variant" in line:
            parts = line.split()
            try:
                pid = int(parts[0])
                v = parts[parts.index("--variant") + 1]
            except (ValueError, IndexError):
                continue
            out[pid] = v
    return out


def stop(timeout=25):
    """Kill every trainer and CONFIRM it is gone.

    Fire-and-forget SIGTERM is not enough: three trainers once survived a
    "restart", kept writing to the same logs for four hours, and the report I
    gave described their numbers as if they came from the new code. Escalate to
    SIGKILL, then verify the process table is actually clear before returning.
    """
    procs = running()
    if not procs:
        subprocess.run([PY, str(HERE / "reap.py"), "--kill"])
        return
    print(f"stopping {len(procs)} trainer(s): {sorted(procs.values())}")
    for pid in procs:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    deadline = time.time() + timeout
    while time.time() < deadline and running():
        time.sleep(2)

    left = running()
    if left:
        print(f"  SIGTERM ignored by {sorted(left)}, sending SIGKILL")
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(3)

    left = running()
    if left:
        raise SystemExit(f"REFUSING TO START: trainers still alive: {left}")
    print("  all trainers confirmed stopped")
    subprocess.run([PY, str(HERE / "reap.py"), "--kill"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16,
                    help="envs PER VARIANT (three variants share the machine)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="start from random weights instead of --resume. Use "
                         "this whenever the REWARD FUNCTION changed: a policy "
                         "trained under the old numbers carries habits the new "
                         "ones never paid for, and the resulting curve belongs "
                         "to neither reward.")
    args = ap.parse_args()

    if args.stop:
        stop()
        return 0

    stop()
    for v in VARIANTS:
        (HERE / "data").mkdir(exist_ok=True)
        (HERE / "data" / f"rl_view.{v}.txt").write_text("0")
        log = HERE / f"rl_train.{v}.log"
        cmd = [PY, "-u", str(HERE / "train_rl.py"),
               "--variant", v,
               "--envs", str(args.envs),
               "--rollout", str(args.rollout),
               "--lam", "0.97",
               "--updates", str(args.updates),
               "--max-steps", str(args.max_steps),
               "--target-depth", "5",
               "--threads", str(args.threads)]
        if not args.fresh:
            cmd.append("--resume")
        with open(log, "wb") as f:
            subprocess.Popen(cmd, cwd=str(HERE), stdout=f, stderr=f,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"variant {v}: {args.envs} envs -> {log.name}")

    time.sleep(50)
    procs = running()
    print(f"\nalive: {len(procs)} trainer(s) {sorted(procs.values())}")
    for v in VARIANTS:
        log = HERE / f"rl_train.{v}.log"
        lines = [l for l in log.read_text(errors="replace").splitlines()
                 if l.startswith("u0") or "variant" in l or "resumed" in l]
        print(f"--- {v} ---")
        for l in lines[-2:]:
            print("   " + l[:130])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

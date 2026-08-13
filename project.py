"""
DCSS agent project — control panel.

    python project.py            # static project.html
    python project.py --serve    # live panel on http://localhost:8099

Static mode gives sorting and filtering. Serve mode adds the things that touch
disk or processes: starting runs, stopping them, pruning games, live logs.

Pruning ARCHIVES rather than deletes — rows move to games.pruned.jsonl.
"""
import argparse
import html
import json
import math
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).parent
DATA = HERE / "data"
AR = HERE / "autoresearch"
OUT = HERE / "project.html"
GAMES = HERE / "games.jsonl"
PRUNED = HERE / "games.pruned.jsonl"
TRACES = HERE / "data" / "traces.jsonl"
RL_LOG = HERE / "data" / "rl_log.jsonl"
RL_ARCHIVE = HERE / "data" / "archive"
RL_REPLAYS = HERE / "data" / "rl_replays"
RL_TILES = HERE / "data" / "tiles"
RL_LIVE = HERE / "data" / "rl_live.json"
RL_ENVS = HERE / "data" / "rl_envs.json"
RL_VIEW = HERE / "data" / "rl_view.txt"
RECORDINGS = HERE / "recordings"
RL_ACTIONS = ["explore", "autofight", "travel", "descend", "rest", "escape"]
MAX_ENTROPY = 1.7918          # ln(6): a uniformly random policy over 6 actions
# Episodes needed in the rolling window before its numbers are worth showing at
# full confidence. The window holds 60; half of it is enough that one lucky run
# cannot dominate, and it is past the point where slow-failing episodes have
# started landing in the sample.
MIN_EPISODES = 30

# Why each archived run was stopped. Default is "superseded" — NOT "failed",
# because a run stopped to change hardware settings is not the same thing as a
# run that learned the wrong objective, and collapsing the two would overstate
# how many reward functions were actually broken.
RL_VERDICTS = {
    "hpshaped": "reward hacked — HP shaping made standing still optimal",
    "noopheavy": "reward hacked — no-op penalty outweighed the goal",
    "run3a-8env": "reward v3 · reached D:5 · stopped to raise 8→24 envs, weights kept",
}
PORT = 8099
WEBTILES = "http://localhost:8090"

MODEL = Path(r"C:\Users\jorda\models\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf")
STEP_RE = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+)\s+elapsed\s+(\d+)s")
WSL = ["wsl", "-d", "Ubuntu", "--", "bash", "-lc"]


# ───────────────────────── probes ─────────────────────────

def port_open(p, host="127.0.0.1"):
    try:
        socket.create_connection((host, p), timeout=1.2).close()
        return True
    except OSError:
        return False


def wsl_ok(cmd, timeout=20):
    try:
        r = subprocess.run(WSL + [cmd], capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


def wsl_out(cmd, timeout=20):
    try:
        r = subprocess.run(WSL + [cmd], capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout
    except Exception:
        return ""


def ps_processes():
    """Match on process NAME first so the probe can't detect itself."""
    q = ("Get-CimInstance Win32_Process | Where-Object { "
         "($_.Name -eq 'llama-server.exe') -or "
         "(($_.Name -like 'python*') -and ($_.CommandLine -like '*train_dcss*')) } "
         "| Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", q],
                           capture_output=True, text=True, timeout=20)
        if not r.stdout.strip():
            return []
        d = json.loads(r.stdout)
        return d if isinstance(d, list) else [d]
    except Exception:
        return []


def active_runs():
    runs = []
    for p in ps_processes():
        name = (p.get("Name") or "").lower()
        runs.append({
            "kind": "qwen" if name.startswith("llama-server") else "training",
            "pid": p.get("ProcessId"),
            "detail": (p.get("CommandLine") or "")[:110],
            "stoppable": True,
        })
    if port_open(8090):
        runs.append({"kind": "webtiles", "pid": None,
                     "detail": "DCSS server :8090", "stoppable": False})

    n = wsl_out("ps -eo args | grep -c '[p]ty_agent.py'").strip() or "0"
    try:
        n = int(n)
    except ValueError:
        n = 0
    if n:
        runs.append({"kind": "collectors", "pid": "wsl",
                     "detail": f"{n} pty worker(s) collecting games",
                     "stoppable": True})
    if wsl_ok("ps -eo args | grep '[w]ebtiles_agent.py' | head -1"):
        runs.append({"kind": "player", "pid": "wsl",
                     "detail": "webtiles agent (watchable)", "stoppable": True})

    # The PPO run. This lives in WSL and is NOT matched by the Windows probe
    # above (which looks for train_dcss, the imitation trainer), so without
    # this the panel reported "nothing running" through an entire overnight
    # training session.
    if wsl_ok("ps -eo args | grep '[t]rain_rl.py' | head -1"):
        rl = rl_status().get("summary") or {}
        envs = wsl_out("pgrep -cx crawl").strip() or "?"
        detail = f"PPO · {envs} games in parallel"
        if rl:
            detail += (f" · update {rl['update']} · mean depth {rl['mean_depth']}"
                       f" · D:5 {100*rl['solve_rate']:.0f}%")
        runs.append({"kind": "rl-training", "pid": "wsl",
                     "detail": detail, "stoppable": True})
    return runs


def gather_status():
    return [
        {"name": "WSL", "ok": wsl_ok("echo ok")},
        {"name": "DCSS", "ok": wsl_ok("test -x /root/crawl/crawl-ref/source/crawl && echo ok")},
        {"name": "webtiles", "ok": port_open(8090)},
        # Qwen is the local CHAT model. It plays no part in the DCSS agent —
        # it is listed only because it holds GPU memory that training needs.
        {"name": "Qwen chat (GPU)", "ok": port_open(8080)},
        {"name": "gguf", "ok": MODEL.exists()},
        {"name": "autoresearch", "ok": (AR / ".venv").exists()},
    ]


# ───────────────────────── data ─────────────────────────

def parse_run_log():
    curve, final = [], {}
    p = AR / "run.log"
    if not p.exists():
        return curve, final
    t = p.read_text(encoding="utf-8", errors="replace")
    for m in STEP_RE.finditer(t):
        curve.append([int(m.group(3)), float(m.group(2))])
    for k in ("val_action_loss", "val_top1", "memory_gb", "steps"):
        m = re.search(rf"^{k}:\s*([\d.]+)", t, re.M)
        if m:
            final[k] = float(m.group(1))
    m = re.search(r"majority baseline: val_top1=([\d.]+) val_action_loss~([\d.]+)", t)
    if m:
        final["base_top1"], final["base_loss"] = float(m.group(1)), float(m.group(2))
    return curve, final


def parse_results():
    rows, p = [], AR / "results.tsv"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        c = line.rstrip("\n").split("\t")
        if len(c) < 2 or c[0].lower().startswith("commit"):
            continue
        try:
            loss = float(c[1])
        except ValueError:
            continue
        rows.append({"commit": c[0][:8], "loss": loss,
                     "top1": c[2] if len(c) > 2 else "",
                     "status": c[4] if len(c) > 4 else "",
                     "desc": c[5] if len(c) > 5 else ""})
    return rows


SKIP_SHARDS = {"games.jsonl", "games.pruned.jsonl"}


def _fill(g):
    for k, v in (("turns", 0), ("xl", 0), ("depth", 0), ("score", 0),
                 ("agent", ""), ("death", ""), ("ts", ""), ("actions", 0),
                 ("source", "pty"), ("game", "")):
        g.setdefault(k, v)
    return g


def read_games():
    """Canonical games PLUS anything sitting in un-merged worker shards.

    Without the shard half, a game collected from the dashboard is invisible
    until someone remembers to hit "merge shards" — which makes the collect
    button look broken even when it worked perfectly.

    Shard rows get `_id = None` so they can't be selected for archiving; they
    aren't line-addressable in games.jsonl yet.
    """
    games = []
    if GAMES.exists():
        for i, line in enumerate(GAMES.read_text(encoding="utf-8",
                                                 errors="replace").splitlines()):
            if not line.strip():
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            g["_id"] = i
            games.append(_fill(g))

    for shard in sorted(p for p in HERE.glob("games.*.jsonl")
                        if p.name not in SKIP_SHARDS):
        for line in shard.read_text(encoding="utf-8",
                                    errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            g["_id"] = None
            g["pending"] = True
            games.append(_fill(g))

    _mark_duplicates(games)
    return games


def _mark_duplicates(games):
    """Flag rows a worker produced by resuming a WEDGED save.

    `pty_agent` used to reuse one save per worker, so once a game got stuck the
    next session reloaded it and reported byte-identical stats. Those rows are
    one game counted many times. Marking beats silently dropping them: the
    count in the table is evidence about the collector, and hiding it is how a
    dashboard ends up looking healthier than the data underneath it.
    """
    seen = {}
    for g in games:
        m = re.match(r"\d{8}-\d{6}-(.+?)-\d+$", g.get("game") or "")
        key = (m.group(1) if m else "?", g.get("turns"), g.get("xl"),
               g.get("depth"), g.get("actions"))
        if key in seen and g.get("turns"):
            g["dup_of"] = seen[key]
        else:
            seen[key] = g.get("game")


def prune_games(ids):
    ids = {int(i) for i in ids}
    if not GAMES.exists() or not ids:
        return 0
    kept, moved = [], []
    for i, line in enumerate(GAMES.read_text(encoding="utf-8",
                                             errors="replace").splitlines()):
        if not line.strip():
            continue
        (moved if i in ids else kept).append(line)
    if not moved:
        return 0
    with open(PRUNED, "a", encoding="utf-8") as f:
        f.write("\n".join(moved) + "\n")
    GAMES.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(moved)


def data_health():
    """Is the training data worth training on?

    The load-bearing check is label entropy. Data from a random agent has no
    relationship between screen and key, so nothing can learn it — while every
    other panel still looks perfectly healthy.
    """
    out = {"rows": 0, "actions": {}, "unique_screens": 0, "entropy": 0.0,
           "level": "ok", "note": "", "games": 0, "bytes": 0}
    if not TRACES.exists():
        out["level"] = "warn"
        out["note"] = "No traces captured yet."
        return out

    counts, screens, games, n = {}, set(), set(), 0
    with open(TRACES, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = o.get("action")
            if a is None:
                continue
            counts[a] = counts.get(a, 0) + 1
            n += 1
            screens.add(hash(o.get("state") or ""))
            if o.get("game"):
                games.add(o["game"])

    out["rows"] = n
    out["actions"] = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:12])
    out["unique_screens"] = len(screens)
    out["games"] = len(games)
    out["bytes"] = TRACES.stat().st_size

    if n and len(counts) > 1:
        h = -sum((c / n) * math.log(c / n) for c in counts.values())
        out["entropy"] = round(h / math.log(len(counts)), 3)

    if n > 50 and out["entropy"] >= 0.97:
        out["level"] = "bad"
        out["note"] = (f"Labels are near-uniform (entropy {out['entropy']} of max). "
                       "This is what data from a RANDOM agent looks like: no "
                       "relationship exists between screen and key, so no model "
                       "can learn it. Collect from a competent teacher first.")
    elif n and out["unique_screens"] < max(2, n * 0.2):
        out["level"] = "warn"
        out["note"] = (f"Only {out['unique_screens']} unique screens across {n} "
                       "rows — the agent may be repeating a state.")
    elif n:
        out["note"] = f"{out['unique_screens']} unique screens across {n} rows."
    return out


def storage():
    files = list(RECORDINGS.glob("*.ttyrec")) if RECORDINGS.exists() else []
    return {
        "ttyrecs": len(files),
        "ttyrec_bytes": sum(f.stat().st_size for f in files),
        "traces_bytes": TRACES.stat().st_size if TRACES.exists() else 0,
        "games_bytes": GAMES.stat().st_size if GAMES.exists() else 0,
    }


LOGS = {
    "collectors": ("win", str(HERE / "fleet.log")),
    "player": ("win", str(HERE / "pty.log")),
    "webtiles": ("wsl", "/tmp/webtiles.log"),
    "training": ("win", str(AR / "run.log")),
}


def tail_log(which, lines=150):
    spec = LOGS.get(which)
    if not spec:
        return f"unknown log: {which}"
    kind, path = spec
    try:
        if kind == "wsl":
            return wsl_out(f"tail -n {lines} {path} 2>/dev/null | cut -c1-300") \
                   or "(empty)"
        p = Path(path)
        if not p.exists():
            return "(no log yet)"
        txt = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(l[:300] for l in txt[-lines:]) or "(empty)"
    except Exception as e:
        return f"error reading log: {e}"


def _wsl_launch(cmd):
    """Fire a backgrounded WSL job so that it actually survives.

    Two things are load-bearing here, both learned the hard way:

    1. subprocess.run, not Popen. With Popen the wsl.exe process is reaped
       before bash finishes, and nothing starts — while the caller still gets
       a cheerful success message.
    2. A trailing `sleep`. If the command ends on a bare `&`, bash exits
       immediately and the backgrounded child is torn down with the session
       before it has execed. `ops.py fleet` worked only because it happened to
       end with `sleep 1; ps ...`.

    Returns the launcher's own output so the caller can confirm a live pid.
    """
    r = subprocess.run(WSL + [cmd + " sleep 2; pgrep -f pty_agent.py | wc -l"],
                       capture_output=True, text=True, timeout=90)
    return (r.stdout or "").strip()


def start_run(what, n=1, extra=0):
    try:
        if what == "games":
            # A fresh worker name per launch: a wedged save under a reused name
            # makes every later run exit instantly at turn 0.
            tag = f"ui{int(time.time()) % 100000}"
            cmd = (f"cd /mnt/c/Users/jorda/dcss-research && "
                   f"nohup /root/pty-venv/bin/python pty_agent.py --games {int(n)} "
                   f"--max-actions {int(extra) or 220} --name {tag} --tag {tag} "
                   f"--no-ttyrec >> fleet.log 2>&1 &")
            live = _wsl_launch(cmd)
            return (f"collecting {n} game(s) as {tag}"
                    f" — {live or '?'} worker(s) live")
        if what == "watch":
            cmd = ("cd /mnt/c/Users/jorda/dcss-research && "
                   "nohup /root/pty-venv/bin/python webtiles_agent.py "
                   f"--actions {int(n) or 400} --delay 0.5 >> pty.log 2>&1 &")
            _wsl_launch(cmd)
            return "watchable game started — open webtiles"
        if what == "training":
            subprocess.Popen(["python", str(AR / "train_dcss.py")], cwd=str(AR),
                             stdout=open(AR / "run.log", "wb"),
                             stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return "training started (5 minute budget)"
        if what == "merge":
            r = subprocess.run(["python", str(HERE / "ops.py"), "merge"],
                               capture_output=True, text=True, timeout=120)
            return r.stdout.strip() or "merged"
        return f"unknown: {what}"
    except Exception as e:
        return f"failed: {e}"


def stop_run(kind, pid):
    try:
        if kind == "collectors":
            wsl_out("pkill -f pty_agent.py")
            return "collectors stopped"
        if kind == "player":
            wsl_out("pkill -f webtiles_agent.py")
            return "player stopped"
        if kind == "rl-training":
            # Bracket the pattern so the pkill command line does not match
            # ITSELF — an unbracketed `pkill -f train_rl.py` kills the shell
            # running it and the real process survives.
            wsl_out("pkill -f '[t]rain_rl.py'")
            # Killing the trainer orphans its crawl children, which then burn
            # a core each indefinitely. This has happened before.
            wsl_out("sleep 3; pkill -9 -x crawl")
            return "PPO training stopped (weights kept in data/rl_policy.pt)"
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Id {int(pid)} -Force"], timeout=20)
        return f"stopped pid {pid}"
    except Exception as e:
        return f"failed: {e}"


def live_game():
    """Game id of a webtiles game in progress, if any."""
    if not port_open(8090):
        return None
    out = wsl_out("ls /root/crawl/crawl-ref/source/rcs/running/ 2>/dev/null")
    return out.strip().splitlines()[0] if out.strip() else None


def replay_frames(game_id, limit=2000):
    """Screens for one game. We already store the full screen with every
    logged action, so replay needs no extra recording and no terminal
    emulator in the browser."""
    # RL episodes are stored one file per game rather than appended to the
    # shared trace log: the PPO run produces ~30 screens/second, and mixing
    # that into traces.jsonl would both dwarf the collected data and skew the
    # action-distribution panel, which is the tool that diagnoses reward bugs.
    # Episodes live in per-variant directories (rl_replays_a/_b/_c). This used
    # to look only in the pre-variant `rl_replays/`, so every RL replay 404'd
    # and the viewer reported "replay no longer stored" for games that were
    # sitting on disk. Search all of them, newest layout first.
    rl = next((p for p in
               [RL_REPLAYS / f"{game_id}.jsonl"]
               + sorted(DATA.glob(f"rl_replays_*/{game_id}.jsonl"))
               if p.exists()), None)
    if rl is not None:
        out = []
        for line in rl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = {"t": o.get("t", 0), "action": o.get("action", ""),
                     "screen": o.get("state", "")}
            # Pass the network's own output through to the viewer. Dropping
            # these here is what made the activation panel silently empty even
            # though the trainer was recording them correctly.
            for k in ("probs", "value", "sal", "colors"):
                if k in o:
                    frame[k] = o[k]
            out.append(frame)
            if len(out) >= limit:
                break
        return out

    if not TRACES.exists():
        return []
    out = []
    with open(TRACES, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or game_id not in line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("game") != game_id:
                continue
            out.append({"t": o.get("t", 0), "action": o.get("action", ""),
                        "screen": o.get("state", "")})
            if len(out) >= limit:
                break
    return out


def parse_findings():
    p = HERE / "FINDINGS.md"
    if not p.exists():
        return []
    blocks, cur = [], None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("## "):
            if cur:
                blocks.append(cur)
            cur = {"title": line[3:].strip(), "body": []}
        elif cur is not None:
            cur["body"].append(line)
    if cur:
        blocks.append(cur)
    for b in blocks:
        b["body"] = "\n".join(b["body"]).strip()
    return blocks


_rl_cache = {}


def _vsafe(v):
    """Variant ids come from the URL and are interpolated into filenames, so
    anything not in the known set becomes 'a' rather than a path fragment."""
    return v if v in RL_VARIANTS else "a"


def machine_stats():
    """GPU, CPU, RAM, disk — what the box is actually doing right now.

    GPU comes from nvidia-smi (the WMI AdapterRAM field is a 32-bit value and
    reports 4GB on this 8GB card). CPU/RAM come from WSL, which is where the
    training and all 24 games actually live — Windows-side counters would miss
    them entirely.
    """
    out = {}
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=12)
        parts = [p.strip() for p in (r.stdout or "").strip().split(",")]
        if len(parts) >= 7:
            out["gpu"] = {
                "name": parts[0], "util": _num(parts[1]),
                "mem_used": _num(parts[2]), "mem_total": _num(parts[3]),
                "temp": _num(parts[4]), "power": _num(parts[5]),
                "power_cap": _num(parts[6]),
            }
    except Exception:
        pass

    try:
        txt = wsl_out("nproc; cat /proc/loadavg; free -m | sed -n 2p; "
                      "df -m /mnt/c | sed -n 2p", timeout=15)
        lines = [l for l in (txt or "").splitlines() if l.strip()]
        if len(lines) >= 4:
            cores = int(lines[0].strip())
            load = [float(x) for x in lines[1].split()[:3]]
            mem = lines[2].split()
            disk = lines[3].split()
            out["cpu"] = {"cores": cores, "load1": load[0], "load5": load[1],
                          "load15": load[2],
                          "pct": round(100 * load[0] / max(1, cores))}
            out["ram"] = {"total": int(mem[1]), "used": int(mem[2]),
                          "pct": round(100 * int(mem[2]) / max(1, int(mem[1])))}
            out["disk"] = {"total_gb": round(int(disk[1]) / 1024),
                           "free_gb": round(int(disk[3]) / 1024),
                           "pct": int(disk[4].rstrip("%"))}
    except Exception:
        pass

    try:
        out["games"] = int((wsl_out("pgrep -cx crawl", timeout=10) or "0").strip())
    except (ValueError, AttributeError):
        out["games"] = 0
    return out


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _rl_rows(path):
    """Parse an rl_log jsonl, memoised on (mtime, size).

    Archived runs never change but are re-read on every page load otherwise,
    and there are now several thousand rows across them. Keyed on stat rather
    than name so the LIVE log still refreshes as it grows.
    """
    try:
        st = path.stat()
    except OSError:
        return []
    key = (str(path), st.st_mtime, st.st_size)
    hit = _rl_cache.get(str(path))
    if hit and hit[0] == key:
        return hit[1]

    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    _rl_cache[str(path)] = (key, rows)
    return rows


RL_VARIANTS = {
    "a": "env picks the item",
    "b": "agent reads menu, picks",
    "c": "env does everything",
}


def rl_variants():
    """All three equipment variants, for side-by-side comparison.

    They share the machine and differ only in how equipment is handled, so the
    dashboard shows them together — a single blended number would hide exactly
    the difference the experiment exists to measure.
    """
    out = {}
    for v, label in RL_VARIANTS.items():
        s = rl_status(DATA / f"rl_log.{v}.jsonl")
        s["label"] = label
        s["variant"] = v
        s["live"] = _rl_recent(DATA / f"rl_live.{v}.json")
        out[v] = s
    return out


def rl_status(path=None):
    """The PPO run: learning curve, action mix, and whether it is alive.

    Reports the ARCHIVED failed runs alongside the current one. Two reward
    functions have already been discarded here, and a panel that quietly shows
    only the latest attempt would present a fresh curve as if it were the whole
    story — which is exactly the mistake that made an earlier version of this
    dashboard look healthy while the data underneath was unlearnable.
    """
    rows = _rl_rows(path or RL_LOG)
    past = []
    if RL_ARCHIVE.exists():
        for p in sorted(RL_ARCHIVE.glob("rl_log.*.jsonl")):
            r = _rl_rows(p)
            if r:
                past.append({
                    "name": p.name.replace("rl_log.", "").replace(".jsonl", ""),
                    "updates": len(r),
                    "best_depth": max(x.get("best_depth", 0) for x in r),
                    "peak_mean_depth": round(max(x.get("mean_depth", 0) for x in r), 2),
                    "final_entropy": r[-1].get("entropy"),
                    "verdict": RL_VERDICTS.get(
                        p.name.replace("rl_log.", "").replace(".jsonl", ""),
                        "superseded"),
                })

    if not rows:
        return {"rows": [], "past": past, "live": _rl_recent(), "summary": None}

    # Only the CURRENT run. `update` restarts at 1 on every resume, so a log
    # holding two runs plots a line that jumps back to x=1 — which is exactly
    # what the depth chart was showing. Rows written before the `run` field
    # existed fall back to a split wherever the update number decreases.
    if any("run" in r for r in rows):
        last_run = rows[-1].get("run")
        rows = [r for r in rows if r.get("run") == last_run]
    else:
        start = 0
        for i in range(1, len(rows)):
            if rows[i]["update"] <= rows[i - 1]["update"]:
                start = i
        rows = rows[start:]

    # Downsample for the browser; the shape matters, not every point.
    step = max(1, len(rows) // 300)
    thin = rows[::step]
    if thin[-1] is not rows[-1]:          # keep the newest point, never twice
        thin.append(rows[-1])

    last = rows[-1]
    tot = max(1, sum(last.get("actions", {}).values()))
    mix = [{"name": k, "pct": round(100 * last.get("actions", {}).get(k, 0) / tot, 1)}
           for k in RL_ACTIONS]
    recent = rows[-20:]
    return {
        "rows": [{"u": r["update"], "depth": r["mean_depth"], "ret": r["mean_return"],
                  "ent": r["entropy"], "solve": r["solve_rate"],
                  "best": r["best_depth"]} for r in thin],
        "past": past,
        # Deliberately NOT active_runs(): that shells into WSL and costs ~2s,
        # and payload() already pays for it once. A log written to in the last
        # three minutes is the same answer for free.
        "live": _rl_recent(),
        "summary": {
            "update": last["update"], "steps": last["steps"],
            "elapsed_min": last["elapsed_s"] // 60,
            "mean_depth": last["mean_depth"], "best_depth": last["best_depth"],
            "solve_rate": last["solve_rate"], "entropy": last["entropy"],
            # ln(number of actions) — the entropy of a uniformly random policy.
            # It differs per variant (7, 10 and 14 actions), so a fixed ln(6)
            # made variant b read as having MORE entropy than random.
            "max_entropy": (math.log(len(last["actions"]))
                            if last.get("actions") else MAX_ENTROPY),
            "mean_return": last["mean_return"],
            "trend_depth": round(sum(r["mean_depth"] for r in recent) / len(recent), 2),
        "equips": last.get("equips"), "berserks": last.get("berserks"),
        "berserk_wasted": last.get("berserk_wasted"),
        "mean_ac": last.get("mean_ac"), "ac_gained": last.get("ac_gained"),
        "mean_wpn": last.get("mean_wpn"), "wpn_gained": last.get("wpn_gained"),
            # SAMPLE SIZE for every per-episode number on this card. solve_rate,
            # mean_depth, mean_return and outcomes all come from ONE rolling
            # deque(maxlen=60) of finished episodes, so they share this n and
            # they are all meaningless while it is small.
            #
            # Worse than meaningless: biased UP. An episode that reaches D:5
            # ends the moment it gets there, while one heading for the step
            # limit runs ~31 updates before it finishes — so an unfilled window
            # holds the fast endings and excludes the slow failures. Measured:
            # variant c read 75% solve / 4.25 mean depth at n=4, and settled at
            # 7% / 2.38 once n reached 60. The dashboard showed the 75% as its
            # headline and highlighted that variant as the leader.
            "n_episodes": sum(last.get("outcomes", {}).values()),
            "n_trusted": MIN_EPISODES,
            "outcomes": last.get("outcomes", {}),
            "mix": mix,
        },
    }


def _rl_recent(path=None, window=180):
    """True if the file was written to recently. Cheaper and more honest than
    probing for a process across the WSL boundary."""
    try:
        return (time.time() - (path or RL_LOG).stat().st_mtime) < window
    except OSError:
        return False


def payload(served):
    curve, final = parse_run_log()

    # These four each shell out to WSL or enumerate processes, and measured
    # 2.56 + 1.71 + 1.21 + 1.18 = 6.66s run one after another — longer than
    # the cache TTL, so EVERY request paid the full cost and the cache never
    # once hit. They do not depend on each other, and they are pure I/O wait,
    # so the wall time is now the slowest one rather than the sum.
    slow = {"status": gather_status, "runs": active_runs,
            "health": data_health, "live": live_game,
            "machine": machine_stats}
    with ThreadPoolExecutor(max_workers=len(slow)) as ex:
        futs = {k: ex.submit(f) for k, f in slow.items()}
        got = {k: f.result() for k, f in futs.items()}

    return {
        "served": served,
        "stamp": time.strftime("%H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "status": got["status"],
        "runs": got["runs"],
        "games": read_games(),
        "experiments": parse_results(),
        "curve": curve,
        "final": final,
        "health": got["health"],
        "rl": rl_status(),
        "variants": rl_variants(),
        "storage": storage(),
        "live": got["live"],
        "machine": got["machine"],
        "webtiles": WEBTILES,
        "findings": parse_findings(),
    }


# ───────────────────────── rendering ─────────────────────────

def md_inline(s):
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def findings_html(findings):
    out = []
    for f in findings:
        paras = "".join("<p>" + md_inline(p.strip()).replace("\n", " ") + "</p>"
                        for p in re.split(r"\n\s*\n", f["body"]) if p.strip())
        out.append(f'<article class="finding"><h4>{md_inline(f["title"])}</h4>'
                   f'{paras}</article>')
    return "".join(out) or '<p class="muted">FINDINGS.md is empty.</p>'


CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#f7f7f5; --panel:#ffffff; --ink:#12130f; --ink-2:#54564e; --ink-3:#8b8d84;
  --line:#e4e4de; --line-2:#efefe9; --accent:#2a78d6; --accent-soft:#eaf2fd;
  --good:#0f8a34; --warn:#a9700a; --bad:#c0392f; --mono:ui-monospace,"SF Mono",Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --bg:#0e0f0d; --panel:#171815; --ink:#f2f2ee; --ink-2:#a9aba1; --ink-3:#74766e;
  --line:#262723; --line-2:#1e1f1c; --accent:#4f95e8; --accent-soft:#15243a;
  --good:#3fb45f; --warn:#d59a2a; --bad:#e0685c;
}}
:root[data-theme="dark"]{
  --bg:#0e0f0d; --panel:#171815; --ink:#f2f2ee; --ink-2:#a9aba1; --ink-3:#74766e;
  --line:#262723; --line-2:#1e1f1c; --accent:#4f95e8; --accent-soft:#15243a;
  --good:#3fb45f; --warn:#d59a2a; --bad:#e0685c;
}
body{margin:0}
.app{background:var(--bg);color:var(--ink);min-height:100vh;
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}

/* top bar */
.bar{position:sticky;top:0;z-index:20;background:var(--panel);
  border-bottom:1px solid var(--line);padding:0 22px}
.bar-in{max-width:1200px;margin:0 auto;display:flex;align-items:center;
  gap:18px;height:54px}
.brand{font-weight:640;letter-spacing:-.01em;font-size:14.5px;white-space:nowrap}
.brand span{color:var(--ink-3);font-weight:400;margin-left:8px;font-size:12px}
.pills{display:flex;gap:5px;margin-left:4px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:11px;
  color:var(--ink-2);padding:3px 8px;border:1px solid var(--line);border-radius:4px}
.pill i{width:6px;height:6px;border-radius:50%;background:var(--bad);display:block}
.pill.ok i{background:var(--good)}
.spacer{margin-left:auto}
.clock{font-family:var(--mono);font-size:11.5px;color:var(--ink-3)}

/* layout */
main{max-width:1200px;margin:0 auto;padding:20px 22px 56px}
section{margin-bottom:18px}
.head{display:flex;align-items:baseline;gap:12px;margin:0 0 9px}
.head h2{font-size:11px;font-weight:650;text-transform:uppercase;
  letter-spacing:.09em;color:var(--ink-3);margin:0}
.vpick{display:inline-flex;gap:3px;margin-right:8px}
.vpick .vb{font-size:11px;padding:3px 9px;font-family:var(--mono)}
.vpick .vb.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.vgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1px;
  background:var(--line)}
.vcard{background:var(--panel);padding:14px 16px}
.vcard.lead{background:var(--accent-soft)}
.vh{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.vh b{font-size:13px;font-family:var(--mono);color:var(--ink)}
.vh span{font-size:11px;color:var(--ink-3)}
.vh .dot{width:7px;height:7px;border-radius:50%;background:var(--line);
  margin-left:auto;flex:none}
.vh .dot.live{background:var(--good);animation:pulse 2s infinite}
.vbig{font-size:30px;font-weight:600;letter-spacing:-.02em;color:var(--ink);
  font-variant-numeric:tabular-nums;line-height:1}
.vbig em{font-size:12px;font-style:normal;color:var(--ink-3);margin-left:4px}
.vt{width:100%;margin-top:10px;font-size:11.5px}
.vt td{padding:2px 0;color:var(--ink-2)}
.vt td:last-child{text-align:right;color:var(--ink);font-variant-numeric:tabular-nums}
.envgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(72px,1fr));gap:6px}
.envcell{display:flex;flex-direction:column;align-items:stretch;gap:3px;
  padding:6px 7px;border:1px solid var(--line);border-radius:5px;
  background:var(--panel);cursor:pointer;font-size:10.5px;text-align:left}
.envcell:hover{border-color:var(--accent)}
.envcell.on{border-color:var(--accent);background:var(--accent-soft)}
.envcell .ei{color:var(--ink-3);font-family:var(--mono)}
.envcell .ed{color:var(--ink);font-weight:650;font-size:12px}
.envcell .eh{display:block;height:4px;background:var(--line);border-radius:2px;
  overflow:hidden}
.envcell .eh i{display:block;height:100%;background:var(--good)}
.envcell .eh i.low{background:var(--bad)}
.livewrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,340px);gap:18px;
  align-items:start}
@media(max-width:900px){.livewrap{grid-template-columns:1fr}}
#liveScreen{font-family:var(--mono);font-size:11px;line-height:1.18;white-space:pre;
  overflow-x:auto;margin:0;color:var(--ink);background:#0b0c0a;padding:10px;
  border-radius:6px;border:1px solid var(--line)}
#liveCanvas{max-width:100%;height:auto;image-rendering:pixelated;
  border:1px solid var(--line);border-radius:6px;background:#000}
#liveBrain{border:0;padding:0;margin:0}
#liveBrain .brain-grid{grid-template-columns:1fr}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px}
.mrow{display:flex;justify-content:space-between;align-items:baseline;
  font-size:11.5px;color:var(--ink-2);margin-bottom:5px}
.mrow b{font-size:15px;color:var(--ink);font-variant-numeric:tabular-nums}
.mrow b.hot{color:var(--bad)}
.mtrack{height:7px;background:var(--line);border-radius:3px;overflow:hidden}
.mtrack span{display:block;height:100%;background:var(--accent);border-radius:3px}
.mtrack span.hot{background:var(--bad)}
.mdet{font-size:10.5px;margin-top:5px}
.viewsel{display:flex;gap:6px;align-items:center;margin-bottom:8px}
.viewsel button{font-size:11px;padding:3px 10px}
.viewsel button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
#tiles{max-width:100%;height:auto;image-rendering:pixelated;
  border:1px solid var(--line);border-radius:5px;background:#000}
#brain{margin-top:10px;border-top:1px solid var(--line);padding-top:10px}
.brain-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.brain-grid{grid-template-columns:1fr}}
.blabel{font-size:10px;font-weight:650;text-transform:uppercase;
  letter-spacing:.08em;color:var(--ink-3);margin-bottom:5px}
/* Action probabilities: accent-coloured so the options read at a glance, with
   the chosen action brighter still. The previous flat grey made a 10-action
   distribution look like ten dead rows. */
.bar-row .fill{background:var(--accent)}
.bar-row .fill.hot{background:var(--good);box-shadow:0 0 8px var(--good)}
.bar-row.picked .lab{color:var(--ink);font-weight:650}
.bar-row.picked .val{color:var(--good);font-weight:650}
.sal{display:grid;grid-template-columns:repeat(20,1fr);gap:1px}
.sal i{display:block;aspect-ratio:1;background:#2f7d5c;border-radius:1px}
.dup{display:inline-block;font-size:10px;font-weight:650;letter-spacing:.04em;
  padding:1px 5px;border-radius:4px;background:#fdf0e3;color:#8a5a1f;
  border:1px solid #f0d9bd}
.head h3{font-size:11px;font-weight:650;text-transform:uppercase;
  letter-spacing:.09em;color:var(--ink-3);margin:0}
.head .sub{font-size:11.5px;color:var(--ink-3)}
.head .right{margin-left:auto;display:flex;gap:6px;align-items:center}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:7px}
.pad{padding:14px 16px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}

/* kpis */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));
  background:var(--panel);border:1px solid var(--line);border-radius:7px;
  overflow:hidden}
.kpi{padding:12px 14px;border-right:1px solid var(--line-2)}
.kpi:last-child{border-right:none}
.kpi .k{font-size:10px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--ink-3);margin-bottom:3px}
.kpi .v{font-size:21px;font-weight:620;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.kpi .n{font-size:11px;color:var(--ink-3);margin-top:1px}
.kpi.good .v{color:var(--good)} .kpi.bad .v{color:var(--bad)}

/* controls */
button,select,input[type=search],input[type=number]{
  font:12px/1.4 inherit;color:var(--ink);background:var(--panel);
  border:1px solid var(--line);border-radius:5px;padding:5px 9px}
button{cursor:pointer}
button:hover:not(:disabled){border-color:var(--ink-3)}
button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover{filter:brightness(1.07)}
button.danger{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 45%,transparent)}
button.ghost{border-color:transparent;color:var(--accent);padding:3px 6px}
button.ghost:hover{border-color:var(--line)}
input[type=number]{width:64px}
.toolbar{display:flex;gap:7px;align-items:center;flex-wrap:wrap;
  padding:11px 16px;border-bottom:1px solid var(--line-2)}
.toolbar .grow{flex:1}
.muted{color:var(--ink-3);font-size:11.5px}
/* A number the window is too small to support. Greyed rather than hidden:
   the value is still the best guess, it just must not read as a result. */
.warn{color:var(--warn);font-size:11.5px}
.vbig.weak{color:var(--ink-3);opacity:.55}
.omix{display:flex;gap:2px;height:6px;border-radius:3px;overflow:hidden;margin:6px 0 2px}
.omix i{display:block}
.omix .died{background:var(--warn)}
.omix .solved{background:var(--accent)}
.omix .wedged{background:var(--ink-3)}

/* tables */
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{position:sticky;top:54px;background:var(--panel);text-align:left;
  font-size:10px;font-weight:650;text-transform:uppercase;letter-spacing:.07em;
  color:var(--ink-3);padding:8px 12px;border-bottom:1px solid var(--line)}
th.s{cursor:pointer;user-select:none;white-space:nowrap}
th.s:hover{color:var(--ink)}
tbody td{padding:7px 12px;border-bottom:1px solid var(--line-2);color:var(--ink-2)}
tbody tr:hover td{background:var(--line-2)}
tr.sel td{background:var(--accent-soft)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono)}
.mono{font-family:var(--mono);font-size:11.5px}
.strong{color:var(--ink)}
.tw{max-height:420px;overflow:auto}

/* runs */
.run{display:flex;align-items:center;gap:11px;padding:9px 16px;
  border-bottom:1px solid var(--line-2);font-size:12.5px}
.run:last-child{border-bottom:none}
.run .dot{width:7px;height:7px;border-radius:50%;background:var(--good);
  animation:pulse 1.9s ease-in-out infinite;flex:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.28}}
.run .kind{font-weight:600;min-width:82px;color:var(--ink)}
.run .detail{color:var(--ink-3);font-family:var(--mono);font-size:11px;
  flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* health */
.bar-row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:11.5px}
.bar-row .lab{width:26px;font-family:var(--mono);color:var(--ink-2)}
/* Word labels (RL action names) need room; 26px is sized for the single
   keystrokes the data-health panel shows and clips "autofight" to "autof". */
.bar-row.wide .lab{width:80px;font-family:inherit;color:var(--ink-2);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-row .track{flex:1;height:9px;background:var(--line);border-radius:3px;
  overflow:hidden;border:1px solid var(--line-2)}
.bar-row .fill{height:100%;background:var(--accent);border-radius:3px}
.bar-row .val{width:52px;text-align:right;color:var(--ink-3);
  font-variant-numeric:tabular-nums}
.note{font-size:12px;padding:9px 11px;border-radius:5px;margin-top:10px;
  border:1px solid var(--line)}
.note.bad{border-color:var(--bad);background:color-mix(in srgb,var(--bad) 8%,transparent)}
.note.warn{border-color:var(--warn);background:color-mix(in srgb,var(--warn) 10%,transparent)}

/* charts */
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line-2)} .axis{stroke:var(--line)}
.ln{fill:none;stroke:var(--accent);stroke-width:1.75;stroke-linejoin:round}
.dot1{fill:var(--accent)}
.base{stroke:var(--bad);stroke-width:1.5;stroke-dasharray:4 4}
.baselab{fill:var(--bad);font-size:10px}
.tick{fill:var(--ink-3);font-size:9.5px;font-variant-numeric:tabular-nums}
.axlab{fill:var(--ink-3);font-size:10px}

/* logs */
pre.log{margin:0;padding:12px 14px;font-family:var(--mono);font-size:11px;
  line-height:1.45;color:var(--ink-2);max-height:280px;overflow:auto;
  white-space:pre-wrap;word-break:break-word}

/* findings */
.finding{padding:13px 0;border-bottom:1px solid var(--line-2)}
.finding:last-child{border-bottom:none}
.finding h4{margin:0 0 5px;font-size:13px;font-weight:620;color:var(--ink)}
.finding p{margin:0 0 7px;font-size:12.5px;line-height:1.6;color:var(--ink-2)}
code{font-family:var(--mono);font-size:11.5px;background:var(--line-2);
  padding:1px 4px;border-radius:3px}

/* modal */
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);
  z-index:60;align-items:center;justify-content:center;padding:22px}
.modal.on{display:flex}
.sheet{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  width:100%;max-width:840px;padding:14px 16px}
.sheet-head{display:flex;justify-content:space-between;align-items:center;
  gap:12px;font-size:12.5px;color:var(--ink-2);margin-bottom:10px}
#screen{font-family:var(--mono);font-size:11.5px;line-height:1.2;white-space:pre;
  overflow:auto;margin:0;background:#0c0d0b;color:#d7d8d0;padding:11px;
  border-radius:5px;min-height:330px}
.pctl{display:flex;gap:9px;align-items:center;margin:10px 0 5px}
.pctl input[type=range]{flex:1}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);
  background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:9px 15px;font-size:12.5px;box-shadow:0 6px 22px rgba(0,0,0,.2);
  opacity:0;transition:opacity .22s;pointer-events:none;z-index:80}
.toast.on{opacity:1}
"""

JS = r"""
let D = window.__DATA__;
let sortKey='ts', sortDir=-1, filter='', logName='collectors';
const sel = new Set();
const $ = s => document.querySelector(s);
const esc = s => String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const nf = n => (n||0).toLocaleString();

function toast(m,ms){const t=$('#toast');t.textContent=m;t.classList.add('on');
  clearTimeout(window.__tt);
  window.__tt=setTimeout(()=>t.classList.remove('on'),ms||2600);}

function games(){
  let g = D.games.slice();
  if(filter){const f=filter.toLowerCase();
    g=g.filter(x=>(x.agent+' '+x.death+' '+x.ts+' '+x.source).toLowerCase().includes(f));}
  g.sort((a,b)=>{let A=a[sortKey],B=b[sortKey];
    if(typeof A==='string'||typeof B==='string'){A=String(A||'');B=String(B||'');
      return A<B?-sortDir:A>B?sortDir:0;}
    return ((A||0)-(B||0))*sortDir;});
  return g;
}

function renderBar(){
  $('#pills').innerHTML = D.status.map(s=>
    `<span class="pill ${s.ok?'ok':''}"><i></i>${esc(s.name)}</span>`).join('');
  $('#clock').textContent = D.date+' '+D.stamp;
  $('#watchBtn').style.display = D.live ? '' : 'none';
}

function renderKpis(){
  const g=D.games, f=D.final||{}, h=D.health, st=D.storage;
  const k=(lab,val,note,tone='')=>`<div class="kpi ${tone}"><div class="k">${lab}</div>
    <div class="v">${val}</div><div class="n">${note||''}</div></div>`;
  const turns=g.map(x=>x.turns||0);
  const out=[];
  out.push(k('games', nf(g.length), g.length?`${nf(D.health.games)} tagged`:'none yet'));
  out.push(k('best turns', nf(turns.length?Math.max(...turns):0), 'longest run'));
  out.push(k('median turns', nf(turns.length?turns.sort((a,b)=>a-b)[Math.floor(turns.length/2)]:0),'typical'));
  out.push(k('traces', nf(h.rows), 'state/action pairs'));
  out.push(k('label entropy', (h.entropy||0).toFixed(3),
    h.level==='bad'?'no signal':'of maximum', h.level==='bad'?'bad':''));
  if(f.val_action_loss!=null){
    const good=f.base_loss!=null&&f.val_action_loss<f.base_loss;
    out.push(k('val loss', f.val_action_loss.toFixed(4),
      good?'beats baseline':'worse than baseline', good?'good':'bad'));
  }
  out.push(k('disk', ((st.ttyrec_bytes+st.traces_bytes)/1048576).toFixed(1)+' MB',
    `${st.ttyrecs} recordings`));
  $('#kpis').innerHTML = out.join('');
}

function renderRuns(){
  $('#runs').innerHTML = D.runs.length ? D.runs.map(r=>`
    <div class="run"><span class="dot"></span>
      <span class="kind">${esc(r.kind)}</span>
      <span class="detail">${esc(r.detail)}</span>
      ${r.stoppable&&D.served?`<button class="danger" data-kind="${esc(r.kind)}" data-pid="${esc(r.pid)}">stop</button>`:'<span class="muted">—</span>'}
    </div>`).join('') : '<div class="run"><span class="muted">nothing running</span></div>';
  document.querySelectorAll('#runs button').forEach(b=>{
    b.onclick=async()=>{ if(!confirm(`Stop ${b.dataset.kind}?`))return;
      b.disabled=true;
      toast(await (await fetch(`/api/stop?kind=${b.dataset.kind}&pid=${b.dataset.pid}`,{method:'POST'})).text());
      refresh();};
  });
}

function renderHealth(){
  const h=D.health, tot=Object.values(h.actions).reduce((a,b)=>a+b,0)||1;
  const max=Math.max(...Object.values(h.actions),1);
  $('#health').innerHTML =
    Object.entries(h.actions).map(([k,v])=>`
      <div class="bar-row"><span class="lab">${esc(k==='\r'?'⏎':k)}</span>
        <span class="track"><span class="fill" style="width:${(v/max*100).toFixed(1)}%"></span></span>
        <span class="val">${(v/tot*100).toFixed(1)}%</span></div>`).join('')
    + (h.note?`<div class="note ${h.level==='ok'?'':h.level}">${esc(h.note)}</div>`:'');
}

function renderGames(){
  const g=games();
  const ar=k=>sortKey===k?(sortDir<0?' ↓':' ↑'):'';
  const th=(k,l,c='')=>`<th class="s ${c}" data-k="${k}">${l}${ar(k)}</th>`;
  $('#ghead').innerHTML=`<tr><th style="width:24px"></th><th style="width:26px"></th>
    ${th('ts','when')}${th('agent','agent')}${th('source','via')}
    ${th('turns','turns','num')}${th('xl','xl','num')}${th('depth','d','num')}
    ${th('actions','keys','num')}${th('death','outcome')}</tr>`;
  $('#gbody').innerHTML = g.length ? g.map(x=>`
    <tr class="${sel.has(x._id)?'sel':''}">
      <td>${x.pending?'<span class="muted" title="in an un-merged shard">•</span>'
            :`<input type="checkbox" ${sel.has(x._id)?'checked':''} data-id="${x._id}">`}</td>
      <td>${x.game?`<button class="ghost eye" data-game="${esc(x.game)}" title="Spectate">◉</button>`:''}</td>
      <td class="mono">${esc(String(x.ts).replace('T',' ').slice(0,16))}</td>
      <td class="strong">${esc(x.agent)}</td>
      <td class="muted">${esc(x.source)}</td>
      <td class="num strong">${nf(x.turns)}</td>
      <td class="num">${x.xl||0}</td><td class="num">${x.depth||0}</td>
      <td class="num">${nf(x.actions)}</td>
      <td>${esc(x.death)}${x.dup_of?` <span class="dup" title="identical stats to ${esc(x.dup_of)} — worker resumed a wedged save">dup</span>`:''}</td></tr>`).join('')
    : `<tr><td colspan="10" class="muted" style="padding:26px;text-align:center">no games yet</td></tr>`;
  const nd=g.filter(x=>x.dup_of).length;
  $('#gcount').innerHTML=`${g.length} of ${D.games.length}`+
    (nd?` · <span class="dup">${nd} dup</span> <span class="muted">(resumed a wedged save)</span>`:'');
  $('#selc').textContent=sel.size?`${sel.size} selected`:'';
  $('#pruneBtn').disabled=!sel.size||!D.served;
  document.querySelectorAll('#ghead th.s').forEach(h=>h.onclick=()=>{
    const k=h.dataset.k; if(sortKey===k)sortDir=-sortDir; else{sortKey=k;sortDir=-1;}
    renderGames();renderCharts();});
  document.querySelectorAll('#gbody input').forEach(cb=>cb.onclick=e=>{
    const id=+e.target.dataset.id; e.target.checked?sel.add(id):sel.delete(id);
    renderGames();});
  document.querySelectorAll('#gbody .eye').forEach(b=>b.onclick=()=>openReplay(b.dataset.game));
}

function chart(el,pts,o={}){
  const w=680,h=190,pad=40;
  if(!pts.length){el.innerHTML=`<p class="muted" style="padding:26px;text-align:center">${o.empty||'no data'}</p>`;return;}
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]).concat(o.base!=null?[o.base]:[]);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(x1===x0)x1=x0+1; const sp=(y1-y0)||1; y0-=sp*.12; y1+=sp*.12;
  const X=v=>pad+(v-x0)/(x1-x0)*(w-pad-14), Y=v=>h-pad-(v-y0)/(y1-y0)*(h-pad-14);
  let s=`<svg viewBox="0 0 ${w} ${h}">`;
  for(let i=0;i<4;i++){const v=y0+(y1-y0)*i/3,y=Y(v);
    s+=`<line class="grid" x1="${pad}" y1="${y}" x2="${w-14}" y2="${y}"/>`;
    s+=`<text class="tick" x="${pad-7}" y="${y+3.5}" text-anchor="end">${v.toFixed(o.dp??0)}</text>`;}
  for(let i=0;i<4;i++){const v=x0+(x1-x0)*i/3;
    s+=`<text class="tick" x="${X(v)}" y="${h-pad+15}" text-anchor="middle">${Math.round(v)}</text>`;}
  s+=`<line class="axis" x1="${pad}" y1="${h-pad}" x2="${w-14}" y2="${h-pad}"/>`;
  if(o.base!=null){const y=Y(o.base);
    s+=`<line class="base" x1="${pad}" y1="${y}" x2="${w-14}" y2="${y}"/>`;
    s+=`<text class="baselab" x="${w-16}" y="${y-5}" text-anchor="end">baseline</text>`;}
  s+=`<path class="ln" d="${pts.map((p,i)=>(i?'L':'M')+X(p[0])+','+Y(p[1])).join(' ')}"/>`;
  const L=pts[pts.length-1];
  s+=`<circle class="dot1" cx="${X(L[0])}" cy="${Y(L[1])}" r="3"/>`;
  s+=`<text class="axlab" x="${pad}" y="${h-6}">${o.x||''}</text></svg>`;
  el.innerHTML=s;
}

function meter(label,pct,detail,warn){
  const p=Math.max(0,Math.min(100,pct||0));
  return `<div class="mtr"><div class="mrow"><span>${label}</span>
      <b class="${warn&&p>=warn?'hot':''}">${p}%</b></div>
    <div class="mtrack"><span style="width:${p}%" class="${warn&&p>=warn?'hot':''}"></span></div>
    <div class="muted mdet">${detail}</div></div>`;
}

/* ---- live game -----------------------------------------------------------
   The trainer republishes env 0's frame every step, so this polls a small file
   rather than holding a socket open. Independent of the 10s dashboard refresh:
   the game moves several times a second and the rest of the page does not.
*/
let liveTiles=false, liveTimer=null, liveLast=null, watchEnv=0, watchVar='a';

function renderVpick(){
  $('#vpick').innerHTML=Object.keys(D.variants||{a:1}).map(v=>
    `<button class="vb${v===watchVar?' on':''}" data-v="${v}">${v}</button>`).join('');
  $('#vpick').querySelectorAll('.vb').forEach(b=>b.onclick=()=>{
    watchVar=b.dataset.v; watchEnv=0; renderVpick(); pollEnvs(); pollLive();});
}

async function pollEnvs(){
  let list=[];
  try{ list=await (await fetch('/api/envs?v='+watchVar,{cache:'no-store'})).json(); }catch(e){}
  const el=$('#envGrid');
  if(!list.length){ el.innerHTML='<span class="muted">no games running</span>'; return; }
  el.innerHTML=list.map(e=>{
    const hp=Math.round(100*(e.hp??1));
    return `<button class="envcell${e.env===watchEnv?' on':''}" data-env="${e.env}"
      title="env ${e.env} · XL ${e.xl} · ${e.steps} steps${e.outcome?' · '+e.outcome:''}">
      <span class="ei">#${e.env}</span>
      <span class="ed">D:${e.depth}</span>
      <span class="eh"><i style="width:${hp}%" class="${hp<40?'low':''}"></i></span>
    </button>`;}).join('');
  el.querySelectorAll('.envcell').forEach(b=>b.onclick=async()=>{
    watchEnv=+b.dataset.env;
    await fetch(`/api/watch?v=${watchVar}&env=${watchEnv}`);
    pollEnvs();
  });
}

async function pollLive(){
  let f=null;
  try{ f=await (await fetch('/api/live?v='+watchVar,{cache:'no-store'})).json(); }catch(e){}
  if(!f){ $('#liveMeta').textContent=`variant ${watchVar}: no live game`; return; }
  // Games pause during the PPO update (24 envs x 48 steps = 1152 samples,
  // several seconds on CPU). Without saying so, a frozen screen looks broken.
  const stalled = liveLast && liveLast.step===f.step && liveLast.env===f.env;
  liveLast=f;
  $('#liveMeta').textContent=
    `${watchVar} · env ${f.env} · step ${f.step} · chose '${f.action}'`
    + (stalled ? ' · paused (updating policy)' : '');
  if(liveTiles){ drawTilesInto($('#liveCanvas'),f); }
  else { $('#liveScreen').textContent=f.state||''; }
  renderBrainInto($('#liveBrain'),f);
}

async function setLiveView(t){
  liveTiles=t;
  $('#liveAscii').classList.toggle('on',!t);
  $('#liveTiles').classList.toggle('on',t);
  $('#liveScreen').style.display=t?'none':'';
  $('#liveCanvas').style.display=t?'':'none';
  if(t) await loadAtlas();
  if(liveLast) (t?drawTilesInto($('#liveCanvas'),liveLast)
                 :$('#liveScreen').textContent=liveLast.state||'');
}

function renderMachine(){
  const m=D.machine||{}, el=$('#machine');
  if(!m.cpu&&!m.gpu){el.innerHTML='<p class="muted">no machine stats</p>';return;}
  const out=[];
  if(m.gpu){
    const g=m.gpu, memPct=Math.round(100*g.mem_used/Math.max(1,g.mem_total));
    out.push(meter('GPU memory',memPct,
      `${g.name} · ${(g.mem_used/1024).toFixed(1)} / ${(g.mem_total/1024).toFixed(1)} GB`,90));
    // power.limit reads N/A on this laptop GPU, so only show a cap if we got one.
    const pw=g.power!=null?(g.power_cap!=null?`· ${g.power}W / ${g.power_cap}W`
                                             :`· ${g.power}W`):'';
    out.push(meter('GPU util',g.util,`${g.temp!=null?g.temp+'°C':''} ${pw}`,95));
  }
  if(m.cpu) out.push(meter('CPU',m.cpu.pct,
    `${m.cpu.cores} cores · load ${m.cpu.load1} / ${m.cpu.load5} / ${m.cpu.load15}`,90));
  if(m.ram) out.push(meter('RAM (WSL)',m.ram.pct,
    `${(m.ram.used/1024).toFixed(1)} / ${(m.ram.total/1024).toFixed(1)} GB`,90));
  if(m.disk) out.push(meter('Disk C:',m.disk.pct,
    `${m.disk.free_gb} GB free of ${m.disk.total_gb} GB`,90));
  out.push(`<div class="mtr"><div class="mrow"><span>games running</span>
     <b>${m.games||0}</b></div><div class="muted mdet">live crawl processes</div></div>`);
  el.innerHTML=`<div class="mgrid">${out.join('')}</div>`;
}

// HOW episodes end, not just how often they succeed. This was in the payload
// all along and never drawn, which is why "dying at D:3" vs "wedged on D:1"
// — two completely different problems needing opposite fixes — looked
// identical on this dashboard: both just a low solve rate.
function outcomeBar(s){
  const oc=s.outcomes||{}, n=Object.values(oc).reduce((a,b)=>a+b,0);
  if(!n) return '';
  const grp={solved:0,died:0,wedged:0};
  for(const [k,v] of Object.entries(oc)){
    if(k.startsWith('reached')) grp.solved+=v;
    else if(k==='died') grp.died+=v;
    else grp.wedged+=v;               // step limit + stalled: burning the clock
  }
  const seg=Object.entries(grp).filter(([,v])=>v)
    .map(([k,v])=>`<i class="${k}" style="flex:${v}" title="${k} ${v}"></i>`).join('');
  const pc=k=>Math.round(100*grp[k]/n);
  return `<div class="omix">${seg}</div><div class="muted">`+
    `${pc('solved')}% solved · ${pc('died')}% died · ${pc('wedged')}% out of time</div>`;
}

function renderVariants(){
  const V=D.variants||{}, keys=Object.keys(V);
  const el=$('#variants');
  if(!keys.length){el.innerHTML='<p class="muted" style="padding:14px">no variants</p>';return;}
  const best=Math.max(...keys.map(k=>V[k].summary?V[k].summary.solve_rate:0));
  el.innerHTML='<div class="vgrid">'+keys.map(k=>{
    const v=V[k], s=v.summary;
    if(!s) return `<div class="vcard"><div class="vh"><b>${k}</b>
      <span>${v.label}</span></div><p class="muted">not started</p></div>`;
    // Sample size gates the headline. Every per-episode number below shares
    // one 60-episode rolling window and is biased UP while it is filling, so
    // a variant is only eligible to be "lead" once its window is trustworthy —
    // otherwise 3 solves out of 4 crowns it at 75%.
    const n=s.n_episodes??0, weak=n<(s.n_trusted??30);
    const lead=!weak&&s.solve_rate>0&&s.solve_rate>=best;
    return `<div class="vcard${lead?' lead':''}">
      <div class="vh"><b>${k}</b><span>${v.label}</span>
        <i class="${v.live?'dot live':'dot'}"></i></div>
      <div class="vbig${weak?' weak':''}">${(100*s.solve_rate).toFixed(0)}<em>% D:5</em></div>
      <div class="${weak?'warn':'muted'}">${weak
        ?`⚠ only ${n} episodes — biased high, not yet meaningful`
        :`over ${n} episodes`}</div>
      <table class="vt">
        <tr><td>mean depth</td><td>${s.mean_depth.toFixed(2)}${
          weak?'<span class="warn"> ⚠</span>':''}</td></tr>
        <tr><td>return</td><td>${s.mean_return.toFixed(2)}${
          weak?'<span class="warn"> ⚠</span>':''}</td></tr>
        <tr><td>update</td><td>${s.update}</td></tr>
        <tr><td>entropy</td><td>${s.entropy.toFixed(2)} / ${s.max_entropy.toFixed(2)}</td></tr>
        <tr><td>AC</td><td>${s.mean_ac??'—'}<span class="muted">${
          s.ac_gained?` (+${s.ac_gained} earned)`:''}</span></td></tr>
        <tr><td>weapon</td><td>${s.mean_wpn??'—'}<span class="${
          s.mean_wpn!=null&&s.mean_wpn<7?'warn':'muted'}">${
          s.mean_wpn!=null&&s.mean_wpn<7?' ⚠ below starting axe'
          :(s.wpn_gained?` (+${s.wpn_gained} earned)`:'')}</span></td></tr>
        <tr><td>equips</td><td>${s.equips??'—'}<span class="muted"> AC-raising</span></td></tr>
        <tr><td>berserks</td><td>${s.berserks??'—'}<span class="muted">${
          s.berserk_wasted?` (${s.berserk_wasted} wasted)`:''}</span></td></tr>
      </table>${outcomeBar(s)}</div>`;}).join('')+'</div>';
}

function renderRL(){
  const R=D.rl||{}, s=R.summary;
  $('#rlLive').textContent = R.live ? 'training now' : (s?'idle':'');
  if(!s){
    $('#rlSummary').innerHTML='<p class="muted">no PPO run yet — start one with '+
      '<code>train_rl.py</code></p>';
    chart($('#rlDepth'),[],{empty:'no run yet'});
    chart($('#rlEnt'),[],{empty:'no run yet'});
    $('#rlMix').innerHTML='<p class="muted">no data</p>';
  } else {
    // Entropy is only meaningful against its own ceiling: ln(6)=1.792 IS a
    // uniformly random policy, so the raw number alone reads as progress when
    // there is none.
    const pct=(100*(1-s.entropy/s.max_entropy)).toFixed(0);
    const cards=[
      ['update',s.update,`${s.steps.toLocaleString()} steps · ${s.elapsed_min}min`],
      ['mean depth',s.mean_depth.toFixed(2),`last 20 updates: ${s.trend_depth}`],
      ['best depth','D:'+s.best_depth,'teacher’s best was D:4'],
      ['D:5 solve',(100*s.solve_rate).toFixed(0)+'%','the actual goal'],
      ['entropy',s.entropy.toFixed(3),`${pct}% below random (${s.max_entropy.toFixed(2)})`],
      ['return',s.mean_return.toFixed(2),'reward v3'],
    ];
    $('#rlSummary').innerHTML='<div class="kpis">'+cards.map(c=>
      `<div class="kpi"><div class="k">${c[0]}</div><div class="v">${c[1]}</div>`+
      `<div class="s">${c[2]}</div></div>`).join('')+'</div>';
    chart($('#rlDepth'),R.rows.map(r=>[r.u,r.depth]),{x:'PPO update',dp:2,base:1});
    chart($('#rlEnt'),R.rows.map(r=>[r.u,r.ent]),
      {x:'PPO update',dp:2,base:s.max_entropy});
    const max=Math.max(...s.mix.map(m=>m.pct),1);
    $('#rlMix').innerHTML=s.mix.map(m=>`
      <div class="bar-row wide"><span class="lab">${m.name}</span>
        <span class="track"><span class="fill" style="width:${(100*m.pct/max).toFixed(1)}%"></span></span>
        <span class="val">${m.pct}%</span></div>`).join('');
  }
  const past=R.past||[];
  $('#rlPast').innerHTML = past.length
    ? '<table><thead><tr><th>run</th><th class="num">updates</th>'+
      '<th class="num">best</th><th class="num">peak mean depth</th>'+
      '<th class="num">final entropy</th><th>why discarded</th></tr></thead><tbody>'+
      past.map(p=>`<tr><td><code>${p.name}</code></td><td class="num">${p.updates}</td>`+
        `<td class="num">D:${p.best_depth}</td><td class="num">${p.peak_mean_depth}</td>`+
        `<td class="num">${(p.final_entropy??0).toFixed(3)}</td>`+
        `<td>${p.verdict}</td></tr>`).join('')+'</tbody></table>'
    : '<p class="muted" style="padding:14px">none</p>';
}

function renderCharts(){
  const g=games();
  chart($('#c1'),g.map((x,i)=>[i+1,x.turns||0]),{x:'game (current sort)',empty:'no games yet'});
  chart($('#c2'),D.experiments.map((e,i)=>[i+1,e.loss]),
    {x:'experiment',base:D.final?.base_loss,dp:2,empty:'no experiments yet'});
}

function renderExps(){
  $('#ebody').innerHTML=D.experiments.length?D.experiments.map(e=>`
    <tr><td class="mono">${esc(e.commit)}</td><td class="num strong">${e.loss.toFixed(4)}</td>
    <td class="num">${esc(e.top1)}</td><td>${esc(e.status)}</td>
    <td class="strong">${esc(e.desc)}</td></tr>`).join('')
    :`<tr><td colspan="5" class="muted" style="padding:22px;text-align:center">no experiments yet</td></tr>`;
}

async function loadLog(){
  $('#log').textContent='loading…';
  $('#log').textContent = await (await fetch('/api/log?which='+encodeURIComponent(logName))).text();
}

/* ---- spectate ---- */
let frames=[],fi=0,timer=null;
async function openReplay(game){
  if(D.live && D.served){ window.open(D.webtiles,'_blank'); return; }
  $('#rtitle').textContent='loading '+game+'…';
  $('#modal').classList.add('on'); $('#screen').textContent='';
  frames=await (await fetch('/api/replay?game='+encodeURIComponent(game))).json();
  if(!frames.length){
    // Replay files rotate out (newest 40 kept) but the games.jsonl row stays.
    // Say so — previously this left a blank black canvas that read as a bug.
    $('#rtitle').textContent=game+' — replay no longer stored';
    $('#screen').textContent='This game’s replay file has been rotated out.\n'+
      'Only the 40 most recent RL episodes keep their frames.';
    $('#screen').style.display=''; $('#tiles').style.display='none';
    $('#brain').style.display='none'; $('#tileNote').textContent='';
    return;}
  $('#rtitle').textContent=game+' — '+frames.length+' frames';
  $('#seek').max=frames.length-1; showFrame(0); play();
}
const RL_ACT=['autofight','explore','rest','descend','travel','escape'];

/* ---- tile renderer -------------------------------------------------------
   DCSS console draws the map in the left ~33 columns; the rest of the 80x24
   grid is the status panel and message lines, which have no tiles and stay as
   text. We look up (glyph,colour) then fall back to glyph alone, and draw the
   character itself when neither resolves — an honest gap beats a wrong sprite.
*/
let ATLAS=null, SHEETS={}, tileMode=false;
// Map occupies cols 0-36; the status panel starts at col 37 ("Minotaur of
// Trog"). Rows 0-16 are the map, 17+ the message area.
const MAP_COLS=37, MAP_ROWS=17, TS=32;

// A row of PROSE, not terrain. Two cases: the message area, and the header of
// the level-map overlay the agent opens with `X` ("An escape hatch in the
// floor (D:1)"). Tiling those turns the `f` of "floor" into a fungus sprite
// and the `r` into a monster — which is exactly what it did.
// Map rows almost never contain three consecutive letters; monsters are
// isolated glyphs surrounded by walls and floor.
const RE_WORD=/[A-Za-z]{3,}/;
function isTextRow(line){ return RE_WORD.test(line.slice(0,MAP_COLS)); }

async function loadAtlas(){
  if(ATLAS) return ATLAS;
  try{
    ATLAS=await (await fetch('/tiles/atlas.json')).json();
    await Promise.all(ATLAS.sheets.map(s=>new Promise(res=>{
      const im=new Image(); im.onload=()=>{SHEETS[s]=im;res();};
      im.onerror=()=>res(); im.src='/tiles/'+s+'.png';})));
  }catch(e){ ATLAS=null; }
  return ATLAS;
}

function lookup(ch,col){
  if(!ATLAS) return null;
  return ATLAS.map[ch+'\t'+col] || ATLAS.map[ch] || null;
}

function drawTiles(f){ drawTilesInto($('#tiles'),f); }

function drawTilesInto(cv,f){
  if(!cv||!ATLAS) return;
  // Replay frames call it `screen` (renamed by the API); live frames carry the
  // raw `state` field straight from the trainer. Accept both.
  const rows=(f.screen||f.state||'').split('\n');
  const crows=(f.colors||'').split('\n');
  cv.width=MAP_COLS*TS; cv.height=MAP_ROWS*TS;
  const g=cv.getContext('2d'); g.imageSmoothingEnabled=false;
  g.fillStyle='#000'; g.fillRect(0,0,cv.width,cv.height);
  const floor=ATLAS.map['.'];
  let miss=0;
  for(let y=0;y<MAP_ROWS;y++){
    const line=rows[y]||'', cline=crows[y]||'';
    if(isTextRow(line)){
      // Draw prose as prose. Half-height so it reads as an overlay caption
      // rather than pretending to be part of the dungeon.
      g.fillStyle='#0b0c0a'; g.fillRect(0,y*TS,cv.width,TS);
      g.fillStyle='#c8c9c1'; g.font='15px ui-monospace,Consolas,monospace';
      g.textAlign='left'; g.textBaseline='middle';
      g.fillText(line.slice(0,MAP_COLS).replace(/\s+$/,''),4,y*TS+TS/2);
      continue;
    }
    for(let x=0;x<MAP_COLS;x++){
      const ch=line[x]||' '; if(ch===' ') continue;
      const col=cline[x]||'w';
      // Floor underneath so monsters and items don't float on black.
      if(floor&&SHEETS[floor.sheet]&&ch!=='#')
        g.drawImage(SHEETS[floor.sheet],floor.x,floor.y,floor.w,floor.h,x*TS,y*TS,TS,TS);
      const t=lookup(ch,col);
      if(t&&SHEETS[t.sheet]){
        g.drawImage(SHEETS[t.sheet],t.x,t.y,t.w,t.h,x*TS,y*TS,TS,TS);
      }else{
        miss++;
        g.fillStyle='#cfcfc7'; g.font='20px ui-monospace,Consolas,monospace';
        g.textAlign='center'; g.textBaseline='middle';
        g.fillText(ch,x*TS+TS/2,y*TS+TS/2);
      }
    }
  }
  const note=cv.id==='tiles'?$('#tileNote'):null;
  if(note) note.textContent = miss? `${miss} cells drawn as text (no sprite mapped)` : '';
}

async function setView(tiles){
  tileMode=tiles;
  $('#viewAscii').classList.toggle('on',!tiles);
  $('#viewTiles').classList.toggle('on',tiles);
  $('#screen').style.display=tiles?'none':'';
  $('#tiles').style.display=tiles?'':'none';
  if(tiles){
    await loadAtlas();
    if(!ATLAS){ $('#tileNote').textContent='tiles unavailable — run build_tiles.py'; }
    else showFrame(fi);
  }else{ $('#tileNote').textContent=''; }
}

function showFrame(i){
  // Rows survive in games.jsonl after their replay file is rotated out, so a
  // game can legitimately have zero frames. Bail rather than throw.
  if(!frames.length){$('#screen').textContent='';return;}
  fi=Math.max(0,Math.min(i,frames.length-1));const f=frames[fi];
  $('#screen').textContent=f.screen;$('#seek').value=fi;
  $('#fmeta').textContent=`frame ${fi+1}/${frames.length} · t=${f.t}s · key '${f.action}'`;
  if(tileMode) drawTiles(f);
  renderBrain(f);}

// What the network computed for THIS frame. Only RL episodes carry it; games
// collected by the rule-based teacher or the random policy have no network,
// so the panel hides itself rather than showing an empty chart.
function renderBrain(f){ renderBrainInto($('#brain'),f); }

function renderBrainInto(el,f){
  if(!el) return;
  if(!f.probs){el.style.display='none';return;}
  el.style.display='';
  // Names come with the frame: variants have 7, 10 and 14 actions, and a
  // hardcoded six-name list labelled everything past `escape` as "undefined".
  const names=f.names&&f.names.length?f.names:RL_ACT;
  const top=Math.max(...f.probs);
  const bars=f.probs.map((p,i)=>{
    const chosen=names[i]===f.action;   // frames store the action NAME
    // Scaled against the LARGEST probability, not against 1.0. A 10-way
    // near-uniform policy tops out around 25%, so absolute widths render as
    // slivers and every option looks equally dead.
    const w=100*p/(top||1);
    return `<div class="bar-row wide${chosen?' picked':''}">
      <span class="lab">${names[i]||('action '+i)}</span>
      <span class="track"><span class="fill${chosen?' hot':''}"
        style="width:${w.toFixed(1)}%"></span></span>
      <span class="val">${(100*p).toFixed(0)}%</span></div>`;}).join('');
  // 120 pooled positions, each covering 16 screen chars = 1/5 of a row.
  const cells=f.sal.map(s=>`<i style="opacity:${Math.max(0.04,s).toFixed(2)}"></i>`).join('');
  el.innerHTML=`<div class="brain-grid">
    <div><div class="blabel">action probabilities</div>${bars}
      <div class="blabel" style="margin-top:8px">value estimate
        <b>${f.value>=0?'+':''}${f.value}</b>
        <span class="muted">expected remaining reward</span></div></div>
    <div><div class="blabel">encoder activation by screen region</div>
      <div class="sal">${cells}</div>
      <div class="muted" style="font-size:10.5px;margin-top:6px">
        120 pooled positions, top-left to bottom-right. Magnitude of signal
        carried, not a causal claim about what it "looked at".</div></div>
  </div>`;}
function play(){stop();timer=setInterval(()=>{if(fi>=frames.length-1){stop();return;}
  showFrame(fi+1);},+$('#speed').value);$('#playBtn').textContent='pause';}
function stop(){if(timer){clearInterval(timer);timer=null;}$('#playBtn').textContent='play';}
function closeReplay(){stop();$('#modal').classList.remove('on');}

function renderAll(){renderBar();renderKpis();renderRuns();renderHealth();
  renderMachine();renderVariants();renderRL();renderGames();renderExps();
  renderCharts();}

async function refresh(){
  if(!D.served)return;
  try{D=await (await fetch('/api/data')).json();renderAll();}catch(e){}
}

async function post(url){return await (await fetch(url,{method:'POST'})).text();}

document.addEventListener('DOMContentLoaded',()=>{
  $('#search').oninput=e=>{filter=e.target.value;renderGames();renderCharts();};
  $('#selall').onclick=()=>{const g=games();const all=g.every(x=>sel.has(x._id));
    g.forEach(x=>all?sel.delete(x._id):sel.add(x._id));renderGames();};
  $('#pruneBtn').onclick=async()=>{
    if(!confirm(`Archive ${sel.size} game(s) to games.pruned.jsonl?\n\nReversible — nothing is deleted.`))return;
    toast(await (await fetch('/api/prune',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ids:[...sel]})})).text());
    sel.clear();refresh();};
  // Immediate feedback matters here: the launch takes a couple of seconds
  // server-side, and without this the button looks dead.
  $('#runGames').onclick=async()=>{
    const b=$('#runGames'); const was=b.textContent;
    b.disabled=true; b.textContent='starting…';
    try{
      toast(await post('/api/start?what=games&n='+$('#nGames').value+
                       '&extra='+$('#nActions').value), 6000);
    } finally { b.disabled=false; b.textContent=was; }
    refresh();
  };
  $('#runWatch').onclick=async()=>{toast(await post('/api/start?what=watch&n=400'));
    setTimeout(()=>window.open(D.webtiles,'_blank'),2500);};
  $('#runTrain').onclick=async()=>{toast(await post('/api/start?what=training'));refresh();};
  $('#mergeBtn').onclick=async()=>{toast(await post('/api/start?what=merge'));refresh();};
  $('#watchBtn').onclick=()=>window.open(D.webtiles,'_blank');
  $('#refreshBtn').onclick=refresh;
  $('#logSel').onchange=e=>{logName=e.target.value;loadLog();};
  $('#logRefresh').onclick=loadLog;
  $('#closeReplay').onclick=closeReplay;
  $('#viewAscii').onclick=()=>setView(false);
  $('#viewTiles').onclick=()=>setView(true);
  $('#liveAscii').onclick=()=>setLiveView(false);
  $('#liveTiles').onclick=()=>setLiveView(true);
  renderVpick();
  pollLive(); liveTimer=setInterval(pollLive,700);
  pollEnvs(); setInterval(pollEnvs,2000);
  $('#playBtn').onclick=()=>timer?stop():play();
  $('#seek').oninput=e=>{stop();showFrame(+e.target.value);};
  $('#speed').onchange=()=>{if(timer)play();};
  document.addEventListener('keydown',e=>{
    if(!$('#modal').classList.contains('on'))return;
    if(e.key==='Escape')closeReplay();
    if(e.key==='ArrowRight'){stop();showFrame(fi+1);}
    if(e.key==='ArrowLeft'){stop();showFrame(fi-1);}
    if(e.key===' '){e.preventDefault();timer?stop():play();}});
  renderAll();
  if(D.served){loadLog();setInterval(refresh,8000);}
});
"""


def page(data):
    ctl = "" if data["served"] else "disabled"
    note = ("" if data["served"] else
            '<span class="muted">static file — start with '
            '<code>python project.py --serve</code> for controls</span>')
    return f"""<style>{CSS}</style>
<div class="app">
  <div class="bar"><div class="bar-in">
    <div class="brand">DCSS agent<span>rule-based teacher &rarr; imitation model</span></div>
    <div class="pills" id="pills"></div>
    <div class="spacer"></div>
    <span class="clock" id="clock"></span>
    <button id="watchBtn" class="primary" style="display:none">watch live</button>
    <button id="refreshBtn">refresh</button>
  </div></div>

  <main>
    <section><div class="kpis" id="kpis"></div></section>

    <section>
      <div class="head"><h2>Live game</h2>
        <span class="sub">the agent playing, right now</span>
        <div class="right">
          <span class="vpick" id="vpick"></span>
          <button id="liveAscii" class="on">ascii</button>
          <button id="liveTiles">tiles</button>
          <span class="muted" id="liveMeta"></span></div></div>
      <div class="panel pad">
        <div id="envGrid" class="envgrid"></div>
        <div class="livewrap" style="margin-top:12px">
          <div><pre id="liveScreen">waiting for the trainer…</pre>
            <canvas id="liveCanvas" style="display:none"></canvas></div>
          <div id="liveBrain"></div>
        </div>
      </div>
    </section>

    <section>
      <div class="head"><h2>Machine</h2>
        <span class="sub">GPU · CPU · memory · disk</span></div>
      <div class="panel pad" id="machine"></div>
    </section>

    <section>
      <div class="head"><h2>Variant comparison</h2>
        <span class="sub">three equipment designs, same machine, same reward</span></div>
      <div class="panel" id="variants"></div>
    </section>

    <section>
      <div class="head"><h2>Reinforcement learning</h2>
        <span class="sub">PPO · policy learns from its own games · no teacher data</span>
        <div class="right"><span class="muted" id="rlLive"></span></div></div>
      <div class="panel pad" id="rlSummary"></div>
      <div class="cols" style="margin-top:12px">
        <div><div class="head"><h3>Mean depth reached</h3>
          <span class="sub">the objective — higher is better</span></div>
          <div class="panel pad" id="rlDepth"></div></div>
        <div><div class="head"><h3>Policy entropy</h3>
          <span class="sub">falling = committing to a strategy</span></div>
          <div class="panel pad" id="rlEnt"></div></div>
      </div>
      <div class="head" style="margin-top:12px"><h3>Action mix</h3>
        <span class="sub">both failed runs were diagnosed here, not from the return curve</span></div>
      <div class="panel pad" id="rlMix"></div>
      <div class="head" style="margin-top:12px"><h3>Discarded runs</h3>
        <span class="sub">kept visible on purpose</span></div>
      <div class="panel" id="rlPast"></div>
    </section>

    <section>
      <div class="head"><h2>Run</h2><span class="sub">{note}</span></div>
      <div class="panel pad" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <button id="runGames" class="primary" {ctl}>collect games</button>
        <input type="number" id="nGames" value="10" min="1" max="500" {ctl}>
        <span class="muted">games ×</span>
        <input type="number" id="nActions" value="220" min="20" max="5000" {ctl}>
        <span class="muted">actions</span>
        <span style="width:14px"></span>
        <button id="runWatch" {ctl}>play a watchable game</button>
        <button id="runTrain" {ctl}>run training</button>
        <button id="mergeBtn" {ctl}>merge shards</button>
      </div>
    </section>

    <div class="cols">
      <section>
        <div class="head"><h2>Active</h2></div>
        <div class="panel" id="runs"></div>
      </section>
      <section>
        <div class="head"><h2>Data health</h2>
          <span class="sub">action distribution</span></div>
        <div class="panel pad" id="health"></div>
      </section>
    </div>

    <section>
      <div class="head"><h2>Games</h2>
        <div class="right"><span class="muted" id="selc"></span>
          <span class="muted" id="gcount"></span></div></div>
      <div class="panel">
        <div class="toolbar">
          <input type="search" id="search" placeholder="filter agent, outcome, date…" class="grow">
          <button id="selall">select all shown</button>
          <button id="pruneBtn" class="danger" {ctl}>archive selected</button>
        </div>
        <div class="tw"><table><thead id="ghead"></thead><tbody id="gbody"></tbody></table></div>
      </div>
    </section>

    <div class="cols">
      <section><div class="head"><h2>Turns per game</h2></div>
        <div class="panel pad" id="c1"></div></section>
      <section><div class="head"><h2>val_action_loss</h2>
        <span class="sub">lower is better</span></div>
        <div class="panel pad" id="c2"></div></section>
    </div>

    <section>
      <div class="head"><h2>Logs</h2><div class="right">
        <select id="logSel">
          <option value="collectors">collectors</option>
          <option value="player">watchable player</option>
          <option value="webtiles">webtiles server</option>
          <option value="training">training</option>
        </select>
        <button id="logRefresh">reload</button></div></div>
      <div class="panel"><pre class="log" id="log">—</pre></div>
    </section>

    <section>
      <div class="head"><h2>Experiments</h2></div>
      <div class="panel"><table><thead><tr><th>commit</th><th class="num">loss</th>
        <th class="num">top1</th><th>status</th><th>description</th></tr></thead>
        <tbody id="ebody"></tbody></table></div>
    </section>

    <section>
      <div class="head"><h2>Findings</h2></div>
      <div class="panel pad">{findings_html(data["findings"])}</div>
    </section>
  </main>
</div>

<div class="modal" id="modal"><div class="sheet">
  <div class="sheet-head"><span id="rtitle"></span>
    <button id="closeReplay">close</button></div>
  <div class="viewsel"><button id="viewAscii" class="on">ascii</button>
    <button id="viewTiles">tiles</button>
    <span class="muted" id="tileNote"></span></div>
  <pre id="screen"></pre>
  <canvas id="tiles" style="display:none"></canvas>
  <div id="brain" style="display:none"></div>
  <div class="pctl"><button id="playBtn">play</button>
    <input type="range" id="seek" min="0" value="0" step="1">
    <select id="speed"><option value="400">0.5×</option>
      <option value="200" selected>1×</option><option value="80">2.5×</option>
      <option value="25">8×</option></select></div>
  <div class="muted" id="fmeta"></div>
  <div class="muted">space play/pause · ← → step · esc close</div>
</div></div>
<div class="toast" id="toast"></div>
<script>window.__DATA__ = {json.dumps(data)};</script>
<script>{JS}</script>"""


# ───────────────────────── server ─────────────────────────

_cache = {"t": 0.0, "d": None}
_lock = threading.Lock()


def cached(ttl=8.0):
    """TTL must exceed how long payload() takes, or the cache can never hit.
    At ttl=3 with a 6.6s payload, every request rebuilt everything from
    scratch — including the browser's own 10s auto-refresh."""
    with _lock:
        now = time.time()
        if _cache["d"] is None or now - _cache["t"] > ttl:
            _cache["d"] = payload(True)
            _cache["t"] = now
        return _cache["d"]


class Handler(BaseHTTPRequestHandler):
    # Browsers open speculative connections and leave them idle; on a
    # single-threaded server that blocks the accept loop and the page hangs.
    protocol_version = "HTTP/1.1"
    timeout = 15

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (socket.timeout, TimeoutError, ConnectionResetError):
            self.close_connection = True

    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self._send(200, page(cached()), "text/html; charset=utf-8")
        elif u.path == "/api/data":
            self._send(200, json.dumps(cached()), "application/json")
        elif u.path == "/api/replay":
            self._send(200, json.dumps(replay_frames(q.get("game", [""])[0])),
                       "application/json")
        elif u.path == "/api/log":
            self._send(200, tail_log(q.get("which", ["collectors"])[0]))
        elif u.path == "/api/envs":
            v = _vsafe(q.get("v", ["a"])[0])
            try:
                self._send(200, (DATA / f"rl_envs.{v}.json").read_text("utf-8"),
                           "application/json")
            except OSError:
                self._send(200, "[]", "application/json")
        elif u.path == "/api/watch":
            # Tell that variant's trainer which game to publish full frames for.
            v = _vsafe(q.get("v", ["a"])[0])
            try:
                n = int(q.get("env", ["0"])[0])
                (DATA / f"rl_view.{v}.txt").write_text(str(max(0, n)), "utf-8")
                self._send(200, f"watching {v}/env {n}")
            except (ValueError, OSError) as e:
                self._send(400, f"bad env: {e}")
        elif u.path == "/api/live":
            # Not cached: the whole point is that it is current.
            v = _vsafe(q.get("v", ["a"])[0])
            try:
                self._send(200, (DATA / f"rl_live.{v}.json").read_text("utf-8"),
                           "application/json")
            except OSError:
                self._send(200, "null", "application/json")
        elif u.path.startswith("/tiles/"):
            # DCSS's own sprite sheets, served for the tile renderer. The name
            # is whitelisted rather than joined onto a path: this handler would
            # otherwise happily serve ../../anything on disk.
            name = u.path[len("/tiles/"):]
            allowed = {f"{s}.png" for s in ("floor", "wall", "feat", "main",
                                            "player")} | {"atlas.json"}
            f = RL_TILES / name
            if name not in allowed or not f.exists():
                self._send(404, "not found")
                return
            ctype = ("application/json" if name.endswith(".json")
                     else "image/png")
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Sheets are ~1-2MB each and never change between builds.
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send(404, "not found")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/prune":
            n = int(self.headers.get("Content-Length", 0))
            ids = json.loads(self.rfile.read(n) or b"{}").get("ids", [])
            self._send(200, f"archived {prune_games(ids)} game(s)")
        elif u.path == "/api/stop":
            self._send(200, stop_run(q.get("kind", [""])[0], q.get("pid", [""])[0]))
        elif u.path == "/api/start":
            self._send(200, start_run(q.get("what", [""])[0],
                                      q.get("n", ["1"])[0], q.get("extra", ["0"])[0]))
        else:
            self._send(404, "not found")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    if a.serve:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
        print(f"control panel: http://localhost:{a.port}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    else:
        OUT.write_text(page(payload(False)), encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

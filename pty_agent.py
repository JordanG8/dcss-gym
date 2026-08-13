"""
Talk to DCSS as what it actually is: a terminal program.

Spawns crawl on a pseudo-terminal, renders its output into an 80x24 character
grid with pyte, sends keystrokes, and logs (screen, key) pairs - which is
exactly the schema prepare_dcss.py already wants:

    {"state": "<80x24 screen as text>", "action": "h"}

No web server, no websocket, no JSON protocol, no login, no menu state
machine. Terminal rendering is also far more stable across DCSS versions than
the webtiles JSON protocol, which is what broke the previous approach.

Usage:
    python pty_agent.py --games 3 --max-actions 400
    python pty_agent.py --games 1 --show          # print the screen live
"""
import argparse
import errno
import json
import os
import pty
import random
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import fcntl
from datetime import datetime
from pathlib import Path

import pyte

HERE = Path(__file__).parent
CRAWL_DIR = Path("/root/crawl/crawl-ref/source")
PLAY_DIR = Path("/root/botplay")          # saves/morgue/rc, kept away from webtiles
TRACES = HERE / "data" / "traces.jsonl"
GAMES = HERE / "games.jsonl"
RECORDINGS = HERE / "recordings"

COLS, ROWS = 80, 24

# Movement + a few useful commands. These are what we log as "actions".
MOVE_KEYS = list("hjklyubn")
EXTRA_KEYS = ["o"]           # auto-explore: makes random play far less useless
ACTION_KEYS = MOVE_KEYS * 4 + EXTRA_KEYS

# Prompts we answer mechanically. These are NOT logged as decisions - they are
# forced by the UI, so training on them would teach noise.
MORE_RE = re.compile(r"--more--|--More--", re.I)
DEATH_RE = re.compile(r"You die\.\.\.|You have died|Goodbye,", re.I)
YESNO_RE = re.compile(r"\[y\]es\b.*\[n\]o|\(y/n\)", re.I)


class Crawl:
    """A running crawl process on a pty, rendered to a character grid."""

    def __init__(self, name="bot", species="Minotaur", background="Berserker",
                 seed=None, ttyrec=None):
        PLAY_DIR.mkdir(parents=True, exist_ok=True)
        (PLAY_DIR / "morgue").mkdir(exist_ok=True)
        (PLAY_DIR / "rcs").mkdir(exist_ok=True)

        # Standard ttyrec, the same format DCSS servers record. Lets ttyplay
        # and the rest of that ecosystem replay our games with no extra work.
        self.ttyrec = open(ttyrec, "wb") if ttyrec else None

        self.master, slave = pty.openpty()
        # crawl asks the tty for its size; without this it assumes something
        # else and the grid we render won't match what it drew.
        fcntl.ioctl(slave, termios.TIOCSWINSZ,
                    struct.pack("HHHH", ROWS, COLS, 0, 0))

        cmd = [
            "./crawl",
            "-name", name,
            "-species", species,
            "-background", background,
            "-dir", str(PLAY_DIR) + "/",
            "-morgue", str(PLAY_DIR / "morgue"),
            "-rcdir", str(PLAY_DIR / "rcs"),
            "-extra-opt-first", "weapon=hand axe",
            "-extra-opt-first", "show_more=false",
            "-extra-opt-first", "clear_messages=true",
            # Autofight refuses below this HP fraction ("You are too injured to
            # fight recklessly!") and silently consumes the keypress. With a
            # policy that fights whenever a monster is visible, that is an
            # infinite Tab loop. 0 = always swing.
            "-extra-opt-first", "autofight_stop=0",
        ]
        if seed is not None:
            cmd += ["-seed", str(seed)]

        env = dict(os.environ, TERM="xterm-256color", LINES=str(ROWS),
                   COLUMNS=str(COLS))
        self.proc = subprocess.Popen(
            cmd, cwd=str(CRAWL_DIR), stdin=slave, stdout=slave, stderr=slave,
            env=env, start_new_session=True,
        )
        os.close(slave)

        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.ByteStream(self.screen)

    def read_until_quiet(self, quiet=0.15, timeout=6.0):
        """Drain output until the game stops drawing for `quiet` seconds."""
        end = time.time() + timeout
        last = time.time()
        got_any = False
        while time.time() < end:
            r, _, _ = select.select([self.master], [], [], 0.05)
            if r:
                try:
                    data = os.read(self.master, 65536)
                except OSError as e:
                    if e.errno == errno.EIO:
                        return got_any     # child exited
                    raise
                if not data:
                    return got_any
                self.stream.feed(data)
                if self.ttyrec:
                    # ttyrec frame: uint32 sec, uint32 usec, uint32 length
                    now = time.time()
                    self.ttyrec.write(struct.pack("<III", int(now),
                                                  int((now % 1) * 1e6), len(data)))
                    self.ttyrec.write(data)
                got_any = True
                last = time.time()
            elif time.time() - last >= quiet:
                return got_any
        return got_any

    def text(self):
        return "\n".join(self.screen.display)

    def send(self, keys):
        os.write(self.master, keys.encode())

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGHUP)  # crawl saves
        except Exception:
            pass
        try:
            self.proc.wait(timeout=8)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            os.close(self.master)
        except Exception:
            pass
        if self.ttyrec:
            self.ttyrec.close()


class Teacher:
    """Rule-based policy driven by the literal terminal screen.

    This lives on the pty transport, not webtiles, for one reason: here we see
    EVERYTHING crawl draws — menus, --more--, level-up prompts, "You can't go
    down here". The webtiles version of this policy kept wedging because menus
    arrive there as separate ui-stack messages it never saw.

    It leans on DCSS's own commands rather than reimplementing tactics:
      Tab autofight, o auto-explore, 5 rest, > descend.
    """

    RE_MORE = re.compile(r"--more--", re.I)
    RE_ATTR = re.compile(r"Increase \(S\)trength", re.I)
    RE_YESNO = re.compile(r"\[y\]es.*\[n\]o|\(y/n\)", re.I)
    RE_DONE = re.compile(r"Done exploring|Partly explored|"
                         r"nothing left to explore", re.I)
    RE_NOSTAIRS = re.compile(r"can't go down|no down staircase|"
                             r"Cannot travel|No target found", re.I)
    RE_NOTARGET = re.compile(r"No target in view|No monsters", re.I)
    RE_MONSTER = re.compile(r"comes into view|is nearby|You encounter", re.I)
    RE_HP = re.compile(r"Health:\s*(\d+)/(\d+)")

    RE_INJURED = re.compile(r"too injured to fight", re.I)

    FALLBACK = ["\x1b", "5", "o", "\r", ">"]

    def __init__(self):
        self.explored = False
        self.combat = False
        self.level_turn = 0
        self.last_turn = -1
        self.no_progress = 0
        self.pending = []          # multi-key sequences (travel to stairs)
        self.traveled = False      # travel already issued on this level
        self.was_fallback = False  # last key was the unwedge cycle, not a choice

    def act(self, screen, turn):
        # No key may be repeated into a wall. If the turn counter hasn't moved,
        # the game rejected the last keypress (autofight below the HP
        # threshold does exactly this, silently) and pressing it again will be
        # rejected identically. Rotate through fallbacks instead.
        if turn == self.last_turn:
            self.no_progress += 1
        else:
            self.no_progress = 0
            self.last_turn = turn
        if self.no_progress >= 4:
            # Too injured to autofight and something is adjacent: rest is
            # pointless, so back off and let explore/stairs move us away.
            k = self.FALLBACK[(self.no_progress // 4) % len(self.FALLBACK)]
            if self.RE_INJURED.search(screen):
                self.combat = False
            # Flag it: these are the policy flailing to unwedge itself, NOT
            # decisions. Training on them teaches the fallback cycle, which is
            # near-uniform noise and swamps the real signal.
            self.was_fallback = True
            return k
        self.was_fallback = False

        # Blocking prompts first: nothing else works until they're answered.
        if self.RE_ATTR.search(screen):
            return "S"
        if self.RE_MORE.search(screen):
            return "\r"
        if self.RE_YESNO.search(screen):
            return "n"

        if self.RE_DONE.search(screen):
            self.explored = True
        if self.RE_NOSTAIRS.search(screen):
            # travel failed or we're not on stairs: re-issue travel next time
            self.traveled = False
        if self.RE_MONSTER.search(screen):
            self.combat = True
        if self.RE_NOTARGET.search(screen):
            self.combat = False

        m = self.RE_HP.search(screen)
        hp_frac = (int(m.group(1)) / max(1, int(m.group(2)))) if m else 1.0

        if self.combat:
            return "\t"
        if hp_frac < 0.6:
            return "5"
        if turn - self.level_turn > 900:
            self.explored = True
        if not self.explored:
            return "o"

        # Getting to the stairs. Plain `>` does NOT auto-travel in this build —
        # it answers "You can't go down here!" unless you are already standing
        # on a staircase. The level map is the route: X opens it, > jumps the
        # cursor to the next down staircase, Enter travels there. Then `>`
        # descends. Unlike the webtiles agent, the pty transport renders that
        # map so we are not blind inside it.
        if self.pending:
            return self.pending.pop(0)
        if not self.traveled:
            self.traveled = True
            self.pending = [">", "\r"]
            return "X"
        return ">"

    def on_new_level(self, turn):
        self.explored = False
        self.combat = False
        self.level_turn = turn
        self.traveled = False
        self.pending = []


def parse_status(screen_text):
    """Pull turn / HP / place off the status lines. Best effort."""
    out = {}
    m = re.search(r"Health:\s*(\d+)/(\d+)", screen_text) or \
        re.search(r"HP:\s*(\d+)/(\d+)", screen_text)
    if m:
        out["hp"], out["hp_max"] = int(m.group(1)), int(m.group(2))
    # DCSS's status line says "Time: 6.0 (1.0)", not "Turn:".
    m = re.search(r"Time:\s*([\d.]+)", screen_text)
    if m:
        out["turns"] = int(float(m.group(1)))
    m = re.search(r"XL:\s*(\d+)", screen_text)
    if m:
        out["xl"] = int(m.group(1))
    m = re.search(r"(Dungeon|Temple|Orc|Lair|D):(\d+)", screen_text)
    if m:
        out["depth"] = int(m.group(2))
    return out


def wipe_save(name):
    """Delete this worker's save so the next game starts a NEW character.

    Without this a worker resumes its previous character, which is fine while
    games end cleanly but silently corrupts the dataset when one wedges: the
    stuck save is reloaded every time and each subsequent "game" reports
    byte-identical turns/XL/depth/actions. Measured on games.jsonl before the
    fix: 18 duplicate groups, 86 redundant rows, 26% of the table.

    Only this worker's own files are touched, so parallel workers sharing
    PLAY_DIR are unaffected.
    """
    saves = PLAY_DIR / "saves"
    if not saves.is_dir():
        return
    for p in saves.glob(f"{name}.*"):
        try:
            p.unlink()
        except OSError:
            pass


def play_one(args, game_no):
    # Unique per worker as well as per second, so parallel workers can't
    # collide on an id.
    game_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.name}-{game_no}"
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    ttyrec_path = None if args.no_ttyrec else RECORDINGS / f"{game_id}.ttyrec"

    if not args.keep_save:
        wipe_save(args.name)

    c = Crawl(name=args.name, species=args.species, background=args.background,
              seed=args.seed, ttyrec=ttyrec_path)
    started = time.time()
    c.read_until_quiet(quiet=0.4, timeout=20)

    traces = []
    teacher = Teacher() if args.policy == "teacher" else None
    last_depth = 1
    last_screen = None
    unchanged = 0
    status = {}
    death_line = ""
    actions = 0

    # Global iteration guard. The prompt branches below `continue` without
    # incrementing `actions`, so `actions < max_actions` alone cannot end the
    # loop: if crawl sits at a prompt our keypress doesn't clear, the worker
    # spins forever, alive and silent. Observed in the wild - two workers ran
    # 25+ minutes without completing a game.
    max_iters = args.max_actions * 6
    iters = 0

    try:
        while actions < args.max_actions and c.alive() and iters < max_iters:
            iters += 1
            scr = c.text()

            if args.show:
                os.system("clear")
                print(scr)
                print(f"[game {game_no} action {actions}]")

            if DEATH_RE.search(scr):
                m = re.search(r"You die\.\.\.|Goodbye,.*", scr)
                death_line = (m.group(0) if m else "died").strip()
                # clear the death screens so the process can exit
                for _ in range(6):
                    c.send("\r")
                    c.read_until_quiet(quiet=0.1, timeout=2)
                break

            if teacher is None:
                # Forced UI prompts: answer them, don't log them as decisions.
                if MORE_RE.search(scr):
                    c.send(" ")
                    c.read_until_quiet()
                    continue
                if YESNO_RE.search(scr):
                    c.send("n")
                    c.read_until_quiet()
                    continue
                key = random.choice(ACTION_KEYS)
            else:
                d = status.get("depth", 1)
                if d != last_depth:
                    teacher.on_new_level(status.get("turns", 0))
                    last_depth = d
                    if d >= args.target_depth:
                        death_line = f"reached D:{d}"
                        break
                key = teacher.act(scr, status.get("turns", 0))
            # Record WHICH policy chose this key. Without it, teacher and
            # random traces are indistinguishable once merged, and the whole
            # dataset becomes untrainable — random labels aren't a function of
            # the screen, so mixing them in poisons the signal.
            traces.append({"state": scr, "action": key, "game": game_id,
                           "policy": args.policy,
                           "fallback": bool(teacher and teacher.was_fallback),
                           "t": round(time.time() - started, 2)})

            c.send(key)
            c.read_until_quiet()
            actions += 1

            new = c.text()
            st = parse_status(new)
            if st:
                status.update(st)

            # Stuck detection: if the screen stops changing, nudge with escape.
            if new == last_screen:
                unchanged += 1
                if unchanged >= 8:
                    c.send("\x1b")
                    c.read_until_quiet()
                    unchanged = 0
            else:
                unchanged = 0
            last_screen = new
    finally:
        c.close()

    # Parallel workers each write their own files; merge_shards() folds them
    # into the canonical ones afterwards. Concurrent appends of multi-KB JSON
    # lines from several processes are not reliably atomic, and a torn line
    # silently corrupts the dataset.
    traces_path = (TRACES if not args.tag
                   else TRACES.with_name(f"traces.{args.tag}.jsonl"))
    games_path = (GAMES if not args.tag
                  else GAMES.with_name(f"games.{args.tag}.jsonl"))

    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traces_path, "a", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "agent": args.agent_name,
        "turns": status.get("turns", 0),
        "xl": status.get("xl", 1),
        "depth": status.get("depth", 1),
        "death": death_line or ("action limit" if actions >= args.max_actions
                                else "stuck at prompt" if iters >= max_iters
                                else "ended"),
        "score": status.get("turns", 0),
        "actions": actions,
        "game": game_id,
        "ttyrec": ttyrec_path.name if ttyrec_path else "",
        "source": "pty",
    }
    with open(games_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    return row, len(traces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--max-actions", type=int, default=300)
    ap.add_argument("--name", default="bot")
    ap.add_argument("--species", default="Minotaur")
    ap.add_argument("--background", default="Berserker")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agent-name", default="pty-random")
    ap.add_argument("--tag", default="", help="shard suffix for parallel workers")
    ap.add_argument("--keep-save", action="store_true",
                    help="resume the previous character instead of starting a "
                         "new one each game (the old behaviour; produces "
                         "duplicate rows whenever a game wedges)")
    ap.add_argument("--no-ttyrec", action="store_true",
                    help="skip recording; bulk data collection doesn't need it")
    ap.add_argument("--show", action="store_true", help="print the live screen")
    ap.add_argument("--policy", choices=["random", "teacher"], default="random")
    ap.add_argument("--target-depth", type=int, default=99)
    args = ap.parse_args()

    total = 0
    for g in range(1, args.games + 1):
        row, n = play_one(args, g)
        total += n
        print(f"game {g}: turns={row['turns']} xl={row['xl']} "
              f"depth={row['depth']} actions={row['actions']} "
              f"traces={n} :: {row['death']}", flush=True)
    print(f"\n{total} (state, action) pairs -> {TRACES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Rule-based DCSS teacher — the competent player whose decisions we clone.

The point of this file is DATA, not glory. A random agent's keys are
unpredictable from the screen, so no model can learn them (label entropy ~1.0).
This one plays by rules, so its keys ARE a function of the state, and that
function is what the model gets to learn.

The policy delegates almost everything to DCSS's own smart commands rather
than reimplementing tactics:

    Tab   autofight — attack/approach the nearest monster
    o     auto-explore — pathfinds, picks up items, stops when something appears
    5     rest until healed or interrupted
    G >   travel to the nearest down staircase
    >     descend

So the rules are only about *which* of those to reach for:

    monster visible          -> Tab
    hurt and nothing around  -> 5
    level not explored       -> o
    level explored           -> G > then >

    python teacher_agent.py --target-depth 5
    watch at http://localhost:8090
"""
import argparse
import asyncio
import json
import re
import sys
import zlib
from datetime import datetime
from pathlib import Path

import websockets

HERE = Path(__file__).parent
GAMES = HERE / "games.jsonl"
TRACES = HERE / "data" / "traces.jsonl"

URI = "ws://127.0.0.1:8090/socket"
USER, PASSWORD = "midca", "midca"
GAME_ID = "bot-web-trunk"

COLS, ROWS = 80, 24
MAP_W, MAP_H = 60, 17

# Monsters render as letters. Items and terrain are punctuation, so this is a
# cheap and surprisingly reliable "is something hostile on screen".
MONSTER_RE = re.compile(r"[a-zA-Z]")
NOT_MONSTER = set("@")

DONE_EXPLORING = re.compile(
    r"Done exploring|Partly explored|explored this level|"
    r"Nothing left to explore|No target found", re.I)
NO_TARGET = re.compile(r"No target in view|No monsters in view", re.I)
COMES_INTO_VIEW = re.compile(r"comes into view|is nearby|You encounter|into view!", re.I)
CANT_DESCEND = re.compile(r"can't go down|no down staircase|Cannot travel", re.I)
DIED = re.compile(r"You die\.\.\.|You have died", re.I)
# Blocking prompts that only ONE specific key answers. Generic dismissers
# (esc/enter/space) do nothing, so the game sits there forever.
ATTR_PROMPT = re.compile(r"Increase \(S\)trength", re.I)
SKILL_PROMPT = re.compile(r"Set skill targets|skill training", re.I)

_inflate = zlib.decompressobj(-zlib.MAX_WBITS)

# webtiles takes printable keys as {"msg":"input","text":...} but control keys
# have to go as {"msg":"key","keycode":...}. Sending ESC/Tab/Enter as text is
# silently ignored — the game just sits there and the turn counter never moves.
KEYCODES = {"\x1b": 27, "\r": 13, "\n": 13, " ": 32, "\t": 9}


async def send_key(ws, key):
    if key in KEYCODES:
        await ws.send(json.dumps({"msg": "key", "keycode": KEYCODES[key]}))
    else:
        await ws.send(json.dumps({"msg": "input", "text": key}))


def decode(raw):
    if isinstance(raw, bytes):
        try:
            raw = _inflate.decompress(raw + b"\x00\x00\xff\xff").decode("utf-8")
        except Exception:
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return obj.get("msgs", [obj]) if isinstance(obj, dict) else []


class View:
    """Glyph grid + player stats. No graphics — the browser draws those."""

    def __init__(self):
        self.grid = {}
        self.player = {}

    def apply_map(self, m):
        if m.get("clear"):
            self.grid.clear()
        cx = cy = 0
        for cell in m.get("cells", []):
            if "x" in cell:
                cx = cell["x"]
            if "y" in cell:
                cy = cell["y"]
            g = cell.get("g")
            if g:
                self.grid[(cx, cy)] = g
            cx += 1

    def pos(self):
        p = self.player.get("pos") or {}
        return p.get("x", 0), p.get("y", 0)

    def monsters_near(self, radius=8):
        px, py = self.pos()
        n = 0
        for (x, y), g in self.grid.items():
            if abs(x - px) <= radius and abs(y - py) <= radius:
                if g not in NOT_MONSTER and MONSTER_RE.fullmatch(g):
                    n += 1
        return n

    def hp_frac(self):
        hp, mx = self.player.get("hp", 0), self.player.get("hp_max", 0) or 1
        return hp / mx

    def render(self):
        px, py = self.pos()
        hw, hh = MAP_W // 2, MAP_H // 2
        lines = []
        for dy in range(-hh, hh + 1):
            lines.append("".join(self.grid.get((px + dx, py + dy), " ")
                                 for dx in range(-hw, hw + 1))
                         [:MAP_W].ljust(MAP_W))
        p = self.player
        lines += [
            f"{p.get('name','?')} {p.get('title','')}".strip()[:COLS],
            f"{p.get('species','')} of {p.get('god','') or 'no god'}"[:COLS],
            f"Health: {p.get('hp',0)}/{p.get('hp_max',0)}  "
            f"Magic: {p.get('mp',0)}/{p.get('mp_max',0)}",
            f"AC: {p.get('ac',0)}  EV: {p.get('ev',0)}  SH: {p.get('sh',0)}",
            f"XL: {p.get('xl',1)}  Place: {p.get('place','Dungeon')}:"
            f"{p.get('depth',0)}  Turn: {p.get('turn',0)}",
        ]
        lines = [l[:COLS] for l in lines][:ROWS]
        while len(lines) < ROWS:
            lines.append("")
        return "\n".join(lines)


class Policy:
    """Which of DCSS's smart commands to reach for."""

    def __init__(self):
        self.explored = False      # level believed fully explored
        self.pending = []          # multi-key sequences (travel)
        self.descend_tries = 0
        self.stuck = 0             # actions since the turn counter last moved
        self.combat = False        # DCSS told us something is fightable
        self.recent = []           # last few game messages, for diagnosis
        self.answer = None         # forced reply to a single-key prompt
        self.level_start_turn = 0
        self.traveled = False      # already issued travel-to-stairs here

    def observe(self, text):
        if DONE_EXPLORING.search(text):
            self.explored = True
        if CANT_DESCEND.search(text):
            # travel didn't reach stairs; go back to exploring
            self.explored = False
            self.traveled = False
            self.descend_tries += 1

        # Let DCSS be the authority on whether anything is actually fightable.
        # Scanning glyphs for letters also matches monsters merely REMEMBERED
        # on the map but out of line of sight, and Tab then answers
        # "No target in view!" forever.
        if COMES_INTO_VIEW.search(text):
            self.combat = True
        if NO_TARGET.search(text):
            self.combat = False

        if ATTR_PROMPT.search(text):
            self.answer = "S"      # Berserker: strength
        elif SKILL_PROMPT.search(text):
            self.answer = "\x1b"

    def on_new_level(self, turn=0):
        self.explored = False
        self.level_start_turn = turn
        self.traveled = False
        self.pending.clear()
        self.descend_tries = 0

    def act(self, view):
        # A prompt with exactly one valid answer takes priority over everything.
        if self.answer:
            k, self.answer = self.answer, None
            return k

        # A level shouldn't take forever. If we've spent a long time here,
        # stop exploring and go find the stairs.
        if view.player.get("turn", 0) - self.level_start_turn > 900:
            self.explored = True

        # If the turn counter has stopped moving the game is sitting at a
        # prompt our key doesn't answer, and every further key is wasted. Cycle
        # through the universal dismissers rather than hammering the same one.
        if self.stuck >= 6:
            k = ["\x1b", "\r", " ", "y"][(self.stuck // 6) % 4]
            self.stuck += 1
            return k

        if self.pending:
            return self.pending.pop(0)

        if self.combat:
            return "\t"                        # autofight

        if view.hp_frac() < 0.6:
            return "5"                         # rest

        if not self.explored:
            return "o"                         # auto-explore

        # Level explored: descend.
        #
        # Deliberately just `>`. DCSS auto-travels to the nearest known down
        # staircase when you press it off-stairs, so no menu is involved.
        # Both alternatives failed here: `G` answers "Unknown command", and `X`
        # opens the level map — a UI overlay this agent is BLIND to, because we
        # only track `map` and `player` messages while menus arrive as separate
        # ui-stack/menu messages. Anything menu-driven wedges us.
        return ">"


async def run(args):
    view, pol = View(), Policy()
    traces, sent, games_played = [], 0, 0
    started = False
    reached = False
    last_depth = 0
    game_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-teach"
    TRACES.parent.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(URI, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"msg": "login", "username": USER,
                                  "password": PASSWORD}))
        loop = asyncio.get_event_loop()
        end = loop.time() + args.timeout

        while loop.time() < end and sent < args.max_actions and not reached:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                msgs = decode(raw)
            except asyncio.TimeoutError:
                msgs = []

            for m in msgs:
                t = m.get("msg")
                if t == "ping":
                    await ws.send(json.dumps({"msg": "pong"}))
                elif t == "login_success":
                    await ws.send(json.dumps({"msg": "play", "game_id": GAME_ID}))
                elif t == "login_fail":
                    print("login failed", file=sys.stderr)
                    return 1
                elif t == "game_started":
                    started = True
                    games_played += 1
                    print(f"game {games_played} started", flush=True)
                elif t == "map":
                    view.apply_map(m)
                elif t == "player":
                    view.player.update(m)
                elif t == "msgs":
                    text = " ".join(x.get("text", "")
                                    for x in m.get("messages", []))
                    pol.observe(text)
                    if text.strip():
                        pol.recent = (pol.recent + [text])[-6:]
                    if DIED.search(text):
                        print(f"  died on D:{view.player.get('depth',0)} "
                              f"at turn {view.player.get('turn',0)}", flush=True)
                        started = False
                        # abandon and start a fresh character
                        await ws.send(json.dumps({"msg": "key", "keycode": 17}))
                        await asyncio.sleep(0.4)
                        await ws.send(json.dumps({"msg": "input", "text": "yes\r"}))
                        await asyncio.sleep(1.0)
                        view = View()
                        pol = Policy()
                        await ws.send(json.dumps({"msg": "play",
                                                  "game_id": GAME_ID}))
                elif t == "game_ended":
                    started = False

            if not started:
                await asyncio.sleep(0.2)
                continue

            depth = view.player.get("depth", 0) or 0
            if depth != last_depth:
                if depth:
                    print(f"  reached D:{depth} "
                          f"(turn {view.player.get('turn',0)}, "
                          f"XL {view.player.get('xl',1)})", flush=True)
                pol.on_new_level(view.player.get("turn", 0))
                last_depth = depth
            if depth >= args.target_depth:
                reached = True
                print(f"TARGET REACHED: D:{depth}", flush=True)
                break

            turn_now = view.player.get("turn", 0)
            if turn_now == getattr(pol, "_last_turn", None):
                pol.stuck += 1
                if pol.stuck == 6:
                    print("  [stuck: turn not advancing] last messages:",
                          flush=True)
                    for line in pol.recent[-4:]:
                        print("    " + re.sub(r"<[^>]+>", "", line)[:150],
                              flush=True)
            else:
                pol.stuck = 0
                pol._last_turn = turn_now

            state = view.render()
            key = pol.act(view)
            traces.append({"state": state, "action": key, "game": game_id,
                           "source": "teacher"})
            await send_key(ws, key)
            sent += 1
            if sent % 100 == 0:
                print(f"  {sent} keys  D:{depth} turn={view.player.get('turn',0)} "
                      f"hp={view.player.get('hp',0)}/{view.player.get('hp_max',0)}",
                      flush=True)
            await asyncio.sleep(args.delay)

        try:
            await ws.send(json.dumps({"msg": "key", "keycode": 17}))
            await asyncio.sleep(0.4)
            await ws.send(json.dumps({"msg": "input", "text": "yes\r"}))
            await asyncio.sleep(0.4)
        except Exception:
            pass

    with open(TRACES, "a", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    p = view.player
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "agent": "teacher-rules",
        "turns": p.get("turn", 0),
        "xl": p.get("xl", 1),
        "depth": p.get("depth", 0),
        "death": "reached target" if reached else "action limit",
        "score": p.get("turn", 0),
        "actions": sent,
        "game": game_id,
        "source": "teacher",
    }
    with open(GAMES, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print(f"\ndone: deepest D:{p.get('depth',0)}, {sent} keys, "
          f"{len(traces)} traces, {games_played} character(s)")
    return 0 if reached else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-depth", type=int, default=5)
    ap.add_argument("--max-actions", type=int, default=4000)
    ap.add_argument("--delay", type=float, default=0.08)
    ap.add_argument("--timeout", type=float, default=3600)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

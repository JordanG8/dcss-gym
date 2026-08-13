"""
Play DCSS through webtiles, so you can WATCH in the browser while the agent
reads pure game state.

One game, two views:
  - your browser renders tiles/graphics from the server's messages
  - this agent reads the same messages as glyphs + stats, no graphics at all

Why not dcss-ai-wrapper: it is ~10,000 lines that reconstruct full game state
and its parser doesn't understand DCSS 0.35. We need far less - a glyph grid
and a status line - so this is ~200 lines that work against current trunk.

DCSS records the .ttyrec itself for games run under webtiles, so replay comes
free.

    python webtiles_agent.py --actions 400 --delay 0.3
    watch at http://localhost:8090
"""
import argparse
import asyncio
import json
import random
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
GAME_ID = "bot-web-trunk"      # preselects Minotaur Berserker + hand axe

COLS, ROWS = 80, 24
MAP_W, MAP_H = 60, 17          # map window; rest of the 80x24 is status

MOVE_KEYS = list("hjklyubn")
ACTION_KEYS = MOVE_KEYS * 4 + ["o"]     # 'o' = auto-explore

# One persistent context for the whole connection - webtiles deflates each
# frame and strips the trailing 00 00 FF FF. A fresh context per frame fails.
_inflate = zlib.decompressobj(-zlib.MAX_WBITS)


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


class GameView:
    """The agent's view: a glyph grid plus player stats. No graphics."""

    def __init__(self):
        self.grid = {}
        self.player = {}

    def apply_map(self, m):
        if m.get("clear"):
            self.grid.clear()
        cx = cy = 0
        for cell in m.get("cells", []):
            # Cells stream left-to-right; x/y appear only when the run jumps.
            if "x" in cell:
                cx = cell["x"]
            if "y" in cell:
                cy = cell["y"]
            g = cell.get("g")
            if g:
                self.grid[(cx, cy)] = g
            cx += 1

    def apply_player(self, m):
        self.player.update(m)

    def render(self):
        """80x24 text: map window centred on the player, then a status block."""
        p = self.player
        pos = p.get("pos") or {}
        px, py = pos.get("x", 0), pos.get("y", 0)

        half_w, half_h = MAP_W // 2, MAP_H // 2
        lines = []
        for dy in range(-half_h, half_h + 1):
            row = []
            for dx in range(-half_w, half_w + 1):
                row.append(self.grid.get((px + dx, py + dy), " "))
            lines.append("".join(row)[:MAP_W].ljust(MAP_W))

        place = p.get("place", "Dungeon")
        depth = p.get("depth", 0)
        status = [
            # DCSS's `title` already includes the article ("the Trooper").
            f"{p.get('name','?')} {p.get('title','')}".strip()[:COLS],
            f"{p.get('species','')} of {p.get('god','') or 'no god'}".strip()[:COLS],
            f"Health: {p.get('hp',0)}/{p.get('hp_max',0)}  "
            f"Magic: {p.get('mp',0)}/{p.get('mp_max',0)}",
            f"AC: {p.get('ac',0)}  EV: {p.get('ev',0)}  SH: {p.get('sh',0)}  "
            f"Str: {p.get('str',0)} Int: {p.get('int',0)} Dex: {p.get('dex',0)}",
            f"XL: {p.get('xl',1)}  Place: {place}:{depth}  Turn: {p.get('turn',0)}",
        ]
        out = lines + status
        out = [l[:COLS] for l in out][:ROWS]
        while len(out) < ROWS:
            out.append("")
        return "\n".join(out)


async def run(args):
    view = GameView()
    started = False
    ended = ""
    sent = 0
    traces = []
    TRACES.parent.mkdir(parents=True, exist_ok=True)
    game_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    async with websockets.connect(URI, max_size=None, ping_interval=None) as ws:
        print(f"connected {URI}", flush=True)
        await ws.send(json.dumps({"msg": "login", "username": USER,
                                  "password": PASSWORD}))

        loop = asyncio.get_event_loop()
        end = loop.time() + args.timeout

        while loop.time() < end and sent < args.actions and not ended:
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
                    print("logged in", flush=True)
                    await ws.send(json.dumps({"msg": "play", "game_id": GAME_ID}))
                elif t == "login_fail":
                    print("LOGIN FAILED - register the account first",
                          file=sys.stderr)
                    return 1
                elif t == "game_started":
                    started = True
                    print("game started -> watch http://localhost:8090",
                          flush=True)
                elif t == "map":
                    view.apply_map(m)
                elif t == "player":
                    view.apply_player(m)
                elif t == "game_ended":
                    ended = str(m.get("reason", "ended"))
                elif t == "msgs":
                    for line in m.get("messages", []):
                        txt = line.get("text", "")
                        if "You die" in txt or "You have died" in txt:
                            ended = "died"

            if started and not ended:
                state = view.render()
                key = random.choice(ACTION_KEYS)
                traces.append({"state": state, "action": key, "game": game_id,
                               "source": "webtiles"})
                await ws.send(json.dumps({"msg": "input", "text": key}))
                sent += 1
                if sent % 50 == 0:
                    p = view.player
                    print(f"  {sent} keys  turn={p.get('turn',0)} "
                          f"hp={p.get('hp',0)}/{p.get('hp_max',0)} "
                          f"cells={len(view.grid)}", flush=True)
                await asyncio.sleep(args.delay)

        # Abandon the character so the next run starts clean.
        try:
            await ws.send(json.dumps({"msg": "key", "keycode": 17}))   # ctrl-q
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"msg": "input", "text": "yes\r"}))
            await asyncio.sleep(0.5)
        except Exception:
            pass

    with open(TRACES, "a", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    p = view.player
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "agent": args.agent_name,
        "turns": p.get("turn", 0),
        "xl": p.get("xl", 1),
        "depth": p.get("depth", 1),
        "death": ended or "action limit",
        "score": p.get("turn", 0),
        "actions": sent,
        "game": game_id,
        "source": "webtiles",
    }
    with open(GAMES, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print(f"done: {sent} keys, turn={row['turns']}, "
          f"{len(traces)} traces, {row['death']}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=300)
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds between keys; higher is easier to watch")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--agent-name", default="webtiles-random")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

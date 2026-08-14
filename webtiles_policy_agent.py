"""Run a trained PPO policy through WebTiles and record a native tile replay.

This is deliberately separate from the rule-based teacher.  Every game action
below comes from ``Policy`` logits over the player-visible glyph/status screen.
Only non-choice UI protocol messages (ping and game start) are handled here.

Example (known historical PPO checkpoint):
  python webtiles_policy_agent.py --checkpoint /mnt/c/Users/jorda/dcss-research/data/rl_policy.c.pt
"""
import argparse
import asyncio
import hashlib
import html
import json
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import websockets

from attic.teacher_agent import View, decode, send_key
from dcss_env import RE_NO_TARGET, VARIANTS
from train_rl import Policy, encode, load_policy_state


HERE = Path(__file__).parent
TILE_REPLAYS = HERE / "data" / "webtiles_replays"
GAMES = HERE / "games.jsonl"
LIVE = HERE / "data" / "webtiles_policy_live.json"
BOT_SAVE_ROOT = Path("/root/crawl/crawl-ref/source/saves")
URI = "ws://127.0.0.1:8090/socket"
GAME_ID = "bot-web-trunk"
RE_CANT_DESCEND = re.compile(r"can't go down|cannot go down|no down staircase", re.I)
RE_CANT_BERSERK = re.compile(r"too berserk|too exhausted|cannot berserk", re.I)
RE_CANT_EXPLORE = re.compile(
    r"done exploring|nothing left to explore|partly explored|"
    r"explored this level", re.I)
RE_DOWN_FEATURE = re.compile(
    r"staircase.*leading down|escape hatch.*floor|gateway.*down|"
    r"stairs.*leading down", re.I)
RE_AUTOFIGHT_BLOCKED = re.compile(
    r"caught in a web|held in a net|cannot attack while held", re.I)


class PolicyView(View):
    """WebTiles view that retains visible monster attitude metadata."""

    def __init__(self):
        super().__init__()
        self.hostiles = set()
        self.terrain = {}
        self.monster_attitudes = {}

    def apply_map(self, message):
        if message.get("clear"):
            self.hostiles.clear()
            self.terrain.clear()
        cx = cy = 0
        for cell in message.get("cells", []):
            if "x" in cell:
                cx = cell["x"]
            if "y" in cell:
                cy = cell["y"]
            if "mon" in cell:
                self.hostiles.discard((cx, cy))
                monster = cell.get("mon")
                if monster:
                    monster_id = monster.get("id")
                    if monster_id is not None and "att" in monster:
                        self.monster_attitudes[monster_id] = monster["att"]
                    attitude = monster.get(
                        "att", self.monster_attitudes.get(monster_id))
                    if attitude == 0:
                        self.hostiles.add((cx, cy))
            elif "g" in cell:
                # A foreground glyph explicitly replaces the old foreground.
                # Letter glyphs are monsters; metadata is normally present,
                # but retaining this visible fallback handles compact deltas.
                self.hostiles.discard((cx, cy))
                glyph = cell.get("g", "")
                if glyph != "@" and re.fullmatch(r"[A-Za-z]", glyph):
                    self.hostiles.add((cx, cy))
            glyph = cell.get("g")
            if (glyph is not None and glyph != "@"
                    and not re.fullmatch(r"[A-Za-z]", glyph)):
                self.terrain[(cx, cy)] = glyph
            cx += 1
        super().apply_map(message)

    def monsters_near(self, radius=8):
        px, py = self.pos()
        exact = sum(abs(x - px) <= radius and abs(y - py) <= radius
                    for x, y in self.hostiles)
        # Compact delta streams can omit `mon` after the initial foreground.
        # The rendered letter is still player-visible and is the same fallback
        # used by the original WebTiles observer.
        return max(exact, super().monsters_near(radius))


def plain(text):
    """Remove WebTiles colour markup from player-visible menu text."""
    return html.unescape(re.sub(r"<[^>]*>", "", str(text or "")))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish_live(payload):
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIVE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(LIVE)


async def neural_key(ws, name, view, macro_delay=0.35):
    """Execute one model-selected macro using the same meanings as PPO C.

    `travel` matches the PPO environment's public level-map macro: open the
    map, cycle to a downward feature, then confirm. `descend` remains a bare
    `>` because it means "go down here" when the network has already navigated
    onto stairs.
    """
    if name == "autofight":
        await send_key(ws, "\t")
    elif name == "explore":
        await send_key(ws, "o")
    elif name == "rest":
        await send_key(ws, "5")
    elif name == "descend":
        await send_key(ws, ">")
    elif name == "travel":
        # `View.grid` is the remembered map the player can see in WebTiles.
        # Never confirm a level-map destination unless a real down-stair glyph
        # is already known; otherwise Crawl can select an unrelated feature or
        # route back upstairs. This is the same public-information guard used
        # by DCSSEnv's travel macro.
        if not any(g == ">" for g in view.grid.values()):
            return
        await send_key(ws, "X")
        # The remaining keys are driven by visible ui_state/message packets in
        # run(); fixed sleeps lose keystrokes when eight games share WebTiles.
    elif name == "escape":
        await send_key(ws, "\x1b")
    elif name == "berserk":
        # Open the visible ability menu. The displayed Berserk hotkey is sent
        # only when its menu packet arrives; fixed sleeps are unreliable under
        # parallel WebTiles load.
        await send_key(ws, "a")
    elif name == "wait":
        await send_key(ws, ".")
    elif name.startswith("move_"):
        keys = {
            "move_n": "k", "move_ne": "u", "move_e": "l",
            "move_se": "n", "move_s": "j", "move_sw": "b",
            "move_w": "h", "move_nw": "y",
        }
        await send_key(ws, keys[name])
    else:
        raise ValueError(f"unsupported neural action: {name}")


def choose(model, screen, deterministic, action_mask=None, hostile_cells=None):
    """Return a neural action plus its complete probability distribution."""
    with torch.no_grad():
        logits, value = model(
            encode(screen, hostile_cells=hostile_cells).unsqueeze(0))
        if action_mask is not None:
            legal = torch.tensor(action_mask, dtype=torch.bool,
                                 device=logits.device)
            if not bool(legal.any()):
                raise ValueError("WebTiles action mask removed every action")
            logits = logits.masked_fill(~legal.unsqueeze(0), -1e9)
        probs = torch.softmax(logits[0], dim=-1)
        if deterministic:
            action = int(probs.argmax())
        else:
            action = int(torch.multinomial(probs, 1))
    return action, [float(x) for x in probs], float(value[0])


def visible_signature(view):
    """Player-visible progress state used to scope a rejected-action mask."""
    p = view.player
    pos = p.get("pos") or {}
    return (p.get("depth"), p.get("turn"), p.get("hp"),
            pos.get("x"), pos.get("y"))


def visible_action_mask(names, view, signature, rejected_at):
    """Remove commands Crawl will reject from this visible map state."""
    mask = [True] * len(names)
    masked = []
    hostile_visible = view.monsters_near() > 0
    rejected = ({"explore", "rest", "travel"}
                if hostile_visible else {"autofight"})
    if not hostile_visible:
        rejected.add("berserk")
    rejected.add("escape")
    rejected.update(name for name, rejected_signature in rejected_at.items()
                    if signature == rejected_signature)
    on_down_stair = getattr(view, "terrain", {}).get(view.pos()) == ">"
    if not any(glyph == ">" for glyph in view.grid.values()):
        rejected.add("travel")
    if on_down_stair:
        rejected.add("travel")
    else:
        rejected.add("descend")
    hp = int(view.player.get("hp", 0) or 0)
    hp_max = int(view.player.get("hp_max", 0) or 0)
    if hp_max and hp >= hp_max:
        rejected.add("rest")
    statuses = view.player.get("status") or []
    status_text = " ".join(
        f"{item.get('light', '')} {item.get('text', '')}"
        for item in statuses if isinstance(item, dict)).lower()
    if "berserk" in status_text:
        rejected.update({"explore", "rest", "descend", "travel", "berserk"})
    elif "exhaust" in status_text:
        rejected.add("berserk")
    for name in names:
        if name in rejected:
            mask[names.index(name)] = False
            masked.append(name)
    return mask, masked


def choose_context(model, screen, choices, deterministic, hostile_cells=None):
    """Make a genuine neural choice in a contextual WebTiles prompt.

    The historical PPO has seven command-mode logits and no separate menu
    head.  For a three-way prompt, the first three logits are re-normalised in
    the prompt's displayed order.  This is deliberately recorded as a
    contextual decision instead of disguising a scripted answer as policy
    output.
    """
    with torch.no_grad():
        logits, value = model(
            encode(screen, hostile_cells=hostile_cells).unsqueeze(0))
        full_probs = torch.softmax(logits[0], dim=-1)
        context_probs = torch.softmax(logits[0, :len(choices)], dim=-1)
        if deterministic:
            choice = int(context_probs.argmax())
        else:
            choice = int(torch.multinomial(context_probs, 1))
    return (choice, [float(x) for x in full_probs],
            [float(x) for x in context_probs], float(value[0]))


def terminal_layout(view, messages):
    """Render the public WebTiles view in the 80x24 PTY layout PPO learned.

    Crawl's terminal puts a 40-column map on the left, a compact player panel
    on the upper-right, and messages in rows 17--23.  This is presentation
    alignment only: each character comes from the player-visible WebTiles map,
    player packet, or message pane.
    """
    lines = [[" "] * 80 for _ in range(24)]

    def put(x, y, value):
        for i, char in enumerate(str(value)[:max(0, 80 - x)]):
            lines[y][x + i] = char

    px, py = view.pos()
    for dy in range(-8, 9):
        for dx in range(-20, 20):
            lines[dy + 8][dx + 20] = view.grid.get((px + dx, py + dy), " ")

    p = view.player
    title = f"{p.get('name', '?')} {p.get('title', '')}".strip()
    species = f"{p.get('species', '')} of {p.get('god') or 'no god'}".strip()
    put(40, 0, title)
    put(40, 1, species)
    put(40, 2, f"Health: {p.get('hp', 0)}/{p.get('hp_max', 0)}")
    put(40, 3, f"Magic:  {p.get('mp', 0)}/{p.get('mp_max', 0)}")
    put(40, 4, f"AC: {p.get('ac', 0):2}            Str: {p.get('str', 0)}")
    put(40, 5, f"EV: {p.get('ev', 0):2}            Int: {p.get('int', 0)}")
    put(40, 6, f"SH: {p.get('sh', 0):2}            Dex: {p.get('dex', 0)}")
    put(40, 7, f"XL: {p.get('xl', 1):2}  Place: {p.get('place', 'Dungeon')}:{p.get('depth', 0)}")
    put(40, 8, f"Turn: {p.get('turn', 0)}")
    for row, message in enumerate(messages[-7:], start=17):
        put(0, row, message)
    return "\n".join("".join(row) for row in lines)


def terminal_hostiles(view):
    """Translate exact player-visible WebTiles monster cells to terminal cells."""
    px, py = view.pos()
    return {(x - px + 20, y - py + 8) for x, y in view.hostiles
            if -20 <= x - px < 20 and -8 <= y - py <= 8}


async def run(args):
    if args.variant != "c":
        raise SystemExit("WebTiles neural runner currently supports variant c only")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    names = [name for name, _key in VARIANTS[args.variant]]
    model = Policy(len(names))
    report = load_policy_state(
        model, torch.load(checkpoint, map_location="cpu"))
    if report["expanded"] or report["skipped"]:
        print(f"checkpoint migration: {report}", flush=True)
    model.eval()
    model_hash = digest(checkpoint)

    bot_save = BOT_SAVE_ROOT / f"{args.username}.cs"
    if args.fresh and bot_save.exists():
        archive = HERE / "data" / "bot_saves"
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / (f"{args.username}-"
                            f"{datetime.now():%Y%m%d-%H%M%S}.cs")
        shutil.move(str(bot_save), str(target))
        print(f"archived previous bot save: {target}", flush=True)

    view = PolicyView()
    events, decisions = [], []
    forced_acks = []
    started = False
    outcome = "action limit"
    reached = False
    sent = 0
    last_rx = 0.0
    last_sent_at = 0.0
    last_decision_screen = None
    last_action_signature = None
    recent_messages = []
    input_mode = None
    pending_context = None
    shop_macro = None
    pending_macro = None
    macro_sent_at = 0.0
    ui_state = 0
    registration_attempted = False
    last_more_ack_at = 0.0
    started_at = time.time()
    last_progress_at = started_at
    last_activity_at = started_at
    progress_signature = None
    rejected_at = {}
    run_id = (datetime.now().strftime("%Y%m%d-%H%M%S")
              + f"-s{args.slot + 1}-neural-c")

    async with websockets.connect(URI, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"msg": "login", "username": args.username,
                                  "password": args.password}))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + args.timeout
        while loop.time() < deadline and sent < args.max_actions and not reached:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=args.poll)
                messages = decode(raw)
            except asyncio.TimeoutError:
                messages = []

            if messages:
                last_rx = loop.time()
            for message in messages:
                events.append({"t": sent, "data": message})
                typ = message.get("msg")
                if typ != "ping":
                    last_activity_at = time.time()
                if typ == "ping":
                    await ws.send(json.dumps({"msg": "pong"}))
                elif typ == "login_success":
                    await ws.send(json.dumps({"msg": "play", "game_id": GAME_ID}))
                elif typ == "login_fail":
                    if args.register and not registration_attempted:
                        registration_attempted = True
                        await ws.send(json.dumps({
                            "msg": "register", "username": args.username,
                            "password": args.password, "email": "",
                        }))
                    else:
                        raise RuntimeError(
                            f"WebTiles login failed for {args.username}")
                elif typ == "register_fail":
                    raise RuntimeError(
                        f"WebTiles registration failed for {args.username}: "
                        f"{message.get('reason', 'unknown reason')}")
                elif typ == "game_started":
                    started = True
                elif typ == "map":
                    view.apply_map(message)
                elif typ == "player":
                    view.player.update(message)
                elif typ == "input_mode":
                    input_mode = message.get("mode")
                    if (input_mode == 1 and pending_macro
                            and pending_macro["kind"] == "travel"
                            and pending_macro["stage"] == "confirm"):
                        pending_macro = None
                    if input_mode == 5 and loop.time() - last_more_ack_at > 0.2:
                        # WebTiles mouse mode 5 is MORE. Some transitions
                        # (notably shafts) omit `msgs.more`, so the mode is the
                        # authoritative player-visible protocol signal. Enter
                        # is mandatory UI progression, not a policy choice.
                        await send_key(ws, "\r")
                        last_more_ack_at = loop.time()
                        forced_acks.append({
                            "t": sent, "kind": "more_input_mode",
                            "key": "Enter",
                        })
                    if input_mode == 8 and shop_macro == "purchasing":
                        await send_key(ws, "Y")
                        forced_acks.append({
                            "t": sent, "kind": "shop_purchase_confirm",
                            "key": "Y",
                        })
                elif typ == "ui_state":
                    ui_state = int(message.get("state", 0) or 0)
                    if (pending_macro and pending_macro["kind"] == "travel"
                            and ui_state == 2
                            and pending_macro["stage"] == "open"):
                        await send_key(ws, ">")
                        pending_macro["stage"] = "select_down"
                        pending_macro["steps"] += 1
                        macro_sent_at = loop.time()
                        forced_acks.append({
                            "t": sent, "kind": "neural_macro_step",
                            "action": "travel", "key": ">",
                        })
                    elif (pending_macro
                          and pending_macro["kind"] == "travel"
                          and ui_state == 0
                          and pending_macro["stage"] == "confirm"):
                        pending_macro = None
                elif typ == "menu" and message.get("tag") == "ability":
                    berserk_key = None
                    for item in message.get("items", []):
                        if "Berserk" in plain(item.get("text")):
                            hotkeys = item.get("hotkeys") or []
                            if hotkeys:
                                berserk_key = chr(int(hotkeys[0]))
                                break
                    if berserk_key:
                        await send_key(ws, berserk_key)
                        forced_acks.append({
                            "t": sent, "kind": "neural_macro_completion",
                            "action": "berserk", "key": berserk_key,
                        })
                    else:
                        await send_key(ws, "\x1b")
                        rejected_at["berserk"] = visible_signature(view)
                        forced_acks.append({
                            "t": sent, "kind": "ability_missing_berserk",
                            "key": "Escape",
                        })
                elif typ == "menu" and message.get("tag") == "shop":
                    if shop_macro == "purchasing":
                        await send_key(ws, "\x1b")
                        shop_macro = None
                        forced_acks.append({
                            "t": sent, "kind": "shop_purchase_exit",
                            "key": "Escape",
                        })
                        continue
                    title = plain((message.get("title") or {}).get("text"))
                    more = plain(message.get("more"))
                    gold_match = re.search(r"You have (\d+) gold", more)
                    gold = int(gold_match.group(1)) if gold_match else 0
                    affordable = []
                    menu_lines = [title, more]
                    for item in message.get("items", []):
                        item_text = plain(item.get("text"))
                        menu_lines.append(item_text)
                        price_match = re.search(r"(\d+) gold", item_text)
                        hotkeys = item.get("hotkeys") or []
                        if (price_match and hotkeys
                                and int(price_match.group(1)) <= gold):
                            affordable.append((
                                f"shop_item_{chr(int(hotkeys[0]))}",
                                chr(int(hotkeys[0]))))
                    if not affordable:
                        # With no affordable item, exiting is the sole legal
                        # transaction. This is transport/UI plumbing, derived
                        # only from the visible gold and price text.
                        await send_key(ws, "\x1b")
                        forced_acks.append({
                            "t": sent, "kind": "shop_no_affordable_items",
                            "key": "Escape", "gold": gold,
                        })
                    else:
                        # Exit plus affordable hotkeys become a contextual
                        # neural action set. Seven is the checkpoint's output
                        # width; remaining items can be reached on later menu
                        # pages/decisions.
                        pending_context = {
                            "kind": "shop", "lines": menu_lines,
                            "choices": [("shop_exit", "\x1b")] + affordable[:6],
                        }
                elif typ == "update_menu" and shop_macro == "marked":
                    footer = plain(message.get("more"))
                    if "buy marked items" in footer:
                        await send_key(ws, "\r")
                        shop_macro = "purchasing"
                        forced_acks.append({
                            "t": sent, "kind": "shop_purchase_checkout",
                            "key": "Enter",
                        })
                elif typ == "game_ended":
                    outcome = str(message.get("reason", "game ended"))
                    break
                elif typ == "msgs":
                    text = " ".join(x.get("text", "")
                                     for x in message.get("messages", []))
                    recent_messages = (recent_messages + [text])[-2:]
                    if (pending_macro and pending_macro["kind"] == "travel"
                            and pending_macro["stage"] == "select_down"
                            and RE_DOWN_FEATURE.search(plain(text))):
                        await send_key(ws, ".")
                        pending_macro["stage"] = "confirm"
                        pending_macro["steps"] += 1
                        macro_sent_at = loop.time()
                        forced_acks.append({
                            "t": sent, "kind": "neural_macro_completion",
                            "action": "travel", "key": ".",
                        })
                    if (RE_NO_TARGET.search(text)
                            or RE_AUTOFIGHT_BLOCKED.search(text)):
                        # Crawl visibly rejected Tab. Keep it unavailable only
                        # while the exact visible progress state is unchanged;
                        # another action that moves, spends a turn, or changes
                        # HP makes it eligible again. This prevents a frozen
                        # no-target loop without injecting a tactical fallback.
                        rejected_at["autofight"] = visible_signature(view)
                    if RE_CANT_DESCEND.search(text):
                        rejected_at["descend"] = visible_signature(view)
                    if RE_CANT_BERSERK.search(text):
                        rejected_at["berserk"] = visible_signature(view)
                    if RE_CANT_EXPLORE.search(text):
                        rejected_at["explore"] = visible_signature(view)
                    if "You die" in text or "You have died" in text:
                        outcome = "died"
                        break

            if outcome in {"died", "game ended"}:
                break
            depth = int(view.player.get("depth", 0) or 0)
            if started and depth >= args.target_depth:
                reached, outcome = True, f"reached D:{depth}"
                break

            p = view.player
            signature = visible_signature(view)
            if signature != progress_signature:
                progress_signature = signature
                last_progress_at = time.time()
            validation_remaining = max(0, args.validation_deadline - time.time())
            phase = "validating" if validation_remaining else "running"
            last_decision = decisions[-1] if decisions else {}
            publish_live({
                "running": True, "phase": phase,
                "validation_remaining_s": round(validation_remaining),
                "depth": depth, "turn": p.get("turn", 0), "hp": p.get("hp", 0),
                "hp_max": p.get("hp_max", 0), "xl": p.get("xl", 1),
                "actions": sent, "input_mode": input_mode,
                "last_action": last_decision.get("action"),
                "action_names": last_decision.get("display_names", names),
                "action_probabilities": last_decision.get(
                    "display_probabilities", []),
                "value": last_decision.get("value"),
                "masked_actions": last_decision.get("masked_actions", []),
                "checkpoint_sha256": model_hash, "run": run_id,
                "attempt": args.attempt, "best_depth": args.best_depth,
                "slot": args.slot, "username": args.username,
            })
            if started and time.time() - last_activity_at > args.stall_timeout:
                outcome = ("interface silent "
                           f"({round(time.time() - last_activity_at)}s)")
                print(f"RESULT {outcome}", flush=True)
                break

            if started and pending_context:
                context = pending_context
                screen = terminal_layout(view, context["lines"][-7:])
                if (screen == last_decision_screen
                        and loop.time() - last_sent_at < args.retry):
                    continue
                choices = context["choices"]
                choice, probs, context_probs, value = choose_context(
                    model, screen, choices, args.deterministic,
                    hostile_cells=terminal_hostiles(view))
                name, key = choices[choice]
                decisions.append({
                    "t": sent, "action": name, "context": context["kind"],
                    "choice_index": choice, "probabilities": probs,
                    "context_probabilities": context_probs, "value": value,
                    "display_names": [item[0] for item in choices],
                    "display_probabilities": context_probs,
                })
                await send_key(ws, key)
                if context["kind"] == "shop" and name != "shop_exit":
                    shop_macro = "marked"
                sent += 1
                last_decision_screen = screen
                last_sent_at = loop.time()
                pending_context = None
                continue

            if started and pending_macro:
                if loop.time() - macro_sent_at < args.retry:
                    continue
                if pending_macro["steps"] >= 12:
                    await send_key(ws, "\x1b")
                    rejected_at["travel"] = pending_macro["signature"]
                    forced_acks.append({
                        "t": sent, "kind": "neural_macro_cancel",
                        "action": "travel", "key": "Escape",
                    })
                    pending_macro = None
                    continue
                key = ("X" if pending_macro["stage"] == "open"
                       else ">" if pending_macro["stage"] == "select_down"
                       else ".")
                await send_key(ws, key)
                pending_macro["steps"] += 1
                macro_sent_at = loop.time()
                forced_acks.append({
                    "t": sent, "kind": "neural_macro_retry",
                    "action": "travel", "key": key,
                })
                continue

            if (started and input_mode == 5
                    and loop.time() - last_more_ack_at > args.retry):
                # A dropped mandatory acknowledgement must not masquerade as
                # a stalled neural policy. This bounded retry is visible in
                # provenance and does not consume a neural decision.
                await send_key(ws, "\r")
                last_more_ack_at = loop.time()
                forced_acks.append({
                    "t": sent, "kind": "more_retry", "key": "Enter",
                })
                continue

            # The level-up prompt is a real three-way choice, so it must not be
            # answered by interface code. The checkpoint predates menu heads;
            # re-use and re-normalise its first three logits in displayed
            # S/I/D order, and keep the complete provenance in the replay.
            attribute_prompt = any("Increase (S)trength" in m
                                   for m in recent_messages)
            if started and view.player and input_mode == 7 and attribute_prompt:
                screen = terminal_layout(view, recent_messages)
                if (screen == last_decision_screen
                        and loop.time() - last_sent_at < args.retry):
                    continue
                choices = (("stat_strength", "S"),
                           ("stat_intelligence", "I"),
                           ("stat_dexterity", "D"))
                choice, probs, context_probs, value = choose_context(
                    model, screen, choices, args.deterministic,
                    hostile_cells=terminal_hostiles(view))
                name, key = choices[choice]
                decisions.append({
                    "t": sent, "action": name, "context": "attribute_prompt",
                    "choice_index": choice, "probabilities": probs,
                    "context_probabilities": context_probs, "value": value,
                    "display_names": [item[0] for item in choices],
                    "display_probabilities": context_probs,
                })
                await send_key(ws, key)
                sent += 1
                last_decision_screen = screen
                last_sent_at = loop.time()
                continue

            # DCSS macro commands can produce several map/player updates while
            # they run. Decide only after the visible stream has settled, the
            # WebTiles equivalent of the PTY environment's read-until-quiet.
            # WebTiles mouse/input modes are defined in enums.js. COMMAND is
            # mode 1 (NORMAL is 0); only COMMAND accepts normal game actions.
            if (not started or not view.player or input_mode != 1
                    or loop.time() - last_rx < args.settle):
                continue

            screen = terminal_layout(view, recent_messages)

            # Do not resend a decision into an unchanged WebTiles state. A
            # macro can take a while, during which transport-level messages
            # still arrive; treating those as new observations made the first
            # adapter spam the same UI. A slow retry retains recovery from a
            # truly ignored key while keeping each model decision auditable.
            if screen == last_decision_screen and loop.time() - last_sent_at < args.retry:
                continue
            action_mask, masked_actions = visible_action_mask(
                names, view, signature, rejected_at)
            # A primitive action that leaves depth, turn, HP and position all
            # unchanged was visibly rejected (normally a move into a wall).
            # Scope the rejection to this exact observation so the policy can
            # immediately choose another direction, while the same direction
            # becomes legal again after any real progress.  This is an action
            # mask derived from public outcome, not a scripted replacement.
            prior_name = decisions[-1].get("action") if decisions else None
            if (prior_name and last_action_signature == signature
                    and (prior_name.startswith("move_")
                         or prior_name == "wait")):
                rejected_at[prior_name] = signature
                action_mask, masked_actions = visible_action_mask(
                    names, view, signature, rejected_at)
            if not any(action_mask):
                outcome = "no legal neural action in visible state"
                print(f"RESULT {outcome}", flush=True)
                break
            action, probs, value = choose(
                model, screen, args.deterministic, action_mask,
                hostile_cells=terminal_hostiles(view))
            name = names[action]
            decisions.append({
                "t": sent, "action": name, "action_index": action,
                "probabilities": probs, "value": value,
                "masked_actions": masked_actions,
                "display_names": names,
                "display_probabilities": probs,
            })
            if name == "travel":
                pending_macro = {
                    "kind": "travel", "stage": "open", "steps": 1,
                    "signature": signature,
                }
                macro_sent_at = loop.time()
            await neural_key(ws, name, view, args.macro_delay)
            last_action_signature = signature
            sent += 1
            last_decision_screen = screen
            last_sent_at = loop.time()
            if sent % 50 == 0:
                p = view.player
                print(f"{sent} neural decisions D:{p.get('depth', 0)} "
                      f"turn={p.get('turn', 0)} hp={p.get('hp', 0)}/"
                      f"{p.get('hp_max', 0)} last={name}", flush=True)

    p = view.player
    TILE_REPLAYS.mkdir(parents=True, exist_ok=True)
    replay = TILE_REPLAYS / f"{run_id}.json"
    replay.write_text(json.dumps({
        "format": "dcss-webtiles-stream-v1",
        "game": run_id,
        "provenance": {
            "agent": "neural-ppo", "variant": args.variant,
            "slot": args.slot, "username": args.username,
            "selection": "argmax" if args.deterministic else "sample",
            "checkpoint": str(checkpoint), "checkpoint_sha256": model_hash,
            "action_names": names,
            "policy_input": "player-visible WebTiles glyph/status screen",
            "adapter_timing": {
                "poll_s": args.poll, "settle_s": args.settle,
                "macro_delay_s": args.macro_delay,
                "retry_s": args.retry,
            },
            "policy_decisions": decisions,
            "forced_ui_acknowledgements": forced_acks,
        },
        "result": {"outcome": outcome, "depth": p.get("depth", 0),
                   "turns": p.get("turn", 0), "actions": sent},
        "events": events,
    }), encoding="utf-8")
    with GAMES.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "agent": "neural-ppo-webtiles", "turns": p.get("turn", 0),
            "xl": p.get("xl", 1), "depth": p.get("depth", 0),
            "death": outcome, "score": p.get("turn", 0), "actions": sent,
            "game": run_id, "source": "webtiles-neural",
            "checkpoint_sha256": model_hash,
        }) + "\n")
    publish_live({
        "running": False, "phase": "complete", "outcome": outcome,
        "depth": p.get("depth", 0), "turn": p.get("turn", 0),
        "hp": p.get("hp", 0), "hp_max": p.get("hp_max", 0),
        "xl": p.get("xl", 1), "actions": sent, "run": run_id,
        "replay": replay.stem, "checkpoint_sha256": model_hash,
        "attempt": args.attempt,
        "best_depth": max(args.best_depth, int(p.get("depth", 0) or 0)),
        "slot": args.slot, "username": args.username,
        "last_action": decisions[-1]["action"] if decisions else None,
        "action_names": (decisions[-1].get("display_names", names)
                         if decisions else names),
        "action_probabilities": (decisions[-1].get(
            "display_probabilities", []) if decisions else []),
        "value": decisions[-1].get("value") if decisions else None,
    })
    print(f"RESULT {outcome}; tile replay: {replay}", flush=True)
    return reached, outcome, replay.stem, int(p.get("depth", 0) or 0)


def main():
    global LIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", choices=["c"], default="c")
    ap.add_argument("--target-depth", type=int, default=5)
    ap.add_argument("--max-actions", type=int, default=1400)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--poll", type=float, default=0.02,
                    help="seconds to wait for the next WebTiles packet")
    ap.add_argument("--settle", type=float, default=0.08,
                    help="quiet time after visible updates before acting")
    ap.add_argument("--macro-delay", type=float, default=0.35,
                    help="UI transition delay between level-map keys")
    ap.add_argument("--retry", type=float, default=3.0,
                    help="seconds before retrying an unchanged visible state")
    ap.add_argument("--stall-timeout", type=float, default=20.0)
    ap.add_argument("--validation-minutes", type=float, default=10.0)
    ap.add_argument("--repeat", action="store_true",
                    help="start fresh neural attempts until target depth")
    ap.add_argument("--fresh", action="store_true",
                    help="archive the disposable bot save before starting")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--username", default="midca")
    ap.add_argument("--password", default="midca")
    ap.add_argument("--register", action="store_true",
                    help="create the local WebTiles account if it is missing")
    ap.add_argument("--live-file", type=Path, default=LIVE)
    args = ap.parse_args()
    if not 0 <= args.slot < 8:
        ap.error("--slot must be between 0 and 7")
    if not re.fullmatch(r"[A-Za-z0-9]+", args.username):
        ap.error("--username must contain only letters and digits")
    LIVE = args.live_file.resolve()
    args.validation_deadline = time.time() + args.validation_minutes * 60
    args.attempt = 0
    args.best_depth = 0
    while True:
        args.attempt += 1
        reached, outcome, replay, depth = asyncio.run(run(args))
        args.best_depth = max(args.best_depth, depth)
        if reached or not args.repeat:
            return 0 if reached else 2
        remaining = max(0, args.validation_deadline - time.time())
        publish_live({
            "running": True,
            "phase": "validating" if remaining else "running",
            "validation_remaining_s": round(remaining),
            "outcome": f"attempt {args.attempt} ended: {outcome}",
            "attempt": args.attempt, "best_depth": args.best_depth,
            "replay": replay,
            "slot": args.slot, "username": args.username,
        })
        print(f"RETRY fresh neural attempt after {outcome}", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())

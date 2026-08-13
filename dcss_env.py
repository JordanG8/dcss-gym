"""
DCSS as a reinforcement-learning environment.

reset() -> obs, step(action_idx) -> (obs, reward, done, info). Nothing in here
knows how to play; it only exposes the game and scores the outcome. The policy
is somewhere else and starts out knowing nothing.

Why this is tractable on a laptop when raw-keystroke RL is not
--------------------------------------------------------------
The action space is DCSS's own MACRO commands, not movement keys. `o` is
auto-explore: one decision, hundreds of game turns. `Tab` is a whole fight.
`X > Enter` is travel across a level. So an episode that reaches D:5 is
~100-200 decisions, not ~10,000 keystrokes. That is a small RL problem.

What the env decides vs what the agent decides
----------------------------------------------
Forced UI prompts (--more--, [y]es/[n]o, the level-up attribute menu) are
answered mechanically here. DCSS offers no *choice* at a --more--; there is one
key that continues and everything else is rejected. Making the policy discover
that is credit-assignment noise, not learning. Every genuine decision - fight,
explore, rest, descend, travel - belongs to the agent.

That line is worth stating in a paper, so it is stated here: the environment
handles keys with no alternative, the policy handles keys with alternatives.
"""
import errno
import fcntl
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import termios
import time
from pathlib import Path

import pyte

CRAWL_DIR = Path("/root/crawl/crawl-ref/source")
PLAY_ROOT = Path("/root/rlplay")
COLS, ROWS = 80, 24

# ── action space ──────────────────────────────────────────────────────────
# Three variants, run side by side, differing ONLY in how equipment is handled.
# Everything else is identical so the comparison means something.
#
#   a  env picks the item   — agent decides "wear something", env opens the
#                             menu and takes the best itself. 10 actions.
#   b  agent picks the item — two-step: open the menu (it becomes the next
#                             observation), read it, then commit with
#                             pick1..3. The only variant that can learn to
#                             tell good gear from bad. 13 actions.
#   c  env does it all      — crawl's autopickup plus a scripted equip after
#                             each pickup. Equipment stops being a decision. 7.
#
# Berserk is in all three: a Minotaur Berserker worships Trog, who grants it
# from XL 1, and it is the single strongest survival tool the character has.
BASE = [
    ("autofight", "\t"),   # attack the nearest monster
    ("explore", "o"),      # auto-explore
    ("rest", "5"),         # rest until healed or interrupted
    ("descend", ">"),      # only legal while standing on down stairs
    ("travel", None),      # level map -> next DOWN staircase -> travel (scripted)
    ("escape", "\x1b"),    # back out of whatever is on screen
    ("berserk", "aa"),     # Trog's Berserk via the ability menu
]
# `None` means the key sequence is computed at runtime by reading the prompt.
PICKUP = ("pickup", None)      # scripted: `,` plus the multi-item menu

# Variant b is TWO-STEP on purpose, and this is the whole point of the
# comparison. Measured 2026-08-13: across 59,955 recorded observations, exactly
# ZERO contained an equip menu — `_equip` opened and closed it inside a single
# action, so the agent never saw the item list. "wear2" therefore meant "the
# second thing in a list you have never observed", and b's three slots could
# not encode a preference about anything. b was not testing choice; it was
# testing a blind index.
#
# Now `open_wear`/`open_wield` open the menu and END THE ACTION, so the menu IS
# the next observation — the policy reads "+2 war axe of flaming" against
# "+0 club" on screen — and `pick1..3` commit to one of them.
#
#   a  env picks   — one action, env opens the menu and takes the best itself
#   b  agent picks — open, LOOK, then choose; the only variant that can learn
#                    to tell good gear from bad
#   c  env does everything, no equipment actions at all
PICKS = [("pick1", None), ("pick2", None), ("pick3", None)]
VARIANTS = {
    "a": BASE + [PICKUP, ("wear", None), ("wield", None)],
    "b": BASE + [PICKUP,
                 ("open_wear", None), ("open_wield", None)] + PICKS,
    "c": BASE,
}

# Default kept for callers that predate variants (smoke tests, tooling).
ACTION_NAMES = [n for n, _ in VARIANTS["a"]]
ACTIONS = [k for _, k in VARIANTS["a"]]
N_ACTIONS = len(ACTIONS)

RE_ITEM_LINE = re.compile(r"^\s*([a-z]) - (.+)$", re.M)
# Items already in use. Selecting the weapon you are holding UNWIELDS it —
# observed in testing: the wield action disarmed the character and left
# "Nothing wielded". Selecting worn armour takes it off. Both are strictly
# harmful, so they are never offered as equip targets.
RE_IN_USE = re.compile(r"\((?:weapon|worn|being worn|in hand)\)", re.I)
RE_GOT = re.compile(r"You now have", re.I)
# Crawl's autopickup announcement: "j - a leather armour". This is what an item
# entering the pack ACTUALLY looks like — "You see here" (RE_ITEM_HERE below)
# only prints when you step onto something autopickup declined to take, which
# since autopickup was enabled is almost never. Message-area use only; the same
# shape is every line of an inventory menu.
RE_PICKED_UP = re.compile(r"^[a-z] - ", re.M)
# NOTE: there used to be RE_WORE / RE_WIELD here, matching "You are now
# wearing ..." to pay the equipment reward. They never fired once. Across
# 28938 replay frames from the 20260813-024545 run, "continue putting on"
# appeared 57 times and "You are now wearing" ZERO times: wearing body armour
# is a multi-turn action, and the completion line has already scrolled out of
# the message area by the time the post-action screen is sampled. `equips` read
# 0 for all 2449 updates of all three variants, which made the whole a/b/c
# equipment comparison signal-free. Equipment is now scored off the AC number
# in the status panel instead — see the ac_gain block in step().
# Crawl announcing loot underfoot. Both phrasings matter: the singular form for
# one item, the list form when several are stacked on the square.
RE_ITEM_HERE = re.compile(r"You see here|Things that are here|"
                          r"There is a stack of", re.I)
# Crawl's actual refusals, verified in-game. "You are too berserk!" is what it
# says when you invoke Berserk while already berserk — not the phrasing I
# guessed, and the penalty silently never fired until this was checked.
RE_EXHAUSTED = re.compile(r"too exhausted|too berserk|already berserk|"
                          r"cannot go berserk|You are exhausted", re.I)
RE_BERSERK_OK = re.compile(r"You go berserk|You feel yourself entering", re.I)

# pyte reports foreground colours by name for the base 16. Anything outside
# that (256-colour escapes) falls back to "w" rather than guessing.
_COLOR_CODE = {
    "default": "w", "black": "k", "red": "r", "green": "g", "brown": "y",
    "blue": "b", "magenta": "m", "cyan": "c", "white": "w",
}

# Shops. Auto-explore and travel walk onto shop tiles, which opens a full-screen
# buying UI that none of the agent's actions dismiss — the game then sits there
# until the stall detector kills the episode. Escaping costs nothing: the agent
# has no money model and buying is not part of the task.
RE_SHOP = re.compile(r"Welcome to [^!]*!|What would you like to (?:buy|do)|"
                     r"Shop|Are you sure you want to leave", re.I)
# Generic full-screen menu we did not anticipate. Being stuck is worse than
# escaping something we might have wanted.
RE_MENU = re.compile(r"\[Esc\]|Esc(?:ape)? to exit|press any key", re.I)

# Combat. DCSS reports damage as prose, not numbers, so these are the verbs it
# actually uses for a landed hit. Minotaurs headbutt, and a hand axe chops and
# cleaves.
RE_HIT = re.compile(
    r"You (?:hit|slash|slice|chop|cleave|carve|bite|claw|punch|headbutt|"
    r"skewer|smack|thump|batter|thrash|shred|maul|gouge|clumsily bash|"
    r"barely scratch|scratch|graze|cut)\b", re.I)
RE_KILL = re.compile(r"You (?:kill|destroy|annihilate|demolish|butcher)\b|"
                     r"is (?:destroyed|annihilated)!", re.I)

# Crawl's own refusals — the game telling us the action made no sense.
RE_NO_TARGET = re.compile(r"No target in view|No reachable target|"
                          r"No monsters in view", re.I)
RE_NO_STAIRS = re.compile(r"can't go down here|can't go up here|"
                          r"Cannot travel|No target found|Sorry, I don't know how", re.I)
RE_NOTHING = re.compile(r"There are no items here|You aren't carrying any|"
                        r"You have nothing to|Okay, then\.", re.I)
# Crawl rejecting the keystroke outright. Applies to ANY action, so it is
# checked separately — but only after the env stopped generating it itself.
RE_UNKNOWN = re.compile(r"Unknown command", re.I)
RE_DONE_EXPLORING = re.compile(r"Done exploring|nothing left to explore|"
                               r"Partly explored", re.I)

# The monster list in the right-hand panel: a glyph, a gap, then a name.
# Presence here means something is visible RIGHT NOW, which is what makes
# resting a mistake rather than a reasonable choice.
RE_MONSTER_ROW = re.compile(r"^(\S)\s{2,}([a-z][a-z ']+)$")

RE_MORE = re.compile(r"--more--", re.I)
RE_YESNO = re.compile(r"\[y\]es\b.*\[n\]o|\(y/n\)", re.I)
# "Keep equipping yourself?" — crawl asks this when a monster INTERRUPTS a
# multi-turn equip (delay.cc:192, EquipOnDelay::try_interrupt). Two traps in
# one prompt, and together they cost the agent its armour:
#
#  1. The text carries no "(y/n)" or "[y]es", so RE_YESNO does not see it and
#     _settle walks straight past.
#  2. It is `yesno(prompt, false, 0, false)` — the `false` is `safe`, which
#     makes crawl demand UPPERCASE. Once _settle finally does react (the
#     rejection message "Uppercase [Y]es or [N]o only, please." happens to
#     match RE_YESNO) it answers lowercase "n", is rejected again, and spins
#     until the 12-iteration budget runs out. The stall detector then kills a
#     perfectly healthy episode.
#
# Answering YES resumes the equip. Answering anything else abandons it — and
# because crawl removes the OLD armour before putting the new one on, abandoning
# leaves the character naked. Measured in the wild: rlb0 went AC 3 -> 2 -> 0,
# stripping its leather armour and then its animal skin over two `wear1`
# actions, and finished the episode unarmoured at AC 0.
RE_KEEP_EQUIP = re.compile(r"Keep equipping yourself", re.I)
# Crawl's complaint when a `safe=false` yesno gets a lowercase answer.
RE_NEED_UPPER = re.compile(r"Uppercase \[Y\]es or \[N\]o only", re.I)
RE_ATTR = re.compile(r"Increase \(S\)trength", re.I)
RE_DEATH = re.compile(r"You die\.\.\.|You have died|Goodbye,", re.I)
RE_HP = re.compile(r"(?:Health|HP):\s*(\d+)/(\d+)")
RE_XL = re.compile(r"XL:\s*(\d+)")
# Armour class, straight off the status panel ("AC:  2"). This is the ground
# truth for "did that equip actually help" — it does not care which action put
# the armour on, how many turns the action took, or whether the message
# scrolled away. NOT anchored to line start: the status panel shares its rows
# with the map, so the real line looks like "  #..  ###.####    AC:  2   Str: 23".
# The leading \s does keep it from matching inside a word.
RE_AC = re.compile(r"(?:^|\s)AC:\s*(\d+)")
# Reward per point of AC gained over the episode's best, and the ceiling on how
# many points one episode may be paid for. See the ac_gain block in step() for
# the budget arithmetic these two numbers come from.
AC_PER_POINT = 1.5
AC_CREDIT_CAP = 10
RE_TIME = re.compile(r"Time:\s*([\d.]+)")

# --- weapons -------------------------------------------------------------
# There is no "AC for weapons" in the status panel, so weapon quality has to be
# scored from the item's NAME — which the panel prints in full, enchantment and
# brand included ("a) +0 hand axe"). No extra keystrokes, same as AC.
#
# Measured first, as the AC bug taught: over 54k replay frames the character
# was holding a NON-AXE in 22.6% of them and a ranged weapon in another 1.5%,
# against 0.3% holding a better axe. The message area showed the mechanism
# outright — "You unwield your +0 hand axe. / c - a +0 club (weapon)". So the
# job here is less "reward good axes" than "stop the wield action handing away
# the good axe it starts with".
#
# The scale is DCSS base damage, but it is deliberately only applied to axes.
AXE_DAMAGE = {
    "executioner's axe": 18,
    "battleaxe": 15,
    "broad axe": 13,
    "war axe": 11,
    "hand axe": 7,
}
# Everything that is not an axe scores BELOW the starting hand axe, whatever
# its damage die says. This is the whole trick and it is not cosmetic: a great
# sword rolls 13 to a war axe's 11, so scoring by raw base damage would teach a
# Berserker — whose trained skill is Axes and who has no Long Blades at all —
# to throw the axe away for a weapon it cannot use. Scoring every non-axe at 4
# means "swap off the axe" is never an upgrade and can never be paid for.
NON_AXE_POWER = 4
# A brand (of flaming, of vorpal, ...) is worth roughly a quarter more damage.
# Flat and small, because on a non-axe it must not add up to beating an axe.
BRAND_BONUS = 2
RE_ENCHANT = re.compile(r"([+-]\d+)")
# Crawl prints a brand TWO ways and both occur in the logs: the status panel
# abbreviates it in parentheses ("+4 dagger (venom)", seen 127 times) while a
# full item name spells it out ("war axe of flaming"). Miss either and branded
# axes score as plain ones.
RE_BRAND = re.compile(r"\((\w+)\)")
RE_BRAND_OF = re.compile(r"\bof\s+\w+")
# The parenthesised form collides with inventory status tags — "(weapon)" and
# "(worn)" mark what is in use and are not brands.
NOT_BRANDS = {"weapon", "worn", "in", "hand", "being"}
# Reward per point of weapon power gained over the episode's best, and the cap.
# Sized UNDER AC's 1.5: the plateau is a survival problem (46% of endings are
# deaths), and armour stops the hit that kills you while a bigger axe only
# shortens the fight. Cap 8 bounds this at +8, so gear as a whole (AC +15,
# weapon +8) tops out at +23 against +35 for a solve. That worst case is
# closer than I would like — realistic runs are nearer +14 — so `ac_gained`
# and `wpn_gained` are logged per update precisely so this can be checked
# against real episodes rather than argued about.
WPN_PER_POINT = 1.0
WPN_CREDIT_CAP = 8
# Anchored on the status panel's "Place:" field. The old pattern matched a bare
# "D:1" ANYWHERE on screen, including feature descriptions like "an escape hatch
# in the floor (D:1)" — which made depth appear to drop after actions that
# cannot change level at all (rest, autofight). 176 apparent depth regressions
# across 12 episodes; only 44 were real.
RE_DEPTH = re.compile(r"Place:\s*\w+:(\d+)")
# The level map's description line for the currently targeted feature.
RE_DOWNSTAIR = re.compile(r"staircase leading down|escape hatch in the floor|"
                          r"stone stairs leading down", re.I)


def weapon_power(desc):
    """Score a weapon from its printed name. Higher is better for THIS character.

    Accepts either form crawl prints: the panel's "+0 hand axe" or an inventory
    line's "a +2 war axe of flaming (weapon)". Unarmed and anything unparseable
    scores 0, which is below every real weapon — being empty-handed should
    never look like an upgrade.
    """
    if not desc:
        return 0
    d = desc.lower()
    base = None
    for nm, dmg in AXE_DAMAGE.items():     # dict is ordered longest-ladder-first
        if nm in d:
            base = dmg
            break
    if base is None:
        # Not an axe. Flat, and below the starting hand axe on purpose.
        return NON_AXE_POWER
    m = RE_ENCHANT.search(d)
    ench = int(m.group(1)) if m else 0
    brand = (any(b.group(1) not in NOT_BRANDS for b in RE_BRAND.finditer(d))
             or bool(RE_BRAND_OF.search(d)))
    return base + ench + (BRAND_BONUS if brand else 0)


def parse_wielded(screen):
    """The wielded item's name, off the status panel. None if not readable.

    The panel prints it as "a) +0 hand axe" while inventory MENUS print
    "a - a +0 club", so the two are already distinguishable by punctuation.
    That is not quite enough on its own: the map is drawn on the same rows and
    ")" is crawl's glyph for a weapon on the floor, so "a) " can occur in the
    map as an ant standing next to a dropped weapon. Anchoring the search to
    the panel's own column — found from the AC label, which is always in it —
    removes that whole class of false read.
    """
    rows = screen.split("\n")[:17]
    col = next((r.index("AC:") for r in rows if "AC:" in r), None)
    if col is None:
        return None
    for row in rows:
        if len(row) <= col:
            continue
        seg = row[col:].strip()
        if len(seg) > 3 and seg[0].isalpha() and seg[1] == ")":
            return seg[2:].strip()
    return None


def parse_status(screen):
    """HP / XL / turn / depth off the status lines. Best effort by design:
    during a menu or the level map the status lines are covered, so callers
    must treat a miss as 'unchanged', never as zero."""
    out = {}
    m = RE_HP.search(screen)
    if m:
        out["hp"], out["hp_max"] = int(m.group(1)), int(m.group(2))
    m = RE_XL.search(screen)
    if m:
        out["xl"] = int(m.group(1))
    m = RE_AC.search(screen)
    if m:
        out["ac"] = int(m.group(1))
    m = RE_TIME.search(screen)
    if m:
        out["turns"] = int(float(m.group(1)))
    m = RE_DEPTH.search(screen)
    if m:
        out["depth"] = int(m.group(1))
    return out


class _Crawl:
    """One crawl process on a pty, rendered to an 80x24 grid by pyte."""

    def __init__(self, play_dir, name, seed=None, autopickup=False):
        # `saves` matters: without it crawl starts, emits its terminal-init
        # sequence, and then blocks forever without drawing anything and
        # without exiting. The process looks perfectly healthy from outside —
        # alive, no error, just ~1900 bytes of setup and then silence.
        for sub in ("", "saves", "morgue", "rcs"):
            (play_dir / sub).mkdir(parents=True, exist_ok=True)

        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ,
                    struct.pack("HHHH", ROWS, COLS, 0, 0))

        cmd = [
            "./crawl",
            "-name", name,
            "-species", "Minotaur",
            "-background", "Berserker",
            "-dir", str(play_dir) + "/",
            "-morgue", str(play_dir / "morgue"),
            "-rcdir", str(play_dir / "rcs"),
            "-extra-opt-first", "weapon=hand axe",
            "-extra-opt-first", "show_more=false",
            "-extra-opt-first", "clear_messages=true",
            # Autofight refuses below this HP fraction and SILENTLY eats the
            # keypress. With 0 it always swings, so action 0 always does
            # something or reports why - either way the agent gets a signal
            # rather than a no-op it cannot distinguish from a bad choice.
            "-extra-opt-first", "autofight_stop=0",
        ]
        # Weapons ()) and armour ([) added to the default autopickup classes.
        # Now enabled for ALL variants, because measurement showed the equip
        # branch was dead: 22.7% of actions were wear/wield attempts and
        # `equips` was 0. (That last inference was WRONG, and worth recording
        # as a warning: `equips` was 0 because its detector was broken, not
        # because the pack was empty. Autopickup is still right to have on —
        # only 4 of 1118 frames had an item underfoot — but it was justified by
        # a number that could not have been anything other than 0.) The variants
        # are supposed to differ only in HOW equipment is chosen, not in whether
        # any exists to choose.
        # MUST be -extra-opt-LAST. Crawl's own defaults call
        # `autopickups.reset()` and rebuild the set (initfile.cc:1615), and that
        # runs AFTER -extra-opt-first — so a value passed "first" is silently
        # overwritten and nothing is ever picked up. Verified: 15 "You see here"
        # messages and zero pickups, not even gold.
        # Classes: $ gold, ? scrolls, ! potions, + books, / wands,
        #          ) weapons, ( missiles, [ armour.
        cmd += ["-extra-opt-last", "autopickup=$?!+/)(["]
        if seed is not None:
            cmd += ["-seed", str(seed)]

        env = dict(os.environ, TERM="xterm-256color",
                   LINES=str(ROWS), COLUMNS=str(COLS))
        self.proc = subprocess.Popen(
            cmd, cwd=str(CRAWL_DIR), stdin=slave, stdout=slave, stderr=slave,
            env=env, start_new_session=True)
        os.close(slave)

        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.ByteStream(self.screen)

    def drain(self, quiet=0.15, timeout=10.0):
        """Read until the game stops drawing for `quiet` seconds.

        The `got_any` guard is load-bearing. Without it, "nothing has arrived
        yet" and "drawing has finished" are the same condition, so a slow start
        returns a BLANK screen that looks like a legitimate observation. The
        policy would then be trained on empty grids and the env would report
        turn 0 forever, with no error anywhere.
        """
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
                        return          # child exited
                    raise
                if not data:
                    return
                self.stream.feed(data)
                got_any = True
                last = time.time()
            elif got_any and time.time() - last >= quiet:
                return
            elif not got_any and not self.alive():
                return

    def text(self):
        return "\n".join(self.screen.display)

    def colors(self):
        """The screen's foreground colours as one char per cell.

        DCSS console encodes a lot in colour that the glyph alone loses — a
        brown `r` and a green `r` are different monsters, `>` in white is a
        staircase you have seen. The POLICY never gets this (its observation
        stays the plain character grid, so RL results remain comparable across
        every run so far); it exists purely so the replay viewer can pick the
        right sprite.

        Uppercase = bold, i.e. the bright half of the 16-colour set.
        """
        buf = self.screen.buffer
        out = []
        for y in range(ROWS):
            row = buf[y]
            line = []
            for x in range(COLS):
                ch = row[x]
                c = _COLOR_CODE.get(ch.fg, "w")
                line.append(c.upper() if ch.bold else c)
            out.append("".join(line))
        return "\n".join(out)

    def send(self, keys):
        try:
            os.write(self.master, keys.encode())
        except OSError:
            pass

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        for sig in (signal.SIGKILL,):
            try:
                os.killpg(os.getpgid(self.proc.pid), sig)
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            os.close(self.master)
        except Exception:
            pass


class DCSSEnv:
    """One independent game slot. Safe to run many in parallel threads: each
    holds its own crawl process, its own -name and its own save directory.

    Sharing either one silently corrupts both games - six workers on the
    default name once produced 36 zero-turn games with no error anywhere.
    """

    def __init__(self, env_id=0, target_depth=5, max_steps=250, seed=None,
                 variant="a"):
        self.env_id = env_id
        self.target_depth = target_depth
        self.max_steps = max_steps
        self.seed = seed
        self.variant = variant
        self.spec = VARIANTS[variant]
        self.action_names = [n for n, _ in self.spec]
        self.n_actions = len(self.spec)
        # Variants must not share save directories — two runs on the same
        # `rl0` silently corrupt each other's games, which is how an earlier
        # 24-env run lost half its episodes.
        self.name = f"rl{variant}{env_id}"
        self.play_dir = PLAY_ROOT / self.name
        self.c = None

    # -- lifecycle ---------------------------------------------------------

    def reset(self):
        if self.c:
            self.c.close()
        # Wipe the save. Crawl resumes an existing character, so without this
        # "episode 2" is really a continuation of episode 1 - the agent would
        # be rewarded for depth it reached under an older policy.
        shutil.rmtree(self.play_dir, ignore_errors=True)

        self.c = _Crawl(self.play_dir, self.name, self.seed, autopickup=True)
        self.c.drain(quiet=0.4, timeout=25)

        self.steps = 0
        self.depth = 1
        self.xl = 1
        self.turns = 0
        self.hp_frac = 1.0
        self.max_depth = 1
        self.stalls = 0
        self.last_turns = -1
        self.last_screen = None
        self.outcome = ""
        self.equips = 0
        # AC baseline. Deliberately read AFTER _settle() below, from the first
        # screen with a readable status panel — a Minotaur Berserker starts in
        # animal skin (AC 2) and must not be paid for gear it was handed at
        # character creation. None means "not read yet": until a real number
        # arrives no gain can be credited, so a covered status line on step 1
        # cannot manufacture a fake +2.
        self.ac = None
        self.max_ac = None
        self.ac_gained = 0
        # Weapon power, same shape as AC: baseline from the first readable
        # panel (the +0 hand axe the Berserker is handed, power 7), credit only
        # above the episode's best.
        self.wpn = None
        self.max_wpn = None
        self.wpn_gained = 0
        self.equip_refused = 0
        # Times the agent picked an item that made it measurably worse off.
        # The number that says whether it is learning to read the menu.
        self.bad_choices = 0
        self.menu_open = None
        self.menu_abandoned = 0
        self.berserks = 0
        self.berserk_wasted = 0
        self.hits = 0
        self.kills = 0
        self.travel_refused = 0
        self.ascents = 0
        self.nonsense = 0
        self._settle()
        scr = self.c.text()
        st = parse_status(scr)
        if "ac" in st:
            self.ac = self.max_ac = st["ac"]
        w = parse_wielded(scr)
        if w is not None:
            self.wpn = self.max_wpn = weapon_power(w)
        return scr

    def color_text(self):
        """Colour grid matching the screen the policy just saw."""
        return self.c.colors() if self.c else ""

    def close(self):
        if self.c:
            self.c.close()
            self.c = None
        shutil.rmtree(self.play_dir, ignore_errors=True)

    # -- forced prompts ----------------------------------------------------

    def _settle(self, budget=12):
        """Clear prompts that offer no choice, so the observation the policy
        sees is always an actual decision point.

        Bounded on purpose. An unbounded 'clear all prompts' loop is how the
        collector once burned 25 minutes per worker on a prompt its keypress
        never dismissed: the process looks healthy and busy the whole time.
        """
        for _ in range(budget):
            scr = self.c.text()
            if not self.c.alive() or RE_DEATH.search(scr):
                return
            if RE_ATTR.search(scr):
                key = "S"           # Minotaur Berserker: strength every time
            elif RE_KEEP_EQUIP.search(scr) or RE_NEED_UPPER.search(scr):
                # Checked BEFORE RE_YESNO: once the uppercase complaint is on
                # screen RE_YESNO also matches, and would answer lowercase and
                # loop forever. Uppercase Y = finish putting the armour on.
                key = "Y"
            elif RE_MORE.search(scr):
                key = "\r"
            elif RE_SHOP.search(scr):
                key = "\x1b"        # walk straight back out of shops
            elif RE_YESNO.search(scr):
                key = "n"           # decline; nothing we want is behind a y/n
            elif RE_MENU.search(scr) and not self._map_like(scr):
                key = "\x1b"        # unrecognised full-screen menu
            else:
                return
            self.c.send(key)
            self.c.drain(quiet=0.1, timeout=4)

    # -- step --------------------------------------------------------------

    @staticmethod
    def monsters_visible(scr):
        """Is anything hostile on screen right now?

        Read from the monster list in the status panel (cols 37+), not from
        message text — messages persist for a turn after the monster is dead.
        """
        # `.strip()`, not `.rstrip()`: the monster entry is padded from column
        # 37 ("      g   hobgoblin"), so anchoring on a non-space at position 0
        # never matched and rest-next-to-a-monster went unpunished.
        for line in scr.split("\n")[10:18]:
            seg = line[37:].strip()
            if seg and RE_MONSTER_ROW.match(seg):
                return True
        return False

    @staticmethod
    def _map_like(scr):
        """True if the screen looks like the dungeon rather than a menu.

        Guards the generic menu-escape: the level map the travel macro opens
        also mentions Esc, and escaping THAT would break the agent's main way
        of finding stairs.
        """
        wall_floor = sum(scr.count(c) for c in "#.")
        return wall_floor > 60

    def _travel(self):
        """Level map -> next down staircase -> travel there. Only if one exists.

        This was a blind `X > Enter`. When no down staircase is known, `>` finds
        nothing, the cursor stays put, and Enter auto-travels to whatever it was
        on — and DCSS auto-travel USES stairs, so the agent climbed back up and
        had to redo the level. Measured: 44 real depth regressions across 12
        episodes, every one preceded by a travel.

        Now the map's description line is checked for a downward feature before
        committing; otherwise back out, which costs a no-op instead of a level.
        """
        self.c.send("X")
        self.c.drain(quiet=0.2, timeout=5)
        seen = set()
        for _ in range(8):
            self.c.send(">")
            self.c.drain(quiet=0.15, timeout=5)
            header = self.c.text().split("\n")[0].strip()
            if RE_DOWNSTAIR.search(header):
                self.c.send("\r")
                self.c.drain()
                return
            # `>` cycles through downward FEATURES, not just staircases — it
            # lands on transporters and hatches too. Keep cycling until a real
            # descent shows up, and stop once the cursor starts repeating.
            if header in seen:
                break
            seen.add(header)
        self.c.send("\x1b")
        self.c.drain(quiet=0.1, timeout=3)
        self.travel_refused += 1

    def _pickup(self):
        """Take what is underfoot, including from the multi-item menu.

        This has to be an explicit action, not autopickup. Crawl's
        `can_autopickup()` returns false whenever `i_feel_safe()` is false —
        i.e. any time a monster is in view (items.cc:3227). A Berserker that
        fights constantly is almost never "safe", so autopickup essentially
        never fires for this agent, which is why the logs were full of
        "You see here ..." and `equips` sat at exactly 0.

        Manual pickup has no such gate.
        """
        self.c.send(",")
        self.c.drain(quiet=0.2, timeout=5)
        scr = self.c.text()
        offered = RE_ITEM_LINE.findall(scr)
        if not offered:
            return                      # single item taken, or nothing here
        # Multi-item menu: select everything on offer, then confirm.
        for ltr, _desc in offered[:6]:
            self.c.send(ltr)
            self.c.drain(quiet=0.1, timeout=3)
        self.c.send("\r")
        self.c.drain()

    def _equip(self, kind, nth=1):
        """Open the wear/wield menu and choose the nth offered item.

        Equipping is two keystrokes — the command, then a letter that depends
        on what is in the pack. Rather than expose raw inventory letters (whose
        meaning changes every game) the env opens the menu, reads the offered
        lines, and sends the requested one. If nothing is offered it escapes,
        which registers as a no-op rather than leaving a menu open for the next
        action to blunder into.

        For WIELD the offered list is now filtered to strict upgrades and sorted
        by weapon power, so `wield1` means "the best weapon in my pack that
        beats what I am holding" in every game. It used to mean "whatever is
        earliest in inventory-letter order", and since the item in use is
        excluded from the menu, the axe the character starts with was never a
        candidate — the action could only ever swap AWAY from it. Measured over
        54k frames: 23.5% of the time the character was holding something worse
        than its starting hand axe, against 0.3% holding something better, with
        "You unwield your +0 hand axe. / c - a +0 club (weapon)" in the log.
        No reward could have fixed that; with one wield action there was no way
        for the policy to express which item it wanted.

        Armour is deliberately NOT filtered this way. AC is a real number in the
        panel, so a bad wear is measured and simply goes unpaid; weapon quality
        has no such ground truth, which is why it needs the ordering.

        THIS IS VARIANT A'S PATH ONLY — the "env picks" control. Variant b now
        goes through _open_menu/_pick instead, where the agent sees the list and
        chooses for itself. Ranking here is not cheating, it is the baseline the
        agent's own choices have to beat.
        """
        self.c.send("W" if kind == "wear" else "w")
        self.c.drain(quiet=0.2, timeout=5)
        menu = self.c.text()
        offered = RE_ITEM_LINE.findall(menu)
        usable = [(ltr, desc) for ltr, desc in offered
                  if not RE_IN_USE.search(desc)]
        if kind == "wield":
            cur = self.wpn if self.wpn is not None else 0
            better = [(weapon_power(desc), ltr) for ltr, desc in usable
                      if weapon_power(desc) > cur]
            better.sort(key=lambda t: -t[0])
            letters = [ltr for _p, ltr in better]
        else:
            letters = [ltr for ltr, _desc in usable]
        if len(letters) >= nth:
            self.c.send(letters[nth - 1])
            self.c.drain()
        elif offered:
            # A menu is open but nothing in it is worth taking — back out so the
            # next action doesn't blunder into it. Counted so the step can be
            # charged as nonsense: with the upgrade filter this is now the
            # common case (asking to wield when carrying no upgrade), and
            # escaping a menu leaves no message for RE_NOTHING to catch.
            self.equip_refused += 1
            self.c.send("\x1b")
            self.c.drain(quiet=0.1, timeout=3)
        else:
            # No menu opened at all (crawl answered "You aren't carrying any
            # armour" and returned to the map). Sending Escape here produces
            # "Unknown command" — which accounted for HALF of all such messages
            # in the logs, and would have been charged to the agent as if it
            # had chosen badly. Send nothing.
            pass

    def _open_menu(self, kind):
        """Open the wear/wield menu and LEAVE IT OPEN, so it becomes the next
        observation. This is the half of the two-step protocol that gives the
        agent something to recognise — without it the item list existed only
        inside a single action and was never once seen (0 of 59,955 frames).

        Returns True if a menu is actually up. Crawl answers "You aren't
        carrying any armour" and stays on the map when the pack has nothing, in
        which case there is nothing to look at and nothing to pick.
        """
        self.c.send("W" if kind == "wear" else "w")
        self.c.drain(quiet=0.2, timeout=5)
        if RE_ITEM_LINE.search(self.c.text()):
            self.menu_open = kind
            return True
        self.menu_open = None
        self.equip_refused += 1
        return False

    def _pick(self, nth):
        """Commit to the nth item in the OPEN menu, in the order it is drawn.

        Order is the menu's own, not a ranking: the agent is choosing from what
        it can see, so re-sorting behind its back would put the decision back in
        the env's hands — which is the thing this protocol exists to stop.

        Returns (chosen_description, ok). ok is False when no menu was open, so
        the caller can charge a wasted action.
        """
        if not self.menu_open:
            return None, False
        offered = RE_ITEM_LINE.findall(self.c.text())
        usable = [(l, d) for l, d in offered if not RE_IN_USE.search(d)]
        if len(usable) < nth:
            # Asked for slot 3 of a two-item menu. Back out rather than leave
            # the menu up for the next action to blunder into.
            self.equip_refused += 1
            self.c.send("\x1b")
            self.c.drain(quiet=0.1, timeout=3)
            self.menu_open = None
            return None, False
        ltr, desc = usable[nth - 1]
        self.c.send(ltr)
        self.c.drain()
        self.menu_open = None
        return desc, True

    def _close_menu(self):
        """Escape a menu the agent left open by doing something else.

        Counted: opening a menu and then walking away from it is the whole cost
        of looking, with none of the benefit. Unpriced, "open, glance, leave" is
        free information plus a free turn of safety, which is exactly the loop
        variant b collapsed into.
        """
        if self.menu_open:
            self.menu_abandoned += 1
            self.c.send("\x1b")
            self.c.drain(quiet=0.1, timeout=3)
            self.menu_open = None

    def step(self, action):
        """Apply one macro-action. Returns (obs, reward, done, info)."""
        prev_xl, prev_turns, prev_depth = self.xl, self.turns, self.depth
        prev_refused = self.travel_refused
        prev_ac, prev_wpn = self.ac, self.wpn
        prev_abandoned = self.menu_abandoned
        chose = None                      # description the agent committed to
        bad_pick = False
        name, keys = self.spec[action]

        # Anything that is not a pick abandons an open menu. Escaping first
        # keeps the keystroke from being eaten by the menu, which is how the
        # agent used to "blunder into" a list it could not see.
        if not name.startswith("pick"):
            self._close_menu()

        if name == "travel":
            self._travel()
        elif name == "pickup":
            self._pickup()
        elif name.startswith("open_"):
            self._open_menu(name.split("_", 1)[1])
        elif name.startswith("pick"):
            chose, ok = self._pick(int(name[-1]))
            if not ok:
                bad_pick = True           # picked from a menu that wasn't there
        elif keys is None:
            kind = "wear" if name.startswith("wear") else "wield"
            nth = int(name[-1]) if name[-1].isdigit() else 1
            self._equip(kind, nth)
        else:
            self.c.send(keys)
            self.c.drain()

        # Variant c never decides about equipment, so the env does it: grab
        # what is underfoot and put it on. Without this, c can never improve
        # its gear at all and the comparison would be unfair rather than
        # informative.
        # Take whatever we are standing on, whichever action put us here.
        #
        # Crawl's autopickup is gated on `i_feel_safe()`, so it never fires for
        # a Berserker in near-permanent combat. And leaving it to the agent's
        # own `pickup` action does not work either: measured over 1118 live
        # frames there were only FOUR moments with an item underfoot, so the
        # policy would have to choose pickup on exactly those steps by chance.
        # Gear would effectively never enter the pack, and the whole equipment
        # branch — including variant b's reason to exist — stays dead.
        #
        # So the env collects; the agent still decides what to WEAR, which is
        # the difference the variants are actually testing.
        #
        # The trigger below used to be RE_ITEM_HERE alone, and that was a THIRD
        # dead branch of the same family as the equips bug: "You see here ..."
        # matched 0 times in 28,938 frames. Autopickup — turned on for all
        # variants precisely to get gear into the pack — takes the item
        # silently, so the message the trigger waits for can no longer be
        # printed. One fix disabled another. Variant c, whose ONLY route to
        # equipment is this block, therefore had ac_gained = 0 for the whole
        # run: it was walking D:1-5 in its starting animal skin by construction.
        #
        # Now it also fires on the autopickup line crawl prints when something
        # enters the pack ("j - a leather armour"), which is the event that
        # actually happens. Checked against `_map_like` so an open menu — where
        # the same "letter - item" shape is everywhere — cannot trigger it.
        scr_now = self.c.text()
        msg_now = "\n".join(scr_now.split("\n")[17:])
        picked = bool(RE_GOT.search(msg_now) or RE_PICKED_UP.search(msg_now))
        if RE_ITEM_HERE.search(scr_now):
            self._pickup()
            picked = picked or bool(RE_GOT.search(self.c.text()))
        if self.variant == "c" and picked and self._map_like(self.c.text()):
            self._equip("wear")
            self._equip("wield")

        self._settle()
        self.steps += 1

        scr = self.c.text()
        st = parse_status(scr)
        # Only overwrite what we actually read. A covered status line means
        # "unknown", and treating unknown as 0 would hand out a large fake
        # negative reward every time a menu happens to be open.
        self.depth = st.get("depth", self.depth)
        self.xl = st.get("xl", self.xl)
        self.turns = st.get("turns", self.turns)
        if "hp" in st:
            self.hp_frac = st["hp"] / max(1, st["hp_max"])

        # AC. Credited against a HIGH-WATER MARK, never a raw delta, for the
        # same reason depth is: a raw delta is farmable. Armour can come off as
        # well as on, so `wear X, unwear X, wear X` would be an income stream —
        # and variant b has three wear slots to cycle. Against max_ac the second
        # wearing pays nothing. It also makes AC lost to corrosion or a
        # swapped-off piece cost nothing, which is correct: the penalty for
        # being unarmoured is already paid in HP by the fights that follow.
        ac_gain = 0
        if "ac" in st:
            self.ac = st["ac"]
            if self.max_ac is None:
                self.max_ac = self.ac      # first readable panel = the baseline
            elif self.ac > self.max_ac:
                ac_gain = min(self.ac - self.max_ac, AC_CREDIT_CAP - self.ac_gained)
                self.max_ac = self.ac

        # Weapon power, identical treatment. Same high-water rule, same reason:
        # a raw delta would pay for the swap-away-swap-back cycle that the wield
        # action was already performing 23.5% of the time.
        wpn_gain = 0
        held = parse_wielded(scr)
        if held is not None:
            self.wpn = weapon_power(held)
            if self.max_wpn is None:
                self.max_wpn = self.wpn
            elif self.wpn > self.max_wpn:
                wpn_gain = min(self.wpn - self.max_wpn,
                               WPN_CREDIT_CAP - self.wpn_gained)
                self.max_wpn = self.wpn

        died = bool(RE_DEATH.search(scr)) or not self.c.alive()
        # Losing a level costs no reward (newly_deep is measured against
        # max_depth) but silently burns the step budget, which is the dominant
        # way episodes end. Count it so it is visible rather than invisible.
        if "depth" in st and self.depth < prev_depth:
            self.ascents += 1
        newly_deep = max(0, self.depth - self.max_depth)
        self.max_depth = max(self.max_depth, self.depth)

        # Did this action actually cost the game a turn? Only meaningful when
        # the status line was readable; a covered status line means unknown.
        turns_seen = "turns" in st
        noop = turns_seen and self.turns == prev_turns

        # --- reward ---
        # v3. Both earlier versions failed the same way: a DENSE shaping term
        # outweighed the SPARSE objective, so the policy optimised the shaping.
        #   v1: +1.0*delta_hp punished every fight (winning still costs HP), so
        #       standing still dominated. explore 22%->3%, escape 14%->32%.
        #   v2: over-corrected. The no-op penalty was worth 0.05*300 = 15 points
        #       per episode while all of D:1->D:5 was worth 4*2.5 = 10. So
        #       "always press a key that does something" beat "go down", and it
        #       learned exactly that: autofight 16%->45%, escape ->0%, with
        #       return improving from -9.79 to -8.56 while depth FELL 2.10->1.70.
        #
        # The invariant this version keeps, and the one to check first if it
        # fails again: the worst-case total shaping cost of an episode must be
        # comfortably SMALLER than the reward for solving it.
        #   worst-case shaping: 300 * (0.01 + 0.02) = -9
        #   solving D:1->D:5   : 4 * 5.0 + 15       = +35
        reward = 0.0
        reward += 5.0 * newly_deep                  # the objective, and it must dominate
        reward += 0.40 * max(0, self.xl - prev_xl)  # proxy for "able to go deeper"
        # Halved when the episode budget went 500 -> 1000 steps, so the total
        # time cost of an episode is UNCHANGED (1000 x 0.005 == 500 x 0.01).
        # Without this the per-step costs would have quietly doubled relative to
        # the +35 for solving, which is the exact ratio error that broke reward
        # v2. The nonsense penalty is deliberately NOT halved: it is charged per
        # mistake, not per step, and a good policy makes few.
        reward -= 0.005                             # time is not free
        if noop:
            reward -= 0.01                          # a key that does nothing is mildly worse

        # --- nonsense actions -------------------------------------------------
        # Specific, game-confirmed mistakes rather than a blanket no-op tax.
        # That distinction matters: reward v2's blanket tax taught the policy to
        # spam whichever key always consumed a turn, because "did something" was
        # all it measured. Here each penalty names a real error, and `rest`
        # beside a monster is punished even though it DOES consume turns.
        #
        # Budget: at 30% nonsense over a 500-step episode this is 150 * 0.08 =
        # -12, against +35 for reaching D:5. Depth still dominates ~3x. If the
        # nonsense rate stops falling while depth stalls, this is the first
        # number to cut.
        msg_area = "\n".join(scr.split("\n")[17:])
        near = self.monsters_visible(scr)
        # NOT charged here, deliberately: `equip_refused`. Asking to wield when
        # you carry no upgrade is a wasted keypress, and the -0.01 no-op cost
        # already covers it. Charging the full nonsense -0.08 as well was tried
        # for exactly 5 updates and measured at **302 refusals per 3 successful
        # equips** — a 100:1 ratio, so gearing up would have netted about -8
        # against the +1.5 the gain pays, and the policy would have learned to
        # never touch equipment. That is the same failure as the bug this run
        # exists to fix, pointing the other way: a penalty that fires so often
        # it drowns the behaviour you are trying to establish.
        #
        # It stays a logged counter. Turn the charge back on (and size it from
        # the measured ratio, not from taste) once ac_gained/wpn_gained are
        # reliably non-zero and the refusal spam is what is left to trim.
        nonsense = (
            (name == "autofight" and RE_NO_TARGET.search(msg_area))
            or (name == "rest" and near)
            or (name == "descend" and RE_NO_STAIRS.search(msg_area))
            or (name == "travel" and self.travel_refused > prev_refused)
            or (name.startswith(("wear", "wield", "pickup"))
                and RE_NOTHING.search(msg_area))
            or (name == "explore" and RE_DONE_EXPLORING.search(msg_area))
            # A key crawl refuses outright, whatever the action was. Safe to
            # charge now that the env's own equip macro no longer emits a
            # stray Escape (which caused ~half of these).
            or RE_UNKNOWN.search(msg_area)
        )
        if nonsense:
            reward -= 0.08
            self.nonsense += 1

        # Equipment, scored by RESULT rather than by the attempt. The old
        # version paid +1.5 for the message "You are now wearing ...", which
        # never once appeared in 28938 frames (see the RE_WORE note at the top
        # of this file) — so this branch was dead for the entire run and the
        # a/b/c comparison measured nothing.
        #
        # Paying for AC instead is also the better signal on its own terms: it
        # ignores wearing something worse than what is already on, and it still
        # fires when armour finishes going on several turns after the action
        # that started it, which is the normal case for body armour.
        #
        # Sizing. AC_PER_POINT is large on purpose — this is the survival lever
        # and the plateau is a survival problem. Budget check, same form as the
        # other terms: D:1-5 loot realistically takes a Berserker from AC 2 to
        # about AC 10, so the cap of 10 credited points bounds this at
        # 10 * 1.5 = +15 against +35 for a solve. Big enough to be worth a
        # detour to the item on the floor, not big enough that standing on D:1
        # in good armour beats descending — which is the exact trap the kill
        # reward had to be cut to 0.15 to avoid.
        if ac_gain > 0:
            reward += AC_PER_POINT * ac_gain
            self.ac_gained += ac_gain
            self.equips += 1
        if wpn_gain > 0:
            reward += WPN_PER_POINT * wpn_gain
            self.wpn_gained += wpn_gain
            self.equips += 1

        # --- choosing BADLY costs ------------------------------------------
        # The gains above are against a high-water mark, which by itself makes a
        # downgrade merely unpaid — free. That is not enough now that the agent
        # picks its own items: "wear the worse armour" has to be a mistake it
        # can feel, or there is nothing to learn from seeing the list.
        #
        # Charged on the DROP, at the same rate the gain pays. That leaves the
        # off-then-on cycle net negative (pay +N, charge -N, regain pays 0
        # because of the high-water mark) so churning gear is never free, while
        # a genuine upgrade that briefly dips through a lower AC still comes out
        # ahead. Only charged when the agent itself chose — variant a's env
        # picks and variant c's autoequip must not be billed for the env's
        # decisions.
        # Dropping the high-water mark to the new level when a loss is charged
        # is what keeps this from killing exploration. Without it the agent pays
        # -3 to downgrade and earns nothing for putting the good axe back on
        # (the old mark still stands), so a single bad pick is permanent and the
        # safest policy is to never open a menu at all — the same trap that made
        # `equip_refused` unchargeable. Resetting the mark makes a round trip net
        # exactly zero: you are refunded precisely what you were charged, never
        # more, so churning gear still cannot be farmed for profit.
        if chose is not None:
            if prev_ac is not None and self.ac is not None and self.ac < prev_ac:
                reward -= AC_PER_POINT * (prev_ac - self.ac)
                self.max_ac = self.ac
                self.ac_gained = max(0, self.ac_gained - (prev_ac - self.ac))
                self.bad_choices += 1
            if prev_wpn is not None and self.wpn is not None and self.wpn < prev_wpn:
                reward -= WPN_PER_POINT * (prev_wpn - self.wpn)
                self.max_wpn = self.wpn
                self.wpn_gained = max(0, self.wpn_gained - (prev_wpn - self.wpn))
                self.bad_choices += 1
        if bad_pick:
            # Committing to a slot with no menu open, or slot 3 of a two-item
            # list. Cheap — it is a protocol slip, not a bad judgement.
            reward -= 0.08
            self.nonsense += 1
        if self.menu_abandoned > prev_abandoned:
            # Opened a menu, then did something else. Priced at the nonsense
            # rate rather than higher: looking and declining is a LEGITIMATE
            # move — it is what the agent should do when the pack holds nothing
            # better — so this must stay cheap enough that checking is
            # affordable. It only has to beat free, because free is what turned
            # menu-opening into a hiding place.
            reward -= 0.08
            self.nonsense += 1
        if not ac_gain and not wpn_gain and name == "pickup" and RE_GOT.search(scr):
            reward += 0.3

        # Combat. Crawl never prints monster HP, so damage is inferred from the
        # verbs it uses for a landed blow. Counted only in the message area
        # (rows 17+) — the status panel and level map would produce false hits.
        #
        # Sizing matters more than the idea. Hits are DENSE and depth is SPARSE,
        # which is the exact shape that broke rewards v1 and v2. Budget check:
        # a combat-heavy 500-step episode lands maybe 200 blows = +4.0, against
        # +35 for reaching D:5. So fighting pays, but camping D:1 to farm
        # kobolds never beats descending.
        msg = "\n".join(scr.split("\n")[17:])
        hits = min(5, len(RE_HIT.findall(msg)))
        kills = min(3, len(RE_KILL.findall(msg)))
        if hits:
            reward += 0.02 * hits
            self.hits += hits
        if kills:
            # Measured, not guessed: at 0.5/kill a 220-step stay on D:1 earned
            # +13.03 (25 kills), against +35 for solving D:1->D:5. Extrapolated
            # to 500 steps that is ~+27 for never descending at all — farming
            # would have beaten the objective. 0.15 keeps it a trickle.
            reward += 0.15 * kills
            self.kills += kills

        # Berserk. No bonus for using it — its payoff is winning the fight it
        # was used for, which depth already rewards. But spamming it into the
        # exhaustion cooldown is a real mistake a good player never makes, so
        # that is charged for explicitly.
        if name == "berserk":
            if RE_EXHAUSTED.search(scr):
                reward -= 0.3
                self.berserk_wasted += 1
            elif RE_BERSERK_OK.search(scr):
                self.berserks += 1

        done = False
        if died:
            # v4: -0.5 -> -3.0. Under v3 the policy plateaued for 80 updates at
            # ~3.0 mean depth / ~10% solve, and the failure mode had flipped:
            # step-limit endings fell 42->33 per 60 episodes while DEATHS rose
            # 20->28, with `rest` down at 5%. Dying was nearly free, so it
            # fought everything and never healed.
            #
            # This does NOT reintroduce v1's failure. v1 died of a DENSE
            # per-step HP term (+k*delta_hp), which made every individual fight
            # locally negative and passivity strictly optimal. A terminal
            # penalty is paid once per episode and leaves fighting profitable
            # whenever it leads downward.
            #
            # Invariant still holds: worst-case shaping 500*(0.01+0.02) + 3.0
            # = -18 against +35 for a solve.
            reward -= 3.0
            self.outcome = "died"
            done = True
        elif self.max_depth >= self.target_depth:
            reward += 10.0
            self.outcome = f"reached D:{self.max_depth}"
            done = True
        elif self.steps >= self.max_steps:
            self.outcome = "step limit"
            done = True

        # A wedged game is not a teaching signal, it is a broken episode. End
        # it rather than feeding the policy hundreds of identical states.
        #
        # Only counted on steps where the turn counter was actually READABLE.
        # The level map and every menu cover the status line, so an unparsed
        # status reads as a frozen turn counter during perfectly normal play —
        # travel-heavy episodes were being killed at step 34 for making
        # progress. Unknown is neither progress nor a stall.
        # `turns_seen or self.menu_open` — the second half closes a blind spot
        # that the two-step equip protocol turned into a wireheading exploit.
        #
        # The exemption for an unreadable turn counter is correct in general (a
        # covered status line means "unknown", and travel-heavy episodes were
        # being killed at step 34 for making progress). But an OPEN MENU also
        # covers the status line, and menus cost no game time, expose the
        # character to no danger, and — because of this exemption — could not
        # trip the stall detector. Variant b found all three: at u83 it was
        # spending 69.7% of its actions opening menus and 0% picking from them,
        # ending at D:1 with turns/action as low as 0.07. Hiding in a menu was
        # strictly cheaper than playing.
        #
        # When we KNOW a menu is up, an unchanged screen is a stall, not an
        # unknown. That is the one case where the exemption must not apply.
        if turns_seen or self.menu_open:
            if (noop or self.menu_open) and scr == self.last_screen:
                self.stalls += 1
            else:
                self.stalls = 0
            self.last_turns = self.turns
        self.last_screen = scr
        if self.stalls >= 20 and not done:
            # Stalling must COST something. Without this it is an escape hatch:
            # variant a collapsed to explore-only (entropy 0.009, 60/60 episodes
            # stalled at D:1) because ending after ~26 steps dodges the step and
            # nonsense costs that a real 500-step attempt has to pay. Priced
            # like death so wedging yourself is never the cheap option.
            reward -= 3.0
            self.outcome = "stalled"
            done = True

        info = {"depth": self.depth, "max_depth": self.max_depth, "xl": self.xl,
                "turns": self.turns, "hp_frac": round(self.hp_frac, 3),
                "steps": self.steps, "outcome": self.outcome,
                "action": name, "equips": self.equips,
                "ac": self.ac, "ac_gained": self.ac_gained,
                "wpn": self.wpn, "wpn_gained": self.wpn_gained,
                "bad_choices": self.bad_choices, "menu": self.menu_open,
                "equip_refused": self.equip_refused,
                "berserks": self.berserks, "berserk_wasted": self.berserk_wasted,
                "hits": self.hits, "kills": self.kills,
                "nonsense": self.nonsense, "ascents": self.ascents}
        return scr, reward, done, info

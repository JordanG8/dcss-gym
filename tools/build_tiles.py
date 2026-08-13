"""Build a glyph -> DCSS sprite atlas so replays can render as real tiles.

DCSS ships the webtiles art as a few big PNGs plus generated `tileinfo-*.js`
index files. Those give us two things: a NAME -> index table, and an index ->
{sx, sy, ex, ey} rectangle inside the sheet. This script joins them and emits
one small JSON the dashboard can use to blit sprites onto a canvas.

The hard part is that a pty gives us CHARACTERS, not tile ids. Webtiles gets
exact sprite numbers from the game itself; we are reverse-mapping the console
view. Terrain, stairs, doors and the player are unambiguous. Monsters are not:
`r` is some rodent, and only the colour narrows it down. So the mapping is keyed
on (glyph, colour) with a glyph-only fallback, and anything unresolved is drawn
as text over the floor tile rather than guessed wrong.

    python build_tiles.py            # build
    python build_tiles.py --find RAT # search available tile names
"""
import argparse
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
STATIC = Path("/root/crawl/crawl-ref/source/webserver/game_data/static")
OUT_DIR = HERE / "data" / "tiles"
SHEETS = ["floor", "wall", "feat", "main", "player"]

RE_INC = re.compile(r"^exports\.([A-Z0-9_]+) = val\+\+;", re.M)
RE_ALIAS = re.compile(r"^val = exports\.([A-Z0-9_]+) = exports\.([A-Z0-9_]+);", re.M)
RE_COORD = re.compile(
    r"\{w: (\d+), h: (\d+), ox: (-?\d+), oy: (-?\d+), "
    r"sx: (\d+), sy: (\d+), ex: (\d+), ey: (\d+)\}")


def parse_sheet(sheet):
    """-> (names {NAME: idx}, coords [ {..} ])  with idx already sheet-relative."""
    text = (STATIC / f"tileinfo-{sheet}.js").read_text(errors="replace")

    names, val = {}, 0
    for line in text.splitlines():
        line = line.strip()
        m = RE_INC.match(line)
        if m:
            names[m.group(1)] = val
            val += 1
            continue
        m = RE_ALIAS.match(line)
        if m and m.group(2) in names:
            names[m.group(1)] = names[m.group(2)]
            val = names[m.group(2)] + 1

    coords = [{"w": int(a), "h": int(b), "ox": int(c), "oy": int(d),
               "sx": int(e), "sy": int(f), "ex": int(g), "ey": int(h)}
              for a, b, c, d, e, f, g, h in RE_COORD.findall(text)]

    # Values are emitted relative to a per-sheet base (main.js starts at
    # TILE_FEAT_MAX, for instance). Normalising by the minimum makes every
    # sheet 0-based, which is what the coords array is indexed by.
    if names:
        base = min(names.values())
        names = {k: v - base for k, v in names.items()}
    return names, coords


# (glyph, colour) -> candidate tile names, best first. Colour codes come from
# dcss_env.colors(): lowercase = normal, UPPERCASE = bold/bright.
# Only the first name that actually exists in the sheets is used.
MAP = {
    # --- terrain: exact, no guessing needed ---
    ("#", None): ["WALL_NORMAL", "WALL_BRICK_DARK_1", "DNGN_ROCK_WALL"],
    (".", None): ["FLOOR_NORMAL", "FLOOR_GREY_DIRT"],
    (",", None): ["FLOOR_NORMAL"],
    (">", None): ["DNGN_STONE_STAIRS_DOWN", "DNGN_STONE_STAIRS_DOWN_1"],
    ("<", None): ["DNGN_STONE_STAIRS_UP", "DNGN_STONE_STAIRS_UP_1"],
    ("+", None): ["DNGN_CLOSED_DOOR", "DNGN_GATE_CLOSED"],
    ("'", None): ["DNGN_OPEN_DOOR"],
    ("\\", None): ["DNGN_PORTAL", "DNGN_ENTER"],
    ("_", None): ["DNGN_ALTAR", "DNGN_UNSEEN"],
    ("^", None): ["DNGN_TRAP_BOLT", "DNGN_TRAP_SPEAR", "DNGN_TRAP_ALARM"],
    ("~", None): ["DNGN_DEEP_WATER", "DNGN_SHALLOW_WATER"],
    ("≈", None): ["DNGN_SHALLOW_WATER"],
    # --- the player ---
    ("@", None): ["BASE_MINOTAUR", "MONS_MINOTAUR"],
    # --- items ---
    ("$", None): ["GOLD01", "GOLD02"],
    (")", None): ["WPN_HAND_AXE", "WPN_DAGGER"],
    ("[", None): ["ARM_LEATHER_ARMOUR", "ARM_ROBE"],
    ("?", None): ["SCR_SCROLL", "SCROLL"],
    ("!", None): ["POTION_OFFSET_1", "POTION_OFFSET", "UNSEEN_POTION"],
    ("%", None): ["FOOD_RATION", "CORPSE"],
    ("/", None): ["WAND_OFFSET_1", "WAND_OFFSET", "UNSEEN_WAND"],
    ("=", None): ["RING_NORMAL_OFFSET_1", "RING_NORMAL_OFFSET"],
    ('"', None): ["AMU_NORMAL_OFFSET_1", "AMU_NORMAL_OFFSET",
                  "UNRAND_AMULET_INVISIBILITY"],
    ("+b", None): ["BOOK_PAPER", "BOOK"],
}

# Monsters: glyph + colour. DCSS console players read these the same way.
MONSTERS = {
    ("r", "y"): "MONS_RAT", ("r", "b"): "MONS_QUOKKA",
    ("r", "g"): "MONS_RIVER_RAT", ("r", "w"): "MONS_RAT",
    ("j", "w"): "MONS_JACKAL", ("j", "y"): "MONS_JACKAL",
    ("g", "y"): "MONS_GNOLL", ("g", "b"): "MONS_GOBLIN",
    ("k", "r"): "MONS_KOBOLD", ("k", "y"): "MONS_KOBOLD",
    ("o", "y"): "MONS_ORC", ("o", "r"): "MONS_ORC_WARRIOR",
    ("b", "w"): "MONS_BAT", ("b", "y"): "MONS_BAT",
    ("S", "g"): "MONS_ADDER", ("s", "w"): "MONS_GIANT_COCKROACH",
    ("w", "w"): "MONS_WORM", ("F", "w"): "MONS_FUNGUS",
    ("f", "w"): "MONS_FUNGUS", ("Z", "w"): "MONS_ZOMBIE_SMALL",
    ("h", "y"): "MONS_HOUND", ("i", "w"): "MONS_HOMUNCULUS",
    ("e", "g"): "MONS_ELF", ("K", "y"): "MONS_KOBOLD",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find", help="search tile names containing this")
    args = ap.parse_args()

    sheets = {}
    for s in SHEETS:
        names, coords = parse_sheet(s)
        sheets[s] = (names, coords)
        print(f"{s:7s} names={len(names):6d} coords={len(coords):6d}")

    if args.find:
        needle = args.find.upper()
        for s, (names, _) in sheets.items():
            hits = [n for n in names if needle in n][:15]
            if hits:
                print(f"  {s}: {', '.join(hits)}")
        return 0

    def resolve(candidates):
        """First candidate that exists, as (sheet, rect)."""
        for name in candidates:
            for s, (names, coords) in sheets.items():
                idx = names.get(name)
                if idx is not None and 0 <= idx < len(coords):
                    c = coords[idx]
                    return {"sheet": s, "x": c["sx"], "y": c["sy"],
                            "w": c["ex"] - c["sx"], "h": c["ey"] - c["sy"]}
        return None

    atlas, missing = {}, []
    for (glyph, colour), cands in MAP.items():
        r = resolve(cands)
        key = glyph if colour is None else f"{glyph}\t{colour}"
        if r:
            atlas[key] = r
        else:
            missing.append((key, cands[0]))

    for (glyph, colour), name in MONSTERS.items():
        r = resolve([name])
        if r:
            atlas[f"{glyph}\t{colour}"] = r
        else:
            missing.append((f"{glyph}/{colour}", name))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    used = sorted({v["sheet"] for v in atlas.values()})
    for s in used:
        src = STATIC / f"{s}.png"
        if src.exists():
            shutil.copy2(src, OUT_DIR / f"{s}.png")

    (OUT_DIR / "atlas.json").write_text(json.dumps({
        "tile": 32, "sheets": used, "map": atlas,
    }), encoding="utf-8")

    print(f"\nresolved {len(atlas)} entries across sheets {used}")
    print(f"copied {len(used)} png(s) -> {OUT_DIR}")
    if missing:
        print(f"\nUNRESOLVED ({len(missing)}) — these fall back to text:")
        for k, n in missing:
            print(f"   {k!r:14s} wanted {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

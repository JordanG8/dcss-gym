"""Deterministic curriculum exercises for the DCSS action interface.

These are deliberately *player-view* fixtures, not hidden-server snapshots.
They test whether a policy can read the same map/status/menu that a player can
see.  Native fixed-seed Sprint runs belong beside these micro-exercises once a
skill is reliable; keeping this fast layer makes every parser and UI regression
cheap to catch first.
"""
from dataclasses import dataclass
from typing import Iterable

from dcss_contract import make_public_observation
from dcss_env import VARIANTS


ROWS, COLS = 24, 80


def _screen(*lines):
    """Build a stable 80x24 terminal fixture from meaningful visible lines."""
    rows = [line[:COLS].ljust(COLS) for line in lines]
    return "\n".join((rows + [" " * COLS] * ROWS)[:ROWS])


def _base_lines(hp="20/20", monster="", message=""):
    return [
        "###############............................  rlgym the Chopper",
        "#.............#............................  Minotaur of Trog",
        "#......@......#............................  Health: " + hp,
        "#.............#............................  Magic: 2/2",
        "###############............................  AC: 2    Str: 23",
        "                                               EV: 11",
        "                                               XL: 1 Next: 0%  Place: Dungeon:1",
        "                                               Time: 10.0 (1.0)",
        "                                               a) +0 hand axe",
        "",
        "                                               " + monster,
        "",
        "",
        "",
        "",
        "",
        "",
        message,
    ]


@dataclass(frozen=True)
class GymScenario:
    name: str
    skill: str
    observation: object
    expected_action: str
    note: str


def _observation(screen, action_names, mask, menu_kind=None, candidates=()):
    return make_public_observation(
        screen=screen,
        colors="w" * (ROWS * COLS),
        status={"hp": 20, "hp_max": 20, "ac": 2, "xl": 1, "depth": 1},
        menu_kind=menu_kind,
        menu_candidates=candidates,
        action_names=action_names,
        action_mask=mask,
    )


def scenarios(variant="b") -> Iterable[GymScenario]:
    """Small, deterministic public-view curriculum for one action variant."""
    names = tuple(name for name, _key in VARIANTS[variant])
    all_legal = [True] * len(names)

    enemy_lines = _base_lines(monster="g   goblin", message="A goblin comes into view.")
    # The hostile is deliberately on the map, adjacent to @.  The message is
    # supporting player-visible context, not a hidden label the policy gets.
    enemy_lines[2] = "#......@g.....#............................  Health: 20/20"
    enemy = _screen(*enemy_lines)
    yield GymScenario(
        "enemy_visible", "enemy recognition",
        _observation(enemy, names, all_legal), "autofight",
        "A visible hostile should be handled before exploration.")

    clear = _screen(*_base_lines(message=""))
    yield GymScenario(
        "no_enemy_visible", "enemy recognition",
        _observation(clear, names, all_legal), "explore",
        "Without a monster, continuing exploration is the productive default.")

    wounded = _screen(*_base_lines(hp="4/20", message="You are badly wounded."))
    yield GymScenario(
        "low_hp_safe_square", "survival",
        _observation(wounded, names, all_legal), "rest",
        "Low health with no visible enemy calls for recovery, not blind combat.")

    if variant != "b":
        return

    menu_mask = [name == "escape" for name in names]
    for pick in ("pick1", "pick2"):
        menu_mask[names.index(pick)] = True
    weapon = _screen(
        "Wield which item? (- for none, * to show all)",
        "a - a +0 club", "b - a +2 war axe", "", "[Esc] cancel")
    yield GymScenario(
        "prefer_better_axe", "equipment",
        _observation(weapon, names, menu_mask, "wield",
                     ("a +0 club", "a +2 war axe")),
        "pick2", "The second visible candidate is the better trained weapon.")

    armour = _screen(
        "Wear which item? (- for none, * to show all)",
        "a - a +2 chain mail", "b - a +0 robe", "", "[Esc] cancel")
    yield GymScenario(
        "prefer_more_armour", "equipment",
        _observation(armour, names, menu_mask, "wear",
                     ("a +2 chain mail", "a +0 robe")),
        "pick1", "The first visible candidate is the armour upgrade.")

    decline_mask = [name == "escape" for name in names]
    bad_weapon = _screen(
        "Wield which item? (- for none, * to show all)",
        "a - a +0 club", "", "[Esc] cancel")
    yield GymScenario(
        "decline_downgrade", "equipment",
        _observation(bad_weapon, names, decline_mask, "wield", ("a +0 club",)),
        "escape", "No offered weapon improves on the visible hand axe.")


def by_name(name, variant="b"):
    for scenario in scenarios(variant):
        if scenario.name == name:
            return scenario
    raise KeyError(name)

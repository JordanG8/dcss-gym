"""Versioned, player-legal boundary between DCSS and learning code.

The terminal is still the transport and the glyph grid remains a valid model
input.  This module makes the information boundary explicit so a future model
can consume structured features without accidentally inheriting omniscient
server state or PTY/UI implementation details.
"""
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


OBSERVATION_VERSION = 1


@dataclass(frozen=True)
class PublicObservation:
    """Information a normal player could inspect on the current screen."""

    version: int
    screen: str
    colors: str
    status: Mapping[str, int]
    menu_kind: str | None
    menu_candidates: tuple[str, ...]
    action_names: tuple[str, ...]
    action_mask: tuple[bool, ...]

    def as_dict(self):
        """JSON-safe form used by replays, Gym tests, and viewer tooling."""
        out = asdict(self)
        out["menu_candidates"] = list(self.menu_candidates)
        out["action_names"] = list(self.action_names)
        out["action_mask"] = list(self.action_mask)
        return out


def make_public_observation(*, screen: str, colors: str,
                            status: Mapping[str, int], menu_kind: str | None,
                            menu_candidates: Sequence[str],
                            action_names: Sequence[str],
                            action_mask: Sequence[bool]) -> PublicObservation:
    """Construct a public observation and enforce its core invariant."""
    if len(action_names) != len(action_mask):
        raise ValueError("action_names and action_mask must have equal length")
    if not any(action_mask):
        raise ValueError("a public observation must expose one legal action")
    return PublicObservation(
        version=OBSERVATION_VERSION,
        screen=screen,
        colors=colors,
        status=dict(status),
        menu_kind=menu_kind,
        menu_candidates=tuple(menu_candidates),
        action_names=tuple(action_names),
        action_mask=tuple(bool(v) for v in action_mask),
    )

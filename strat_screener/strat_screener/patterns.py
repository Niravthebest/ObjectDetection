"""TheStrat combo-pattern detection.

Patterns are detected on the *last completed* bar of a type sequence,
looking back at most 3 bars:

  2-1-2 continuation   2u-1-2u  or  2d-1-2d
  2-1-2 reversal       2u-1-2d  or  2d-1-2u
  3-1-2                3-1-2u   or  3-1-2d
  1-2-2 continuation   1-2u-2u  or  1-2d-2d
  2-2 reversal         2u-2d    or  2d-2u   (a.k.a. "rev strat")

Only the patterns above are recognized; any other sequence returns None
(e.g. a lone inside bar '1' is a pending setup, not yet actionable).
"""
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class PatternMatch:
    name: str
    direction: str  # 'up' or 'down'
    n_bars: int      # how many trailing bars make up the setup (2 or 3)


def _dir(t: Optional[str]) -> Optional[str]:
    if t == "2u":
        return "up"
    if t == "2d":
        return "down"
    return None


def detect_pattern(types: Sequence[Optional[str]], idx: int) -> Optional[PatternMatch]:
    """Detect a pattern ending at position `idx` in `types`.

    `types` is any sequence of '1'/'2u'/'2d'/'3'/None values (e.g. from
    `classify_types(df).values`). Returns the matched PatternMatch or None.
    """
    if idx < 1 or idx >= len(types):
        return None

    t0 = types[idx]
    t1 = types[idx - 1]
    t2 = types[idx - 2] if idx >= 2 else None
    d0 = _dir(t0)

    if d0 is not None and idx >= 2:
        d2 = _dir(t2)
        if t1 == "1" and d2 is not None:
            name = "2-1-2 continuation" if d0 == d2 else "2-1-2 reversal"
            return PatternMatch(name, d0, 3)
        if t2 == "3" and t1 == "1":
            return PatternMatch("3-1-2", d0, 3)
        d1 = _dir(t1)
        if t2 == "1" and d1 is not None and d0 == d1:
            return PatternMatch("1-2-2 continuation", d0, 3)

    d1 = _dir(t1)
    if d0 is not None and d1 is not None and d0 != d1:
        return PatternMatch("2-2 reversal", d0, 2)

    return None

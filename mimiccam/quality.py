# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""The maximum-quality profile for the paid engine, and its price.

One module says what "maximum quality" means and what it costs, because a mode
that quietly bills twice for the same picture is worse than no mode at all.

The picture is identical on both of Decart's realtime tiers: fast mode
(``speed: "fast"``) serves the session from a higher-compute tier for lower
latency and bills 2x. So the best *picture* is the standard tier, and this
profile is that tier with nothing else turned down - the model's native
1280x720 at 30 fps, the whole-person prompt carrying the character's own look
notes, and idle pausing off, because pausing ends the session and puts the raw
camera back on screen until you move: a picture change, not just a bill change.
"""

from __future__ import annotations

from . import credits

MODEL = "lucy-2.5"
WIDTH = 1280
HEIGHT = 720
FPS = 30
TIER = "standard"

USD_PER_MINUTE = round(credits.USD_PER_SECOND * 60, 2)
FAST_USD_PER_MINUTE = round(credits.FAST_USD_PER_SECOND * 60, 2)


def per_minute(fast_mode: bool = False) -> float:
    """What one minute of a connected session costs at the chosen tier."""
    return FAST_USD_PER_MINUTE if fast_mode else USD_PER_MINUTE


def rate_text(fast_mode: bool = False) -> str:
    return f"{credits.money(per_minute(fast_mode))}/min"


def minutes_for(
    balance_usd: float, spent_usd: float = 0.0, fast_mode: bool = False
) -> float:
    """How many minutes a balance buys, at the rate the session is billed at."""
    rate = per_minute(fast_mode)
    left = credits.remaining_usd(balance_usd, spent_usd)
    if rate <= 0 or left <= 0:
        return 0.0
    return left / rate


def minutes_text(minutes: float) -> str:
    """Whole minutes; below one there is no honest number to round to."""
    if minutes <= 0:
        return "no minutes left"
    if minutes < 1:
        return "under a minute left"
    return f"{int(minutes)} min left"


def summary() -> str:
    """One line for the log and the README: what runs, and what it bills."""
    return (
        f"Lucy 2.5, {TIER} tier, {WIDTH}x{HEIGHT} at {FPS} fps - "
        f"{credits.money(per_minute())}/min ({credits.money(per_minute() / 60)}/s)"
    )


def receipt(balance_usd: float = 0.0, spent_usd: float = 0.0) -> str:
    """The arithmetic shown before a maximum-quality session starts.

    Deliberately blunt: the price per minute, what a 30-minute call comes to,
    the fact that the meter follows connected time, and that fast mode is a
    latency purchase rather than a picture upgrade.
    """
    lines = [
        f"Lucy 2.5 re-renders the whole frame - {WIDTH}x{HEIGHT} at {FPS} fps, "
        f"{TIER} tier, the tier with the best picture.",
        f"{rate_text()} = {credits.money(per_minute() * 60)} an hour. "
        f"A 30-minute call is {credits.money(per_minute() * 30)}.",
        "The meter runs for every second the session stays connected. "
        "Stop camera ends it.",
        f"Fast mode is {rate_text(True)} for lower latency - Decart states the "
        "picture is unchanged, so it is not a quality upgrade.",
    ]
    if balance_usd > 0:
        minutes = int(minutes_for(balance_usd, spent_usd))
        lines.append(
            f"Your balance of {credits.money(balance_usd)} is about {minutes} "
            "minutes at this rate." if minutes >= 1 else
            f"Your balance of {credits.money(balance_usd)} is already spent - "
            "nothing is left at this rate."
        )
    else:
        lines.append(
            "No balance set: enter the figure from your dashboard in Settings "
            "and this box counts it down while you stream."
        )
    return "\n".join(lines)

"""Credit maths for the paid engine.

Decart bills realtime video per second of active generation and publishes no
balance endpoint - not on the REST API, not in the realtime protocol, not in the
Python SDK. So MimicCam does the arithmetic itself: seconds come from the
server's own ``generation_tick`` messages, dollars from the published rate, and
the balance is a figure the user types in from their dashboard once. This module
holds that arithmetic in one place so the UI and the engine cannot disagree.
"""

from __future__ import annotations

# https://docs.platform.decart.ai/getting-started/pricing
USD_PER_SECOND = 0.02
FAST_USD_PER_SECOND = 0.04


def usd_per_second(fast_mode: bool) -> float:
    """What one second of generation costs at the configured speed."""
    return FAST_USD_PER_SECOND if fast_mode else USD_PER_SECOND


def usd_from_seconds(seconds: float, fast_mode: bool) -> float:
    return round(max(0.0, float(seconds)) * usd_per_second(fast_mode), 3)


def remaining_usd(balance_usd: float, spent_usd: float) -> float:
    """What is left of the balance the user typed in, never below zero."""
    return max(0.0, round(float(balance_usd) - float(spent_usd), 3))


def money(usd: float) -> str:
    """Two decimals once there is a dollar to show, three while it is pocket change."""
    value = abs(float(usd))
    return f"${value:.2f}" if value >= 0.01 else f"${value:.3f}"


def clock(seconds: float) -> str:
    """Billed seconds as m:ss, which is how people say it out loud."""
    total = int(max(0.0, float(seconds)))
    return f"{total // 60}:{total % 60:02d}"

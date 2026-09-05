"""
devig_pinnacle.py

No-vig probability conversion for the CFB scoring engine.

Sportsbook prices carry a margin (the "vig" or "overround"): the implied
probabilities of the two sides sum to more than 1. To recover the market's
*true* probability estimate we have to strip that margin back out. Pinnacle is
the sharpest widely-available book, so its no-vig line is the closest thing to
a fair-value consensus you can get cheaply — which is why the model compares
against it.

This module supplies:

    devig(home_odds, away_odds, method=...)  -> DevigResult
    compute_edge(model_prob, market_prob)    -> float

Supported de-vig methods (two-outcome markets):

    "multiplicative"  proportional normalization (simplest, slight favorite bias)
    "additive"        subtract the margin equally from each side
    "power"           odds-ratio / logarithmic method (fit an exponent)
    "shin"            Shin (1992) model — accounts for informed money, the
                      method that best matches Pinnacle's closing lines in most
                      public studies. This is the model's default.

All methods take *American* odds (e.g. -280 / +230) and return probabilities
that sum to exactly 1.
"""

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "DevigResult",
    "american_to_implied_prob",
    "devig",
    "compute_edge",
    "DEVIG_METHODS",
]

DEVIG_METHODS = ("multiplicative", "additive", "power", "shin")

# Numerical solver tolerances for the iterative methods (shin, power).
_SOLVER_TOL = 1e-12
_SOLVER_MAX_ITER = 200


@dataclass
class DevigResult:
    """Result of stripping the vig from a two-outcome market."""
    home_prob: float          # no-vig probability, home side
    away_prob: float          # no-vig probability, away side
    overround: float          # raw booksum - 1.0 (the margin, e.g. 0.045 = 4.5%)
    method: str               # which de-vig method produced these
    home_prob_raw: float      # implied prob before de-vigging (with margin)
    away_prob_raw: float
    z: float = 0.0            # Shin insider-money proportion (0 for other methods)

    def __post_init__(self) -> None:
        s = self.home_prob + self.away_prob
        if not math.isclose(s, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"de-vigged probabilities must sum to 1.0, got {s:.8f} "
                f"(method={self.method})"
            )


# ---------------------------------------------------------------------------
# American odds -> implied probability (still carries the vig)
# ---------------------------------------------------------------------------

def american_to_implied_prob(american_odds: float) -> float:
    """
    Convert American odds to the implied probability of that outcome. The
    result still includes the book's margin; it is *not* de-vigged.

    -150  ->  150 / (150 + 100) = 0.600
    +150  ->  100 / (150 + 100) = 0.400
    """
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return -american_odds / (-american_odds + 100.0)


# ---------------------------------------------------------------------------
# Individual de-vig methods (operate on raw implied probabilities)
# ---------------------------------------------------------------------------

def _devig_multiplicative(q_home: float, q_away: float) -> Tuple[float, float, float]:
    """Proportional normalization. Fast, but slightly over-weights favorites."""
    s = q_home + q_away
    return q_home / s, q_away / s, 0.0


def _devig_additive(q_home: float, q_away: float) -> Tuple[float, float, float]:
    """
    Subtract the margin equally from each side. Can produce a negative
    probability on very lopsided books, so we clamp to [0, 1] and renormalize
    as a safety net.
    """
    margin = (q_home + q_away) - 1.0
    p_home = q_home - margin / 2.0
    p_away = q_away - margin / 2.0
    # Safety net for extreme books where equal subtraction goes negative.
    if p_home < 0 or p_away < 0:
        p_home = max(p_home, 0.0)
        p_away = max(p_away, 0.0)
        s = p_home + p_away
        if s == 0:
            return 0.5, 0.5, 0.0
        return p_home / s, p_away / s, 0.0
    return p_home, p_away, 0.0


def _devig_power(q_home: float, q_away: float) -> Tuple[float, float, float]:
    """
    Odds-ratio / logarithmic ("power") method: find exponent k such that
    q_home**(1/k) + q_away**(1/k) == 1. Solved by bisection on k.
    """
    def booksum(k: float) -> float:
        return q_home ** (1.0 / k) + q_away ** (1.0 / k)

    # k = 1 gives the raw booksum (> 1). Increasing k pushes each prob toward 1
    # (raising a value in (0,1) to a smaller power), so the sum grows; we need
    # k < 1 to shrink the sum down to 1.
    lo, hi = 1e-6, 1.0
    for _ in range(_SOLVER_MAX_ITER):
        mid = 0.5 * (lo + hi)
        s = booksum(mid)
        if abs(s - 1.0) < _SOLVER_TOL:
            break
        if s > 1.0:
            hi = mid
        else:
            lo = mid
    k = 0.5 * (lo + hi)
    p_home = q_home ** (1.0 / k)
    p_away = q_away ** (1.0 / k)
    s = p_home + p_away
    return p_home / s, p_away / s, 0.0


def _shin_recovered(q: float, s: float, z: float) -> float:
    """
    Shin's recovered true probability for a single outcome, given the outcome's
    raw implied prob q, the booksum s = sum of raw implied probs, and the
    insider-money proportion z.
    """
    inside = z * z + 4.0 * (1.0 - z) * q * q / s
    return (math.sqrt(inside) - z) / (2.0 * (1.0 - z))


def _devig_shin(q_home: float, q_away: float) -> Tuple[float, float, float]:
    """
    Shin (1992) model. Solves for z in [0, 1) — the proportion of money from
    informed ("insider") bettors — such that the two recovered probabilities
    sum to 1. Returns (p_home, p_away, z).

    f(z) = sum of recovered probs - 1 is monotonically decreasing on [0, 1),
    positive at z=0 and negative as z->1, so plain bisection is robust here.
    """
    s = q_home + q_away
    if s <= 1.0:
        # No margin (or a negative one) — nothing to de-vig via Shin; fall back
        # to proportional normalization.
        return _devig_multiplicative(q_home, q_away)

    def f(z: float) -> float:
        return _shin_recovered(q_home, s, z) + _shin_recovered(q_away, s, z) - 1.0

    lo, hi = 0.0, 1.0 - 1e-9
    flo = f(lo)
    if flo <= 0:
        # Degenerate: already sums to <= 1 at z=0. Normalize.
        return _devig_multiplicative(q_home, q_away)

    for _ in range(_SOLVER_MAX_ITER):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < _SOLVER_TOL:
            break
        if fmid > 0:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    p_home = _shin_recovered(q_home, s, z)
    p_away = _shin_recovered(q_away, s, z)
    # Guard against tiny numerical drift so the sum is exactly 1.
    total = p_home + p_away
    return p_home / total, p_away / total, z


_DISPATCH = {
    "multiplicative": _devig_multiplicative,
    "additive": _devig_additive,
    "power": _devig_power,
    "shin": _devig_shin,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def devig(
    home_odds: float,
    away_odds: float,
    method: str = "shin",
) -> DevigResult:
    """
    Strip the vig from a two-outcome (home/away) market quoted in American odds
    and return no-vig probabilities.

    Parameters
    ----------
    home_odds, away_odds : American odds for each side (e.g. -280, +230).
    method : one of DEVIG_METHODS. Defaults to "shin".

    Returns
    -------
    DevigResult with home_prob + away_prob == 1.0 and the raw overround.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"unknown de-vig method {method!r}; choose from {DEVIG_METHODS}"
        )

    q_home = american_to_implied_prob(home_odds)
    q_away = american_to_implied_prob(away_odds)
    overround = (q_home + q_away) - 1.0

    p_home, p_away, z = _DISPATCH[method](q_home, q_away)

    return DevigResult(
        home_prob=p_home,
        away_prob=p_away,
        overround=overround,
        method=method,
        home_prob_raw=q_home,
        away_prob_raw=q_away,
        z=z,
    )


def compute_edge(model_prob: float, market_prob: float) -> float:
    """
    Edge = how much more likely the model thinks an outcome is than the no-vig
    market does. Positive means the model sees value on that side.

        edge = model_prob - market_prob

    Expressed on the probability scale (0.045 == 4.5 percentage points).
    """
    return model_prob - market_prob


if __name__ == "__main__":
    # Quick demonstration across methods for a -280 / +230 market.
    print("Market: home -280 / away +230")
    q_h = american_to_implied_prob(-280)
    q_a = american_to_implied_prob(230)
    print(f"Raw implied: home {q_h:.4f}  away {q_a:.4f}  "
          f"booksum {q_h + q_a:.4f}  overround {(q_h + q_a - 1):.4f}\n")
    for m in DEVIG_METHODS:
        r = devig(-280, 230, method=m)
        extra = f"  (z={r.z:.4f})" if m == "shin" else ""
        print(f"{m:>14}: home {r.home_prob:.4f}  away {r.away_prob:.4f}{extra}")

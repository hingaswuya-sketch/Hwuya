"""
devig_pinnacle.py

Market-side helpers for the CFB scoring engine. Converts Pinnacle's
two-way American moneyline into a *no-vig* (fair) probability estimate and
computes the model-vs-market edge.

Pinnacle is used as the sharp reference book: its lines are close to the
true market consensus, so removing the bookmaker margin ("the vig" /
"overround") from Pinnacle's price gives the best readily-available
estimate of the real win probability to compare a model against.

Two de-vig methods are provided:

    "multiplicative"  -- proportional normalization; divide each raw implied
                         probability by the booksum. Simple and unbiased for
                         balanced markets, but it distributes the margin
                         evenly and therefore slightly overstates favorites.

    "shin"            -- Shin's (1992/1993) model, which attributes part of
                         the margin to informed ("insider") money. It shifts
                         relatively more of the margin onto the favorite,
                         which empirically de-vigs sharp two-way markets
                         better than proportional normalization. This is the
                         default the CFB engine requests.

Only two-outcome (home/away) markets are handled here, which is all the CFB
moneyline pipeline needs. Shin's model has a clean closed form for n=2.
"""

import math
from dataclasses import dataclass
from typing import Literal

DevigMethod = Literal["multiplicative", "shin"]


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds (total return per unit stake).

    +150 -> 2.50, -200 -> 1.50. Zero is not a valid American price.
    """
    if odds == 0:
        raise ValueError("American odds of 0 are undefined")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to the raw (vig-inclusive) implied probability.

    This is simply 1 / decimal_odds and will, when summed across both sides
    of a two-way market, exceed 1.0 by the bookmaker's margin.
    """
    return 1.0 / american_to_decimal(odds)


# ---------------------------------------------------------------------------
# De-vig result
# ---------------------------------------------------------------------------

@dataclass
class DevigResult:
    """No-vig probabilities for a two-way market.

    Attributes
    ----------
    home_prob, away_prob:
        Fair (no-vig) win probabilities. Guaranteed to sum to 1.0.
    overround:
        The bookmaker margin expressed as booksum minus 1.0. For example a
        raw implied-probability sum of 1.045 yields ``overround == 0.045``
        (i.e. a 4.5% margin). Always >= 0 for a real two-way price.
    method:
        Which de-vig method produced ``home_prob`` / ``away_prob``.
    """
    home_prob: float
    away_prob: float
    overround: float
    method: str


# ---------------------------------------------------------------------------
# De-vig methods
# ---------------------------------------------------------------------------

def _devig_multiplicative(raw_home: float, raw_away: float) -> tuple[float, float]:
    """Proportional normalization: divide each side by the booksum."""
    booksum = raw_home + raw_away
    return raw_home / booksum, raw_away / booksum


def _shin_z(raw_home: float, raw_away: float) -> float:
    """Solve for Shin's insider-money proportion ``z`` for a two-way market.

    ``z`` is the fraction of bettors assumed to be trading on inside
    information. It is the value in [0, 1) for which the Shin-adjusted
    probabilities sum to exactly 1. The mapping from z to the probability
    sum is monotonic, so a bisection converges reliably.
    """
    booksum = raw_home + raw_away
    if booksum <= 1.0:
        # No margin to remove (or an arbitrage price); Shin's z is 0.
        return 0.0

    def prob_sum(z: float) -> float:
        return _shin_prob(raw_home, booksum, z) + _shin_prob(raw_away, booksum, z)

    # prob_sum is monotonically decreasing in z from booksum (z=0) toward 1.
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if prob_sum(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _shin_prob(raw_prob: float, booksum: float, z: float) -> float:
    """Shin-adjusted fair probability for a single outcome given ``z``."""
    inside = z * z + 4.0 * (1.0 - z) * (raw_prob * raw_prob) / booksum
    return (math.sqrt(inside) - z) / (2.0 * (1.0 - z))


def _devig_shin(raw_home: float, raw_away: float) -> tuple[float, float]:
    """Shin's model de-vig for a two-way market."""
    booksum = raw_home + raw_away
    z = _shin_z(raw_home, raw_away)
    home = _shin_prob(raw_home, booksum, z)
    away = _shin_prob(raw_away, booksum, z)
    # Renormalize to defend against tiny numerical drift from the solver.
    total = home + away
    return home / total, away / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def devig(
    home_odds: float,
    away_odds: float,
    method: DevigMethod = "shin",
) -> DevigResult:
    """Remove the bookmaker margin from a two-way American moneyline.

    Parameters
    ----------
    home_odds, away_odds:
        American odds for the two sides (e.g. -280 and +230).
    method:
        "shin" (default) or "multiplicative". See the module docstring.

    Returns
    -------
    DevigResult with fair probabilities summing to 1.0 and the overround.
    """
    raw_home = american_to_implied_prob(home_odds)
    raw_away = american_to_implied_prob(away_odds)
    overround = (raw_home + raw_away) - 1.0

    if method == "multiplicative":
        home_prob, away_prob = _devig_multiplicative(raw_home, raw_away)
    elif method == "shin":
        home_prob, away_prob = _devig_shin(raw_home, raw_away)
    else:
        raise ValueError(f"unknown devig method: {method!r}")

    return DevigResult(
        home_prob=home_prob,
        away_prob=away_prob,
        overround=overround,
        method=method,
    )


def compute_edge(model_prob: float, market_prob: float) -> float:
    """Edge as the model's probability advantage over the fair market price.

    Positive means the model thinks the outcome is more likely than the
    no-vig market implies (a potential value bet); negative means the market
    is higher than the model. Both inputs must be no-vig probabilities on the
    same outcome for the comparison to be meaningful.
    """
    return model_prob - market_prob


if __name__ == "__main__":
    # Smoke test: Pinnacle-style -280 / +230 two-way price.
    for m in ("multiplicative", "shin"):
        r = devig(-280, 230, method=m)
        print(
            f"{m:>14}: home {r.home_prob:.4f}  away {r.away_prob:.4f}  "
            f"sum {r.home_prob + r.away_prob:.6f}  overround {r.overround:.4f}"
        )

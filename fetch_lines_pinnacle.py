"""
fetch_lines_pinnacle.py

Market-side data source for the CFB scoring engine: pulls Pinnacle moneyline
(head-to-head) odds for college football games and hands them to the model as
plain American-odds numbers.

Pinnacle's own public betting API was retired, so the practical way to get
Pinnacle prices programmatically is through an odds aggregator. This module
targets The Odds API (https://the-odds-api.com), which exposes Pinnacle as a
bookmaker key and speaks a stable JSON schema. It is written against stdlib
only (urllib) so the package has no third-party runtime dependency.

Usage
-----
    export ODDS_API_KEY=your_key_here
    python fetch_lines_pinnacle.py

    # or from code:
    from fetch_lines_pinnacle import fetch_pinnacle_lines
    lines = fetch_pinnacle_lines()          # list[GameLine]

Offline / testing
-----------------
The network call and the JSON parsing are deliberately split. `parse_odds_api`
turns a raw decoded payload into GameLine objects with no I/O, so it can be
unit-tested against a saved fixture, and `load_lines_from_file` reads a saved
response from disk.
"""

import json
import os
import urllib.parse
import urllib.request
import urllib.error
import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Dict

logger = logging.getLogger("fetch_lines_pinnacle")

__all__ = [
    "GameLine",
    "fetch_pinnacle_lines",
    "parse_odds_api",
    "load_lines_from_file",
    "OddsApiError",
    "decimal_to_american",
]

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT_KEY = "americanfootball_ncaaf"
PINNACLE_KEY = "pinnacle"


class OddsApiError(RuntimeError):
    """Raised when the odds provider request fails or returns no usable data."""


@dataclass
class GameLine:
    """A single game's Pinnacle moneyline, normalized for the model."""
    home_team: str
    away_team: str
    home_odds: int           # American odds
    away_odds: int           # American odds
    commence_time: str       # ISO-8601 UTC kickoff, as provided by the source
    bookmaker: str = PINNACLE_KEY
    sport_key: str = DEFAULT_SPORT_KEY


def decimal_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds (what The Odds API returns) to American odds, rounded
    to the nearest integer.

        2.50 -> +150
        1.40 -> -250
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100.0)
    return round(-100.0 / (decimal_odds - 1.0))


def parse_odds_api(payload: Any) -> List[GameLine]:
    """
    Convert a decoded The Odds API `/odds` payload into GameLine objects,
    keeping only the Pinnacle h2h (moneyline) market. Pure function, no I/O.

    Games where Pinnacle has no h2h market, or where either side's price is
    missing, are skipped with a warning rather than raising.
    """
    if not isinstance(payload, list):
        raise OddsApiError(
            f"expected a list of games from the odds provider, got "
            f"{type(payload).__name__}"
        )

    lines: List[GameLine] = []
    for game in payload:
        home = game.get("home_team")
        away = game.get("away_team")
        if not home or not away:
            logger.warning("skipping game with missing team names: %r", game.get("id"))
            continue

        pinnacle = next(
            (b for b in game.get("bookmakers", []) if b.get("key") == PINNACLE_KEY),
            None,
        )
        if pinnacle is None:
            logger.info("no Pinnacle line for %s vs %s; skipping", away, home)
            continue

        h2h = next(
            (m for m in pinnacle.get("markets", []) if m.get("key") == "h2h"),
            None,
        )
        if h2h is None:
            logger.info("Pinnacle has no h2h market for %s vs %s; skipping", away, home)
            continue

        prices: Dict[str, float] = {
            o.get("name"): o.get("price")
            for o in h2h.get("outcomes", [])
            if o.get("name") is not None and o.get("price") is not None
        }
        if home not in prices or away not in prices:
            logger.warning(
                "Pinnacle h2h missing a side for %s vs %s (have %s); skipping",
                away, home, list(prices),
            )
            continue

        try:
            home_am = decimal_to_american(float(prices[home]))
            away_am = decimal_to_american(float(prices[away]))
        except (ValueError, TypeError) as exc:
            logger.warning("bad odds for %s vs %s: %s; skipping", away, home, exc)
            continue

        lines.append(
            GameLine(
                home_team=home,
                away_team=away,
                home_odds=home_am,
                away_odds=away_am,
                commence_time=game.get("commence_time", ""),
                sport_key=game.get("sport_key", DEFAULT_SPORT_KEY),
            )
        )

    return lines


def load_lines_from_file(path: str) -> List[GameLine]:
    """Parse a previously-saved The Odds API response from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return parse_odds_api(payload)


def fetch_pinnacle_lines(
    api_key: Optional[str] = None,
    sport_key: str = DEFAULT_SPORT_KEY,
    regions: str = "us,eu",
    timeout: float = 15.0,
) -> List[GameLine]:
    """
    Fetch current Pinnacle CFB moneylines from The Odds API.

    The API key is read from the `api_key` argument or, if omitted, the
    `ODDS_API_KEY` environment variable. Raises OddsApiError on any failure so
    the caller can decide whether to fall back to a cached fixture.

    Pinnacle lives in the EU region on The Odds API, so `regions` includes "eu"
    by default; keeping "us" as well lets the same call double as a sanity check
    against US books if you widen the parse later.
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise OddsApiError(
            "no API key: pass api_key= or set the ODDS_API_KEY environment "
            "variable (get one free at https://the-odds-api.com)"
        )

    params = urllib.parse.urlencode(
        {
            "apiKey": key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": PINNACLE_KEY,
        }
    )
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds?{params}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                logger.info("Odds API requests remaining: %s", remaining)
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        raise OddsApiError(f"Odds API HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise OddsApiError(f"Odds API request failed: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OddsApiError(f"Odds API returned invalid JSON: {exc}") from exc

    lines = parse_odds_api(payload)
    logger.info("parsed %d Pinnacle CFB moneylines", len(lines))
    return lines


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        games = fetch_pinnacle_lines()
    except OddsApiError as err:
        print(f"Could not fetch live lines: {err}")
        raise SystemExit(1)

    if not games:
        print("No Pinnacle CFB moneylines available right now.")
    for g in games:
        print(
            f"{g.away_team} ({g.away_odds:+d}) @ {g.home_team} ({g.home_odds:+d})"
            f"   {g.commence_time}"
        )

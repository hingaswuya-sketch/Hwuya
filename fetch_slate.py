#!/usr/bin/env python3
"""Build a real games.json for edge_finder.py from live sources.

Pulls the college-football schedule + market spreads from ESPN's public
scoreboard API and joins your SP+ ratings table, producing the exact
``{config, games:[...]}`` shape ``edge_finder.py`` consumes -- so no matchup
or line is ever typed by hand.

    python fetch_slate.py --date 2026-09-05 --ratings sp_ratings.json --out games.json
    python fetch_slate.py --saturday --ratings sp_ratings.json | python edge_finder.py scan --data -

Sources
-------
* Schedule + odds : ESPN CFB scoreboard (no key). Spread is read from the
  preferred sportsbook's ``details`` string (default DraftKings, falls back to
  whatever ESPN returns, and records which book was used).
* SP+ ratings     : a local file you maintain -- JSON ``{"Duke": 20.5, ...}``
  or CSV ``team,rating`` rows. Teams are matched on ESPN's abbreviation,
  location, and display names (case/punctuation-insensitive).

Games missing a market line, or an SP+ rating for either side, are written with
``"_unrated": true`` / ``"_noline": true`` and no model input, and counted in
the summary -- they simply don't score until you fill the gap. Completed games
are marked ``played: true`` so the scanner filters them out.

NOTE: some sandboxes block outbound network access; run this where
``site.api.espn.com`` is reachable.  It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/scoreboard?dates={date}&groups={groups}&limit=200"
)


# --------------------------------------------------------------------------- #
# Ratings table
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    """Normalize a team name for matching: lowercase, alnum only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_ratings(path: Path) -> dict[str, float]:
    """Load SP+ ratings from JSON ({name: rating}) or CSV (team,rating)."""
    text = path.read_text(encoding="utf-8")
    table: dict[str, float] = {}
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
        for name, rating in raw.items():
            if name.startswith("_") or not isinstance(rating, (int, float)):
                continue  # skip comments / metadata
            table[_norm(name)] = float(rating)
    else:
        reader = csv.reader(text.splitlines())
        for row in reader:
            if len(row) < 2:
                continue
            name, rating = row[0].strip(), row[1].strip()
            try:
                table[_norm(name)] = float(rating)
            except ValueError:
                continue  # header or junk row
    if not table:
        raise SystemExit(f"error: no ratings parsed from {path}")
    return table


def lookup_rating(team: dict, table: dict[str, float]) -> Optional[float]:
    """Try several ESPN name fields against the ratings table."""
    for key in ("abbreviation", "location", "shortDisplayName", "displayName", "name"):
        val = team.get(key)
        if val and _norm(val) in table:
            return table[_norm(val)]
    return None


# --------------------------------------------------------------------------- #
# ESPN fetch + parse
# --------------------------------------------------------------------------- #
def fetch_scoreboard(date: str, groups: int) -> dict:
    url = ESPN_URL.format(date=date, groups=groups)
    req = urllib.request.Request(url, headers={"User-Agent": "edge-finder/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise SystemExit(
            f"error: could not fetch ESPN scoreboard ({exc}).\n"
            f"       If egress is blocked here, run this where {url.split('/')[2]} "
            f"is reachable."
        )


_SPREAD_RE = re.compile(r"^([A-Za-z0-9&.'-]+)\s+([+-]?\d+(?:\.\d+)?)$")


def choose_book(odds_list: list, prefer: str) -> Optional[dict]:
    """Pick the preferred sportsbook's odds object, else the first available."""
    if not odds_list:
        return None
    for o in odds_list:
        book = (o.get("provider") or {}).get("name", "")
        if prefer.lower() in book.lower():
            return o
    return odds_list[0]


def parse_spread(odds: Optional[dict], home_abbr: str,
                 away_abbr: str) -> Optional[float]:
    """Return the HOME-relative spread (negative = home favored) from one book.

    ``details`` looks like ``"DUKE -7.5"`` (favorite abbr + number) or ``"EVEN"``.
    """
    if odds is None:
        return None
    details = (odds.get("details") or "").strip()
    if details.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
        return 0.0

    m = _SPREAD_RE.match(details)
    if m:
        fav_abbr, num = m.group(1), -abs(float(m.group(2)))  # named team is favorite
        if _norm(fav_abbr) == _norm(home_abbr):
            return num
        if _norm(fav_abbr) == _norm(away_abbr):
            return -num  # away favored -> home is the dog (+)

    # Fallback: numeric ``spread`` field is ESPN's home-relative line.
    spread = odds.get("spread")
    return float(spread) if isinstance(spread, (int, float)) else None


def _price(team_odds: dict) -> Optional[int]:
    """Best available American price for a side's spread (else moneyline)."""
    for key in ("spreadOdds", "pointSpreadOdds"):
        v = team_odds.get(key)
        if isinstance(v, (int, float)):
            return int(v)
    ml = team_odds.get("moneyLine")
    return int(ml) if isinstance(ml, (int, float)) else None


def event_to_game(event: dict, ratings: dict[str, float], prefer: str) -> Optional[dict]:
    comps = event.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    competitors = comp.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    home_team = home.get("team") or {}
    away_team = away.get("team") or {}
    home_abbr = home_team.get("abbreviation", "")
    away_abbr = away_team.get("abbreviation", "")

    status = (comp.get("status") or event.get("status") or {}).get("type", {})
    played = bool(status.get("completed"))

    kickoff = ""
    if comp.get("date"):
        try:
            dt = datetime.fromisoformat(comp["date"].replace("Z", "+00:00"))
            kickoff = dt.astimezone(timezone.utc).strftime("%a %b %-d %H:%MZ")
        except ValueError:
            kickoff = comp["date"]

    odds = choose_book(comp.get("odds") or [], prefer)
    book = (odds.get("provider") or {}).get("name") if odds else None
    home_spread = parse_spread(odds, home_abbr, away_abbr)
    home_price = _price(odds.get("homeTeamOdds", {})) if odds else None
    away_price = _price(odds.get("awayTeamOdds", {})) if odds else None

    game: dict = {
        "away": away_abbr or away_team.get("shortDisplayName", "AWAY"),
        "home": home_abbr or home_team.get("shortDisplayName", "HOME"),
        "kickoff": kickoff,
        "played": played,
        "_book": book,
    }
    if home_spread is None:
        game["_noline"] = True
    else:
        game["home_spread"] = home_spread
        if home_price is not None:
            game["home_price"] = home_price
        if away_price is not None:
            game["away_price"] = away_price

    hr = lookup_rating(home_team, ratings)
    ar = lookup_rating(away_team, ratings)
    if hr is not None and ar is not None:
        game["home_rating"] = hr
        game["away_rating"] = ar
    else:
        game["_unrated"] = True
        missing = []
        if hr is None:
            missing.append(home_team.get("location") or home_abbr)
        if ar is None:
            missing.append(away_team.get("location") or away_abbr)
        game["_missing_ratings"] = missing
    return game


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def next_saturday(today: Optional[datetime] = None) -> str:
    today = today or datetime.now(timezone.utc)
    days = (5 - today.weekday()) % 7  # Saturday == 5
    return (today + timedelta(days=days)).strftime("%Y%m%d")


def build(date: str, ratings_path: Path, prefer: str, groups: int,
          require_ratings: bool) -> dict:
    ratings = load_ratings(ratings_path)
    board = fetch_scoreboard(date, groups)
    events = board.get("events") or []
    games = []
    for ev in events:
        g = event_to_game(ev, ratings, prefer)
        if g is None:
            continue
        if require_ratings and (g.get("_unrated") or g.get("_noline")):
            continue
        games.append(g)
    return {"date": date, "games": games}


def summarize(payload: dict) -> str:
    games = payload["games"]
    scored = [g for g in games if not g.get("_unrated") and not g.get("_noline")
              and not g.get("played")]
    unrated = [g for g in games if g.get("_unrated")]
    noline = [g for g in games if g.get("_noline")]
    played = [g for g in games if g.get("played")]
    lines = [
        f"date {payload['date']}: {len(games)} events "
        f"-> {len(scored)} ready to score",
        f"  filtered: {len(played)} already played, "
        f"{len(noline)} without a line, {len(unrated)} missing SP+ ratings",
    ]
    if unrated:
        miss = sorted({t for g in unrated for t in g.get("_missing_ratings", [])})
        lines.append("  add ratings for: " + ", ".join(miss[:20]))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="fetch_slate",
        description="Build a real games.json from ESPN lines + your SP+ ratings.",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--date", help="YYYY-MM-DD or YYYYMMDD (default: next Saturday).")
    grp.add_argument("--saturday", action="store_true", help="Use the upcoming Saturday.")
    p.add_argument("--ratings", type=Path, required=True,
                   help="SP+ ratings file (.json {name:rating} or .csv team,rating).")
    p.add_argument("--out", type=Path, help="Write here (default: stdout).")
    p.add_argument("--book", default="DraftKings", help="Preferred sportsbook name.")
    p.add_argument("--groups", type=int, default=80, help="ESPN group (80 = FBS).")
    p.add_argument("--require-ratings", action="store_true",
                   help="Drop games missing a line or SP+ rating instead of flagging them.")
    args = p.parse_args(argv)

    if args.date:
        date = args.date.replace("-", "")
    else:
        date = next_saturday()

    payload = build(date, args.ratings, args.book, args.groups, args.require_ratings)

    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(summarize(payload), file=sys.stderr)
        print(f"  wrote {len(payload['games'])} games -> {args.out}", file=sys.stderr)
    else:
        print(summarize(payload), file=sys.stderr)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

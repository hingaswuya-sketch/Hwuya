#!/usr/bin/env python3
"""MLB Head-to-Head Trends.

For every game on a given day, look back at recent meetings between the two
clubs and print the head-to-head trends: the series record, the current
head-to-head win streak, average runs scored, the home team's record when
hosting this opponent, and the most recent results.

Data comes from the public MLB Stats API (``statsapi.mlb.com``); only the
Python standard library is required.

Examples
--------
    python3 mlb_h2h_trends.py                       # today's matchups
    python3 mlb_h2h_trends.py --date 2026-07-25     # a specific day
    python3 mlb_h2h_trends.py --lookback-days 1095  # search 3 seasons back
    python3 mlb_h2h_trends.py --postseason          # include playoff meetings
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional


API_BASE = "https://statsapi.mlb.com/api/v1"
SPORT_ID = 1  # Major League Baseball

# Regular season plus every postseason round. Spring training ("S"),
# exhibition ("E") and the All-Star game ("A") are intentionally excluded.
REGULAR_GAME_TYPES = ("R",)
POSTSEASON_GAME_TYPES = ("F", "D", "L", "W")


# ---- HTTP ------------------------------------------------------------
def fetch_json(path: str, params: dict, timeout: float) -> dict:
    """GET ``API_BASE/path?params`` and return the decoded JSON."""
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{API_BASE}/{path}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-h2h-trends/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"error: MLB API returned HTTP {exc.code} for {url}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"error: could not reach the MLB API ({exc.reason})")
    except (ValueError, TimeoutError) as exc:
        raise SystemExit(f"error: bad response from the MLB API: {exc}")


# ---- data model ------------------------------------------------------
@dataclass
class Team:
    id: int
    name: str


@dataclass
class Game:
    """A single completed game between two clubs."""

    date: str  # YYYY-MM-DD
    home: Team
    away: Team
    home_score: int
    away_score: int

    @property
    def winner_id(self) -> int:
        return self.home.id if self.home_score > self.away_score else self.away.id

    @property
    def loser_id(self) -> int:
        return self.away.id if self.home_score > self.away_score else self.home.id


@dataclass
class Matchup:
    """A scheduled game (not necessarily finished yet)."""

    date: str
    home: Team
    away: Team
    start_time: Optional[str]  # localized "H:MM PM" or None


@dataclass
class Trends:
    """Computed head-to-head trends for one matchup."""

    home: Team
    away: Team
    games: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.games)

    def wins(self, team_id: int) -> int:
        return sum(1 for g in self.games if g.winner_id == team_id)

    def runs(self, team_id: int) -> int:
        total = 0
        for g in self.games:
            total += g.home_score if g.home.id == team_id else g.away_score
        return total

    def avg_runs(self, team_id: int) -> float:
        return self.runs(team_id) / self.total if self.total else 0.0

    def home_site_record(self) -> tuple[int, int]:
        """The scheduled home team's W-L when hosting the scheduled away team."""
        wins = losses = 0
        for g in self.games:
            if g.home.id != self.home.id:
                continue
            if g.winner_id == self.home.id:
                wins += 1
            else:
                losses += 1
        return wins, losses

    def streak(self) -> Optional[tuple[Team, int]]:
        """The active head-to-head streak: (winning team, length)."""
        if not self.games:
            return None
        last_winner = self.games[-1].winner_id
        length = 0
        for g in reversed(self.games):
            if g.winner_id == last_winner:
                length += 1
            else:
                break
        team = self.home if last_winner == self.home.id else self.away
        return team, length

    def last_results(self, count: int) -> list:
        """Most recent games, newest first, as (winner Team, Game) pairs."""
        recent = self.games[-count:][::-1]
        out = []
        for g in recent:
            winner = self.home if g.winner_id == self.home.id else self.away
            out.append((winner, g))
        return out


# ---- API helpers -----------------------------------------------------
def _parse_start_time(game: dict) -> Optional[str]:
    iso = game.get("gameDate")
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return None
    hour = dt.hour % 12 or 12
    return f"{hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def _team(side: dict) -> Team:
    team = side.get("team", {})
    return Team(id=int(team.get("id", 0)), name=team.get("name", "Unknown"))


def get_matchups(target: date, timeout: float) -> list:
    """All games scheduled on ``target`` (regardless of status)."""
    data = fetch_json(
        "schedule",
        {"sportId": SPORT_ID, "date": target.isoformat()},
        timeout,
    )
    matchups = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            matchups.append(
                Matchup(
                    date=target.isoformat(),
                    home=_team(teams.get("home", {})),
                    away=_team(teams.get("away", {})),
                    start_time=_parse_start_time(game),
                )
            )
    return matchups


def _is_final(game: dict) -> bool:
    state = game.get("status", {}).get("abstractGameState", "")
    return state == "Final"


def get_h2h_games(
    home: Team,
    away: Team,
    start: date,
    end: date,
    game_types: tuple,
    timeout: float,
) -> list:
    """Completed games between ``home`` and ``away`` in ``[start, end]``."""
    data = fetch_json(
        "schedule",
        {
            "sportId": SPORT_ID,
            "teamId": home.id,
            "opponentId": away.id,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "gameType": list(game_types),
        },
        timeout,
    )
    games = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            if not _is_final(game):
                continue
            teams = game.get("teams", {})
            g_home, g_away = teams.get("home", {}), teams.get("away", {})
            gh, ga = _team(g_home), _team(g_away)
            ids = {gh.id, ga.id}
            # Guard against the API ignoring opponentId.
            if ids != {home.id, away.id}:
                continue
            if "score" not in g_home or "score" not in g_away:
                continue
            games.append(
                Game(
                    date=(game.get("gameDate", "") or "")[:10],
                    home=gh,
                    away=ga,
                    home_score=int(g_home["score"]),
                    away_score=int(g_away["score"]),
                )
            )
    games.sort(key=lambda g: g.date)
    return games


# ---- formatting ------------------------------------------------------
def _pct(wins: int, total: int) -> str:
    if not total:
        return ".---"
    return f"{wins / total:.3f}".lstrip("0")


def format_trends(t: Trends) -> str:
    lines = []
    hw, aw = t.wins(t.home.id), t.wins(t.away.id)
    lines.append(f"  Series (last {t.total} meeting{'s' if t.total != 1 else ''}):")
    # Show the leader first.
    leader, trailer = (t.home, t.away) if hw >= aw else (t.away, t.home)
    lw, tw = (hw, aw) if hw >= aw else (aw, hw)
    lines.append(f"    {leader.name:<24} {lw:>2}-{tw:<2}  {_pct(lw, t.total)}")
    lines.append(f"    {trailer.name:<24} {tw:>2}-{lw:<2}  {_pct(tw, t.total)}")

    streak = t.streak()
    if streak:
        team, length = streak
        lines.append(f"  Streak: {team.name} has won {length} straight in this series")

    lines.append(
        f"  Avg runs: {t.home.name} {t.avg_runs(t.home.id):.1f}, "
        f"{t.away.name} {t.avg_runs(t.away.id):.1f}"
    )

    hwins, hlosses = t.home_site_record()
    if hwins + hlosses:
        lines.append(
            f"  {t.home.name} at home vs {t.away.name}: {hwins}-{hlosses}"
        )

    recent = t.last_results(5)
    if recent:
        parts = []
        for winner, g in recent:
            hi = max(g.home_score, g.away_score)
            lo = min(g.home_score, g.away_score)
            parts.append(f"{winner.name.split()[-1]} {hi}-{lo}")
        lines.append("  Last 5 (winner, newest first): " + "  |  ".join(parts))
    return "\n".join(lines)


def matchup_header(m: Matchup) -> str:
    when = f" — {m.start_time}" if m.start_time else ""
    return f"{m.away.name} @ {m.home.name}{when}"


# ---- main ------------------------------------------------------------
def run(target: date, lookback_days: int, game_types: tuple, timeout: float) -> None:
    matchups = get_matchups(target, timeout)
    print(f"MLB head-to-head trends for {target.isoformat()}")
    if not matchups:
        print("No games scheduled.")
        return
    print(f"{len(matchups)} game(s) scheduled; "
          f"looking back {lookback_days} day(s).\n")

    start = target - timedelta(days=lookback_days)
    end = target - timedelta(days=1)  # only prior meetings
    for m in matchups:
        print("=" * 60)
        print(matchup_header(m))
        games = get_h2h_games(m.home, m.away, start, end, game_types, timeout)
        if not games:
            print("  No prior meetings in the lookback window.")
        else:
            print(format_trends(Trends(home=m.home, away=m.away, games=games)))
        print()


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"error: --date must be YYYY-MM-DD, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlb_h2h_trends",
        description="Show head-to-head trends for each of a day's MLB games.",
    )
    parser.add_argument(
        "--date",
        help="Day to inspect, YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="How many days back to search for prior meetings (default: 365).",
    )
    parser.add_argument(
        "--postseason",
        action="store_true",
        help="Include postseason meetings, not just regular-season games.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request network timeout in seconds (default: 20).",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.lookback_days < 1:
        raise SystemExit("error: --lookback-days must be a positive integer")
    target = parse_date(args.date) if args.date else date.today()
    game_types = REGULAR_GAME_TYPES
    if args.postseason:
        game_types = REGULAR_GAME_TYPES + POSTSEASON_GAME_TYPES
    run(target, args.lookback_days, game_types, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

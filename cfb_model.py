#!/usr/bin/env python3
"""College Football (CFB) Elo rating and prediction model.

A small, dependency-free tool that rates college football teams from game
results using an Elo model, ranks them, and predicts the outcome of a
matchup (win probability and projected margin).

Games are read from a CSV file with the columns::

    date,home,away,home_score,away_score,neutral

``neutral`` is optional (``1``/``true`` for a neutral-site game, blank or
``0`` otherwise). Rows are processed in file order, so list them
chronologically for the ratings to evolve sensibly.

If no games file is supplied, a small bundled demo season is used so the
model runs out of the box.

Examples
--------
    python cfb_model.py rate --games season.csv
    python cfb_model.py rate                      # bundled demo season
    python cfb_model.py rankings --top 10
    python cfb_model.py predict "Georgia" "Alabama"
    python cfb_model.py predict "Georgia" "Alabama" --neutral
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ---- model parameters -------------------------------------------------
BASE_RATING = 1500.0
# K controls how quickly ratings move after each game.
K_FACTOR = 40.0
# Home-field advantage expressed in Elo points (~2.5 points on the field).
HOME_ADVANTAGE = 65.0
# Elo points per point of scoring margin, used to convert a rating gap
# into a projected point spread (and back).
POINTS_PER_ELO = 25.0
# Fraction of a team's rating pulled back toward the mean between seasons.
SEASON_REGRESSION = 0.25


@dataclass
class Team:
    """A team's running Elo rating and win/loss record."""

    name: str
    rating: float = BASE_RATING
    wins: int = 0
    losses: int = 0
    ties: int = 0
    games: int = 0

    @property
    def record(self) -> str:
        base = f"{self.wins}-{self.losses}"
        return f"{base}-{self.ties}" if self.ties else base


@dataclass
class Game:
    """A single played game."""

    date: str
    home: str
    away: str
    home_score: int
    away_score: int
    neutral: bool = False


@dataclass
class Model:
    """Holds Elo ratings for all teams and updates them from games."""

    teams: dict[str, Team] = field(default_factory=dict)

    # ---- team access -------------------------------------------------
    def team(self, name: str) -> Team:
        key = name.strip()
        if key not in self.teams:
            self.teams[key] = Team(name=key)
        return self.teams[key]

    def find(self, name: str) -> Team:
        """Look up a team by (case-insensitive) name for queries."""
        key = name.strip().lower()
        for team in self.teams.values():
            if team.name.lower() == key:
                return team
        raise SystemExit(
            f"error: unknown team {name!r} (try 'rankings' to see known teams)"
        )

    # ---- core Elo math ----------------------------------------------
    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        """Probability that A beats B given their ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def margin_multiplier(margin: int, rating_diff: float) -> float:
        """Scale the update by margin of victory (autocorrelation-adjusted).

        Uses the widely used 538-style multiplier so blowouts move ratings
        more, but with diminishing returns and without letting a favorite's
        expected win inflate the swing.
        """
        return math.log(abs(margin) + 1.0) * (2.2 / (rating_diff * 0.001 + 2.2))

    def update(self, game: Game) -> None:
        home = self.team(game.home)
        away = self.team(game.away)

        home_adv = 0.0 if game.neutral else HOME_ADVANTAGE
        exp_home = self.expected(home.rating + home_adv, away.rating)

        if game.home_score > game.away_score:
            score_home, winner, loser = 1.0, home, away
        elif game.home_score < game.away_score:
            score_home, winner, loser = 0.0, away, home
        else:
            score_home, winner, loser = 0.5, None, None

        margin = abs(game.home_score - game.away_score)
        # Rating diff from the winner's perspective (0 for a tie).
        if winner is home:
            rating_diff = (home.rating + home_adv) - away.rating
        elif winner is away:
            rating_diff = away.rating - (home.rating + home_adv)
        else:
            rating_diff = 0.0

        mult = self.margin_multiplier(margin, rating_diff) if margin else 1.0
        delta = K_FACTOR * mult * (score_home - exp_home)

        home.rating += delta
        away.rating -= delta

        for team in (home, away):
            team.games += 1
        if winner is None:
            home.ties += 1
            away.ties += 1
        else:
            winner.wins += 1
            loser.losses += 1

    def run(self, games: Iterable[Game]) -> None:
        for game in games:
            self.update(game)

    def regress_to_mean(self) -> None:
        """Pull every rating partway back to the baseline (new season)."""
        for team in self.teams.values():
            team.rating = (
                team.rating * (1 - SEASON_REGRESSION)
                + BASE_RATING * SEASON_REGRESSION
            )

    # ---- rankings & predictions -------------------------------------
    def rankings(self) -> list[Team]:
        return sorted(
            self.teams.values(),
            key=lambda t: (t.rating, t.wins),
            reverse=True,
        )

    def predict(self, home_name: str, away_name: str, neutral: bool = False) -> dict:
        home = self.find(home_name)
        away = self.find(away_name)
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        p_home = self.expected(home.rating + home_adv, away.rating)
        spread = ((home.rating + home_adv) - away.rating) / POINTS_PER_ELO
        return {
            "home": home,
            "away": away,
            "neutral": neutral,
            "p_home": p_home,
            "p_away": 1.0 - p_home,
            "spread": spread,  # positive => home favored by this many points
        }


# ---- data loading ----------------------------------------------------
def _truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "neutral"}


def load_games(path: Path) -> list[Game]:
    if not path.exists():
        raise SystemExit(f"error: games file not found: {path}")
    games: list[Game] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"home", "away", "home_score", "away_score"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(
                    f"error: games file missing columns: {', '.join(sorted(missing))}"
                )
            for i, row in enumerate(reader, start=2):
                try:
                    games.append(
                        Game(
                            date=(row.get("date") or "").strip(),
                            home=row["home"].strip(),
                            away=row["away"].strip(),
                            home_score=int(row["home_score"]),
                            away_score=int(row["away_score"]),
                            neutral=_truthy(row.get("neutral")),
                        )
                    )
                except (ValueError, KeyError, AttributeError) as exc:
                    raise SystemExit(f"error: bad row {i} in {path}: {exc}")
    except OSError as exc:
        raise SystemExit(f"error: could not read {path}: {exc}")
    if not games:
        raise SystemExit(f"error: no games found in {path}")
    return games


def demo_games() -> list[Game]:
    """A small, invented sample season so the model runs with no input."""
    g = lambda d, h, a, hs, as_, n=False: Game(d, h, a, hs, as_, n)
    return [
        g("2024-09-07", "Georgia", "Clemson", 34, 3),
        g("2024-09-07", "Ohio State", "Michigan State", 38, 7),
        g("2024-09-14", "Alabama", "Wisconsin", 42, 10),
        g("2024-09-14", "Texas", "Michigan", 31, 12),
        g("2024-09-21", "Oregon", "Oregon State", 49, 14),
        g("2024-09-28", "Georgia", "Alabama", 27, 24),
        g("2024-10-05", "Ohio State", "Oregon", 32, 31),
        g("2024-10-12", "Texas", "Oklahoma", 34, 3, True),
        g("2024-10-19", "Penn State", "USC", 33, 30),
        g("2024-10-26", "Alabama", "Missouri", 34, 0),
        g("2024-11-02", "Oregon", "Michigan", 38, 17),
        g("2024-11-09", "Georgia", "Ole Miss", 28, 10),
        g("2024-11-16", "Ohio State", "Indiana", 38, 15),
        g("2024-11-23", "Texas", "Kentucky", 31, 14),
        g("2024-11-30", "Ohio State", "Michigan", 13, 20),
        g("2024-11-30", "Georgia", "Georgia Tech", 44, 42),
        g("2024-12-07", "Oregon", "Penn State", 45, 37, True),
        g("2024-12-07", "Texas", "Georgia", 19, 22, True),
    ]


# ---- command handlers ------------------------------------------------
def _build_model(args: argparse.Namespace) -> tuple[Model, int]:
    games = load_games(args.games) if args.games else demo_games()
    model = Model()
    model.run(games)
    return model, len(games)


def _print_rankings(model: Model, top: Optional[int]) -> None:
    ranks = model.rankings()
    if top:
        ranks = ranks[:top]
    width = max((len(t.name) for t in ranks), default=4)
    print(f"{'#':>3}  {'Team':<{width}}  {'Rating':>7}  Record")
    print(f"{'-' * 3}  {'-' * width}  {'-' * 7}  {'-' * 6}")
    for i, team in enumerate(ranks, start=1):
        print(f"{i:>3}  {team.name:<{width}}  {team.rating:>7.0f}  {team.record}")


def cmd_rate(model: Model, args: argparse.Namespace) -> None:
    _print_rankings(model, args.top)


def cmd_rankings(model: Model, args: argparse.Namespace) -> None:
    _print_rankings(model, args.top)


def cmd_predict(model: Model, args: argparse.Namespace) -> None:
    result = model.predict(args.home, args.away, neutral=args.neutral)
    home, away = result["home"], result["away"]
    spread = result["spread"]
    site = "at a neutral site" if result["neutral"] else f"at {home.name}"

    if spread >= 0:
        fav, dog, line = home, away, spread
    else:
        fav, dog, line = away, home, -spread

    print(f"{away.name} ({away.rating:.0f}) at {home.name} ({home.rating:.0f}) {site}")
    print(f"  {home.name} win probability: {result['p_home'] * 100:5.1f}%")
    print(f"  {away.name} win probability: {result['p_away'] * 100:5.1f}%")
    if line < 0.5:
        print("  Projected line: pick'em")
    else:
        print(f"  Projected line: {fav.name} by {line:.1f} (over {dog.name})")


# ---- argument parsing ------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfb_model",
        description="College football Elo ratings and game predictions.",
    )
    parser.add_argument(
        "--games",
        type=Path,
        default=None,
        help="CSV of games (columns: date,home,away,home_score,away_score,neutral). "
        "Defaults to a bundled demo season.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rate = sub.add_parser("rate", help="Compute ratings and show the table.")
    p_rate.add_argument("--top", type=int, default=None, help="Show only the top N teams.")
    p_rate.set_defaults(func=cmd_rate)

    p_rank = sub.add_parser("rankings", help="Show the current ranking table.")
    p_rank.add_argument("--top", type=int, default=None, help="Show only the top N teams.")
    p_rank.set_defaults(func=cmd_rankings)

    p_pred = sub.add_parser("predict", help="Predict a matchup.")
    p_pred.add_argument("home", help="Home team name.")
    p_pred.add_argument("away", help="Away team name.")
    p_pred.add_argument(
        "--neutral", action="store_true", help="Neutral-site game (no home-field edge)."
    )
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model, _ = _build_model(args)
    args.func(model, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

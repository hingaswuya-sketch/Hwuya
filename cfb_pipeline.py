"""
cfb_pipeline.py

Bridge: fetched Pinnacle lines -> model projections.

fetch_lines_pinnacle.py gives you GameLine objects (team names + Pinnacle
moneylines). cfb_model.project_game wants TeamInputs (season stats + talent) for
each side plus a GameContext. This module connects the two: you supply a stats
source keyed by team, and it drives project_game over every fetched line,
skipping games it has no stats for.

The market side has no stats source of its own — that's your model's job — so
the stats provider is an injected dependency. A dict-backed provider is included
(load it from your own JSON export); implement TeamStatsProvider against a DB or
API if you have one.

    from fetch_lines_pinnacle import load_lines_from_file
    from cfb_pipeline import DictStatsProvider, project_lines

    lines = load_lines_from_file("odds.json")
    provider = DictStatsProvider.from_json("team_stats.json")
    for res in project_lines(lines, provider, week_of_season=3):
        print(res.away_team, "@", res.home_team, res.recommended_stake_fraction)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Protocol

from cfb_model import (
    GameContext,
    ProjectionResult,
    TeamEfficiency,
    TeamInputs,
    TeamSituational,
    TeamTalent,
    project_game,
)
from fetch_lines_pinnacle import GameLine

logger = logging.getLogger("cfb_pipeline")

__all__ = [
    "TeamStatsProvider",
    "DictStatsProvider",
    "normalize_team_name",
    "default_context",
    "project_lines",
    "SkippedGame",
]


# ---------------------------------------------------------------------------
# Team-name normalization
# ---------------------------------------------------------------------------

# The Odds API returns full names like "Ohio State Buckeyes"; a stats export may
# be keyed "Ohio State", "ohio-state", or "OHIO STATE". Normalize both sides to
# a common key so lookups don't hinge on exact punctuation/casing.

# Common nickname suffixes worth stripping when they trail a school name. This
# is intentionally conservative — it only strips a known trailing mascot, never
# a word that could be part of the school name itself.
_MASCOTS = {
    "buckeyes", "spartans", "bulldogs", "crimson tide", "tide", "ducks",
    "huskies", "wolverines", "tigers", "aggies", "gators", "sooners",
    "longhorns", "trojans", "bruins", "seminoles", "hurricanes", "volunteers",
    "razorbacks", "rebels", "cornhuskers", "hawkeyes", "wildcats", "cardinals",
    "nittany lions", "fighting irish", "sun devils", "golden bears", "bears",
    "cougars", "utes", "beavers", "cavaliers", "hokies", "tar heels",
    "wolfpack", "cyclones", "jayhawks", "mountaineers", "boilermakers",
    "gophers", "badgers", "terrapins", "knights", "bearcats", "red raiders",
}


def normalize_team_name(name: str) -> str:
    """
    Reduce a team name to a stable lookup key: lowercased, punctuation stripped,
    whitespace collapsed, and a recognized trailing mascot removed.

        "Ohio State Buckeyes" -> "ohio state"
        "ohio-state"          -> "ohio state"
        "Texas A&M Aggies"     -> "texas am"
    """
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    # Strip a trailing mascot (up to two words, e.g. "crimson tide").
    for span in (2, 1):
        parts = s.split()
        if len(parts) > span:
            tail = " ".join(parts[-span:])
            if tail in _MASCOTS:
                s = " ".join(parts[:-span])
                break
    return s


# ---------------------------------------------------------------------------
# Stats provider
# ---------------------------------------------------------------------------

class TeamStatsProvider(Protocol):
    """Anything that can hand back model inputs for a team by name."""

    def get(self, team_name: str) -> Optional[TeamInputs]:
        ...


class DictStatsProvider:
    """
    In-memory provider backed by a name-keyed dict of TeamInputs. Lookups are
    normalized, so "Ohio State Buckeyes" resolves to a "Ohio State" entry.
    """

    def __init__(self, teams: Dict[str, TeamInputs]):
        self._by_key: Dict[str, TeamInputs] = {
            normalize_team_name(name): inputs for name, inputs in teams.items()
        }

    def get(self, team_name: str) -> Optional[TeamInputs]:
        return self._by_key.get(normalize_team_name(team_name))

    @classmethod
    def from_json(cls, path: str) -> "DictStatsProvider":
        """
        Load team stats from a JSON file. Shape: an object mapping team name to
        an object with "efficiency", "talent", and "situational" sub-objects
        whose fields match the TeamEfficiency / TeamTalent / TeamSituational
        dataclasses.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, dict]) -> "DictStatsProvider":
        teams: Dict[str, TeamInputs] = {}
        for name, blocks in raw.items():
            teams[name] = TeamInputs(
                name=name,
                efficiency=TeamEfficiency(**blocks["efficiency"]),
                talent=TeamTalent(**blocks["talent"]),
                situational=TeamSituational(**blocks["situational"]),
            )
        return cls(teams)


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def default_context(
    line: GameLine,
    week_of_season: int,
    common_opponents_count: int = 0,
    **overrides,
) -> GameContext:
    """
    Build a neutral-ish GameContext for a fetched line. The Odds API carries no
    injury/weather/travel data, so those default to zero — a caller with that
    data should pass it via **overrides or supply its own context factory.

    The home team is taken from the line (The Odds API's home_team), and the
    game is treated as a normal home game (not neutral site) unless overridden.
    """
    base = dict(
        is_home=line.home_team,
        is_neutral_site=False,
        travel_distance_penalty=0.0,
        wind_speed_mph=0.0,
        altitude_flag=False,
        injury_adjustment_home=0.0,
        injury_adjustment_away=0.0,
        trap_game_flag_home=False,
        trap_game_flag_away=False,
        bowl_eligibility_shock_home=0.0,
        bowl_eligibility_shock_away=0.0,
        week_of_season=week_of_season,
        common_opponents_count=common_opponents_count,
    )
    base.update(overrides)
    return GameContext(**base)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class SkippedGame:
    """A fetched line that could not be projected, and why."""
    home_team: str
    away_team: str
    reason: str


ContextFactory = Callable[[GameLine], GameContext]


def project_lines(
    lines: Iterable[GameLine],
    provider: TeamStatsProvider,
    week_of_season: int,
    context_factory: Optional[ContextFactory] = None,
    devig_method: str = "shin",
    skipped: Optional[List[SkippedGame]] = None,
) -> Iterator[ProjectionResult]:
    """
    Project every fetched line for which the provider has both teams' stats.

    Yields a ProjectionResult per playable game. Games missing stats for either
    side are skipped; if a `skipped` list is passed, a SkippedGame record is
    appended for each so the caller can report coverage.

    context_factory, if given, maps a GameLine to a GameContext (use it to inject
    injuries/weather/schedule connectivity). Otherwise default_context is used
    with the provided week_of_season and zero situational context.
    """
    make_context = context_factory or (
        lambda ln: default_context(ln, week_of_season=week_of_season)
    )

    for line in lines:
        home = provider.get(line.home_team)
        away = provider.get(line.away_team)
        if home is None or away is None:
            missing = []
            if home is None:
                missing.append(line.home_team)
            if away is None:
                missing.append(line.away_team)
            reason = f"no stats for: {', '.join(missing)}"
            logger.info("skipping %s @ %s (%s)", line.away_team, line.home_team, reason)
            if skipped is not None:
                skipped.append(SkippedGame(line.home_team, line.away_team, reason))
            continue

        yield project_game(
            home,
            away,
            make_context(line),
            pinnacle_home_odds=line.home_odds,
            pinnacle_away_odds=line.away_odds,
            devig_method=devig_method,
        )


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    here = os.path.dirname(os.path.abspath(__file__))
    lines_path = os.path.join(here, "tests", "fixtures", "odds_api_ncaaf.json")
    stats_path = os.path.join(here, "tests", "fixtures", "team_stats_sample.json")

    from fetch_lines_pinnacle import load_lines_from_file

    lines = load_lines_from_file(lines_path)
    provider = DictStatsProvider.from_json(stats_path)

    skipped: List[SkippedGame] = []
    results = list(project_lines(lines, provider, week_of_season=3, skipped=skipped))

    print(f"\nProjected {len(results)} game(s); skipped {len(skipped)}.\n")
    for r in results:
        flag = "  <-- PLAY" if r.recommended_stake_fraction > 0 else ""
        print(
            f"{r.away_team} @ {r.home_team}: margin {r.projected_margin:+.1f}, "
            f"model {r.model_home_win_prob:.3f} vs market {r.market_home_win_prob:.3f}, "
            f"edge {r.edge*100:+.1f}%, stake {r.recommended_stake_fraction:.3f}{flag}"
        )
    for s in skipped:
        print(f"(skipped) {s.away_team} @ {s.home_team}: {s.reason}")

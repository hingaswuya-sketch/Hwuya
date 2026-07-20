"""
series_favorite_tracker.py

Purpose
-------
Given two MLB teams currently playing a series, pull each completed game in
that series (via the free MLB Stats API — no key needed), pair it with the
pregame moneyline for that game, and report:

  - Who won each game (favorite or underdog, by closing/opening ML)
  - The score
  - The running streak of consecutive favorite-wins or underdog-wins
    within the series

Odds plug-in
------------
MLB Stats API has scores/schedule but NOT odds. This script exposes a single
function, `get_moneylines(date, away, home)`, that you should wire up to your
existing `fetch_lines.py` (BookmakersReview / Circa) module from the betting
pipeline. A manual-entry fallback is included so the script is runnable
standalone if fetch_lines isn't available in the current environment.

Usage
-----
    python series_favorite_tracker.py "Yankees" "Red Sox"
    python series_favorite_tracker.py "Yankees" "Red Sox" --start 2026-07-18

If moneylines aren't available via your odds module, the script will prompt
you to paste them in (or edit MANUAL_ODDS below) rather than silently
guessing.
"""

import argparse
import sys
from datetime import datetime, timedelta
import urllib.request
import json

STATS_API = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------------------
# Manual odds fallback: date -> {"away": ml, "home": ml}
# Fill this in per game if you don't have fetch_lines wired up yet.
# Negative number = favorite, positive = underdog (standard American odds).
# ---------------------------------------------------------------------------
MANUAL_ODDS = {
    # "2026-07-18": {"away": -135, "home": +115},
}


def get_team_id(team_name: str) -> int:
    url = f"{STATS_API}/teams?sportId=1"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    for t in data["teams"]:
        if team_name.lower() in t["name"].lower():
            return t["id"]
    raise ValueError(f"Could not find team matching '{team_name}'")


def get_series_games(team_a_id: int, team_b_id: int, start_date: str, days_ahead: int = 7):
    """Return completed games between the two teams starting from start_date."""
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url = (
        f"{STATS_API}/schedule?sportId=1&teamId={team_a_id}"
        f"&startDate={start_date}&endDate={end_date}&gameType=R"
    )
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)

    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            away_id = g["teams"]["away"]["team"]["id"]
            home_id = g["teams"]["home"]["team"]["id"]
            if team_b_id in (away_id, home_id) and g["status"]["abstractGameState"] == "Final":
                games.append({
                    "date": date_block["date"],
                    "away": g["teams"]["away"]["team"]["name"],
                    "home": g["teams"]["home"]["team"]["name"],
                    "away_score": g["teams"]["away"]["score"],
                    "home_score": g["teams"]["home"]["score"],
                })
    return games


def get_moneylines(date: str, away: str, home: str):
    """
    Plug point: wire this to your fetch_lines.py (BookmakersReview/Circa)
    to pull the opening or closing moneyline for this game/date.
    Falls back to MANUAL_ODDS, then to an interactive prompt.
    """
    if date in MANUAL_ODDS:
        return MANUAL_ODDS[date]["away"], MANUAL_ODDS[date]["home"]

    print(f"\nNo odds on file for {away} @ {home} ({date}).")
    try:
        away_ml = int(input(f"  Enter {away} (away) moneyline: ").strip())
        home_ml = int(input(f"  Enter {home} (home) moneyline: ").strip())
        return away_ml, home_ml
    except (ValueError, EOFError):
        print("  Skipping odds for this game — treating as unknown.")
        return None, None


def analyze_series(away_team: str, home_team: str, start_date: str):
    away_id = get_team_id(away_team)
    home_id = get_team_id(home_team)
    games = get_series_games(away_id, home_id, start_date)

    if not games:
        print("No completed games found for this series in the given window.")
        return

    print(f"\n{away_team} vs {home_team} — series results\n" + "-" * 50)

    streak_side = None
    streak_count = 0
    log = []

    for g in games:
        away_ml, home_ml = get_moneylines(g["date"], g["away"], g["home"])
        winner = g["away"] if g["away_score"] > g["home_score"] else g["home"]

        if away_ml is None or home_ml is None:
            fav = "UNKNOWN"
        else:
            fav_team = g["away"] if away_ml < home_ml else g["home"]
            fav = "FAVORITE" if winner == fav_team else "UNDERDOG"

        if fav == streak_side:
            streak_count += 1
        else:
            streak_side = fav
            streak_count = 1

        log.append({
            "date": g["date"],
            "score": f"{g['away']} {g['away_score']} - {g['home_score']} {g['home']}",
            "winner": winner,
            "result": fav,
            "streak": streak_count,
        })

    for entry in log:
        print(f"{entry['date']}  {entry['score']:<40}  Winner: {entry['winner']:<20}  "
              f"[{entry['result']}]  streak: {entry['streak']}")

    last = log[-1]
    print("-" * 50)
    if last["result"] != "UNKNOWN":
        print(f"Current streak: {last['result']} has won {last['streak']} straight in this series.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track favorite/underdog results within an MLB series.")
    parser.add_argument("away_team", help="Away team name (or partial, e.g. 'Yankees')")
    parser.add_argument("home_team", help="Home team name (or partial, e.g. 'Red Sox')")
    parser.add_argument("--start", default=datetime.now().strftime("%Y-%m-%d"),
                         help="Series start date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    analyze_series(args.away_team, args.home_team, args.start)

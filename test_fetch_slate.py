#!/usr/bin/env python3
"""Offline tests for fetch_slate parsing/join (no network).

Run: python test_fetch_slate.py
"""
from fetch_slate import (
    choose_book, parse_spread, event_to_game, lookup_rating, _norm, next_saturday,
)

RATINGS = {_norm("Duke"): 20.5, _norm("Tulane"): 7.2, _norm("Ohio State"): 30.0}


def test_norm():
    assert _norm("Ohio State") == "ohiostate"
    assert _norm("Texas A&M") == "texasam"


def test_choose_book_prefers_named():
    odds = [{"provider": {"name": "ESPN BET"}}, {"provider": {"name": "DraftKings"}}]
    assert choose_book(odds, "DraftKings")["provider"]["name"] == "DraftKings"
    # Falls back to first when preferred is absent.
    assert choose_book(odds, "Caesars")["provider"]["name"] == "ESPN BET"
    assert choose_book([], "DraftKings") is None


def test_parse_spread_home_favorite():
    assert parse_spread({"details": "DUKE -7.5"}, "DUKE", "TULN") == -7.5


def test_parse_spread_away_favorite():
    # Away team favored -> home is the underdog (positive line).
    assert parse_spread({"details": "TULN -3"}, "DUKE", "TULN") == 3.0


def test_parse_spread_even_and_fallback():
    assert parse_spread({"details": "EVEN"}, "A", "B") == 0.0
    assert parse_spread({"spread": -4.5}, "A", "B") == -4.5   # numeric fallback
    assert parse_spread({"details": "garbage"}, "A", "B") is None
    assert parse_spread(None, "A", "B") is None


def test_lookup_rating_by_location():
    team = {"abbreviation": "DUKE", "location": "Duke", "displayName": "Duke Blue Devils"}
    assert lookup_rating(team, RATINGS) == 20.5
    assert lookup_rating({"abbreviation": "XYZ", "location": "Nowhere"}, RATINGS) is None


def _event(details, completed=False, home_abbr="DUKE", away_abbr="TULN",
           home_loc="Duke", away_loc="Tulane"):
    return {
        "competitions": [{
            "date": "2026-09-05T19:30Z",
            "status": {"type": {"completed": completed}},
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": home_abbr, "location": home_loc,
                 "displayName": f"{home_loc} X", "shortDisplayName": home_loc}},
                {"homeAway": "away", "team": {"abbreviation": away_abbr, "location": away_loc,
                 "displayName": f"{away_loc} Y", "shortDisplayName": away_loc}},
            ],
            "odds": [{
                "provider": {"name": "DraftKings"},
                "details": details,
                "homeTeamOdds": {"spreadOdds": -110},
                "awayTeamOdds": {"spreadOdds": -110},
            }],
        }],
    }


def test_event_to_game_full():
    g = event_to_game(_event("DUKE -7.5"), RATINGS, "DraftKings")
    assert g["home"] == "DUKE" and g["away"] == "TULN"
    assert g["home_spread"] == -7.5
    assert g["home_rating"] == 20.5 and g["away_rating"] == 7.2
    assert g["home_price"] == -110
    assert g["_book"] == "DraftKings"
    assert not g["played"]


def test_event_to_game_unrated_and_played():
    g = event_to_game(_event("DUKE -7.5", completed=True,
                              away_abbr="XYZ", away_loc="Nowhere"), RATINGS, "DraftKings")
    assert g["played"] is True
    assert g.get("_unrated") is True
    assert "Nowhere" in g["_missing_ratings"]


def test_event_to_game_noline():
    ev = _event("DUKE -7.5")
    ev["competitions"][0]["odds"] = []
    g = event_to_game(ev, RATINGS, "DraftKings")
    assert g.get("_noline") is True
    assert "home_spread" not in g


def test_next_saturday_is_saturday():
    from datetime import datetime, timezone
    s = next_saturday(datetime(2026, 9, 2, tzinfo=timezone.utc))  # a Wednesday
    assert s == "20260905"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

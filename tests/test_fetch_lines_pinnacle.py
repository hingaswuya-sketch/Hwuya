"""Tests for fetch_lines_pinnacle: odds conversion and payload parsing."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_lines_pinnacle import (  # noqa: E402
    OddsApiError,
    decimal_to_american,
    fetch_pinnacle_lines,
    load_lines_from_file,
    parse_odds_api,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "odds_api_ncaaf.json")


class TestDecimalToAmerican(unittest.TestCase):
    def test_favorite(self):
        self.assertEqual(decimal_to_american(1.40), -250)

    def test_underdog(self):
        self.assertEqual(decimal_to_american(2.50), 150)

    def test_even_money_boundary(self):
        self.assertEqual(decimal_to_american(2.00), 100)

    def test_invalid_odds_raise(self):
        with self.assertRaises(ValueError):
            decimal_to_american(1.0)
        with self.assertRaises(ValueError):
            decimal_to_american(0.5)


class TestParseOddsApi(unittest.TestCase):
    def setUp(self):
        self.lines = load_lines_from_file(FIXTURE)

    def test_only_pinnacle_games_parsed(self):
        # Two of three fixture games have a Pinnacle line; the third does not.
        self.assertEqual(len(self.lines), 2)
        names = {(g.away_team, g.home_team) for g in self.lines}
        self.assertIn(("Michigan State Spartans", "Ohio State Buckeyes"), names)
        self.assertIn(("Alabama Crimson Tide", "Georgia Bulldogs"), names)
        self.assertNotIn(("Washington Huskies", "Oregon Ducks"), names)

    def test_uses_pinnacle_not_other_books(self):
        osu = next(g for g in self.lines if g.home_team == "Ohio State Buckeyes")
        # Pinnacle home price 1.357 -> -279 American, not DraftKings' 1.28.
        self.assertEqual(osu.home_odds, decimal_to_american(1.357))
        self.assertEqual(osu.away_odds, decimal_to_american(3.30))
        self.assertEqual(osu.bookmaker, "pinnacle")

    def test_commence_time_preserved(self):
        osu = next(g for g in self.lines if g.home_team == "Ohio State Buckeyes")
        self.assertEqual(osu.commence_time, "2026-09-12T23:30:00Z")

    def test_non_list_payload_raises(self):
        with self.assertRaises(OddsApiError):
            parse_odds_api({"not": "a list"})

    def test_empty_payload_is_empty(self):
        self.assertEqual(parse_odds_api([]), [])


class TestFetchRequiresKey(unittest.TestCase):
    def test_missing_key_raises(self):
        # Ensure no ambient key leaks in from the environment.
        saved = os.environ.pop("ODDS_API_KEY", None)
        try:
            with self.assertRaises(OddsApiError):
                fetch_pinnacle_lines(api_key=None)
        finally:
            if saved is not None:
                os.environ["ODDS_API_KEY"] = saved


class TestIntegrationWithModel(unittest.TestCase):
    """A fetched Pinnacle line should feed straight into the de-vig step."""

    def test_devig_a_fetched_line(self):
        from devig_pinnacle import devig
        lines = load_lines_from_file(FIXTURE)
        g = next(x for x in lines if x.home_team == "Georgia Bulldogs")
        r = devig(g.home_odds, g.away_odds, method="shin")
        self.assertAlmostEqual(r.home_prob + r.away_prob, 1.0, places=9)
        # Near pick'em market -> both sides close to 0.5.
        self.assertTrue(0.45 < r.home_prob < 0.55)


if __name__ == "__main__":
    unittest.main()

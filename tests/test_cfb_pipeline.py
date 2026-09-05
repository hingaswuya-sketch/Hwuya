"""Tests for cfb_pipeline: name normalization, stats provider, and the driver."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfb_model import ProjectionResult  # noqa: E402
from cfb_pipeline import (  # noqa: E402
    DictStatsProvider,
    SkippedGame,
    default_context,
    normalize_team_name,
    project_lines,
)
from fetch_lines_pinnacle import GameLine, load_lines_from_file  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
ODDS = os.path.join(FIX, "odds_api_ncaaf.json")
STATS = os.path.join(FIX, "team_stats_sample.json")


class TestNormalize(unittest.TestCase):
    def test_strips_mascot(self):
        self.assertEqual(normalize_team_name("Ohio State Buckeyes"), "ohio state")

    def test_two_word_mascot(self):
        self.assertEqual(normalize_team_name("Alabama Crimson Tide"), "alabama")

    def test_punctuation_and_case(self):
        self.assertEqual(normalize_team_name("ohio-state"), "ohio state")
        self.assertEqual(normalize_team_name("  OHIO   STATE  "), "ohio state")

    def test_does_not_overstrip(self):
        # "State" is not a mascot; must survive.
        self.assertEqual(normalize_team_name("Michigan State Spartans"), "michigan state")


class TestProvider(unittest.TestCase):
    def setUp(self):
        self.provider = DictStatsProvider.from_json(STATS)

    def test_lookup_by_full_name(self):
        # Stats keyed "Ohio State"; look up by the Odds API's full name.
        self.assertIsNotNone(self.provider.get("Ohio State Buckeyes"))

    def test_lookup_by_bare_name(self):
        self.assertIsNotNone(self.provider.get("Georgia"))

    def test_missing_team_returns_none(self):
        self.assertIsNone(self.provider.get("Fake University Aardvarks"))


class TestDefaultContext(unittest.TestCase):
    def test_home_and_week_set(self):
        line = GameLine("Ohio State Buckeyes", "Michigan State Spartans", -279, 330,
                        "2026-09-12T23:30:00Z")
        ctx = default_context(line, week_of_season=3)
        self.assertEqual(ctx.is_home, "Ohio State Buckeyes")
        self.assertEqual(ctx.week_of_season, 3)
        self.assertFalse(ctx.is_neutral_site)

    def test_overrides_apply(self):
        line = GameLine("A", "B", -110, -110, "")
        ctx = default_context(line, week_of_season=5, injury_adjustment_home=-3.0,
                              common_opponents_count=2)
        self.assertEqual(ctx.injury_adjustment_home, -3.0)
        self.assertEqual(ctx.common_opponents_count, 2)


class TestProjectLines(unittest.TestCase):
    def setUp(self):
        self.lines = load_lines_from_file(ODDS)
        self.provider = DictStatsProvider.from_json(STATS)

    def test_projects_both_covered_games(self):
        # Both Pinnacle games (OSU/MSU, UGA/Bama) have stats -> 2 projections.
        results = list(project_lines(self.lines, self.provider, week_of_season=3))
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, ProjectionResult)

    def test_records_skipped_when_stats_missing(self):
        # Provider with only Ohio State + Michigan State -> UGA/Bama skipped.
        partial = DictStatsProvider.from_json(STATS)
        partial._by_key.pop("georgia")
        skipped = []
        results = list(project_lines(self.lines, partial, week_of_season=3, skipped=skipped))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(skipped), 1)
        self.assertIsInstance(skipped[0], SkippedGame)
        self.assertIn("Georgia", skipped[0].reason)

    def test_context_factory_used(self):
        # Inject full schedule connectivity so confidence can exceed zero.
        results = list(project_lines(
            self.lines, self.provider, week_of_season=8,
            context_factory=lambda ln: default_context(
                ln, week_of_season=8, common_opponents_count=3),
        ))
        self.assertTrue(any(r.confidence > 0 for r in results))

    def test_devig_method_passthrough(self):
        shin = list(project_lines(self.lines, self.provider, week_of_season=3,
                                  devig_method="shin"))
        mult = list(project_lines(self.lines, self.provider, week_of_season=3,
                                  devig_method="multiplicative"))
        self.assertNotAlmostEqual(shin[0].market_home_win_prob,
                                  mult[0].market_home_win_prob, places=6)


if __name__ == "__main__":
    unittest.main()

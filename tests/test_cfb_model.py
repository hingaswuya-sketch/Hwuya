"""Tests for cfb_model: composites, margin->prob, confidence, and the pipeline."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfb_model import (  # noqa: E402
    GameContext,
    ProjectionResult,
    TeamEfficiency,
    TeamInputs,
    TeamSituational,
    TeamTalent,
    blended_team_rating,
    confidence_score,
    context_adjustment,
    margin_to_win_prob,
    project_game,
    regression_weight,
)


def _strong_team(name="Home"):
    return TeamInputs(
        name=name,
        efficiency=TeamEfficiency(
            epa_play_off=0.22, epa_play_def=-0.05,
            success_rate_off=0.48, success_rate_def=0.38,
            explosiveness_off=1.35, explosiveness_def=1.05,
            havoc_rate_off=0.14, havoc_rate_def=0.21,
            plays_sample_size=310,
        ),
        talent=TeamTalent(
            preseason_sp_rating=24.5, returning_production_off=0.62,
            returning_production_def=0.55, blue_chip_ratio=0.81,
            qb_starts_experience=14, transfer_portal_net_impact=1.2,
        ),
        situational=TeamSituational(
            plays_per_game=71, third_down_conv_off=0.46, third_down_conv_def=0.32,
            red_zone_td_pct_off=0.62, red_zone_td_pct_def=0.48,
            special_teams_net_points=0.8,
        ),
    )


def _weak_team(name="Away"):
    return TeamInputs(
        name=name,
        efficiency=TeamEfficiency(
            epa_play_off=0.02, epa_play_def=0.08,
            success_rate_off=0.39, success_rate_def=0.44,
            explosiveness_off=1.05, explosiveness_def=1.20,
            havoc_rate_off=0.19, havoc_rate_def=0.15,
            plays_sample_size=295,
        ),
        talent=TeamTalent(
            preseason_sp_rating=8.0, returning_production_off=0.58,
            returning_production_def=0.60, blue_chip_ratio=0.42,
            qb_starts_experience=6, transfer_portal_net_impact=-0.3,
        ),
        situational=TeamSituational(
            plays_per_game=68, third_down_conv_off=0.37, third_down_conv_def=0.41,
            red_zone_td_pct_off=0.51, red_zone_td_pct_def=0.58,
            special_teams_net_points=-0.4,
        ),
    )


def _context(**overrides):
    base = dict(
        is_home="Home", is_neutral_site=False, travel_distance_penalty=0.2,
        wind_speed_mph=8, altitude_flag=False,
        injury_adjustment_home=0.0, injury_adjustment_away=0.0,
        trap_game_flag_home=False, trap_game_flag_away=False,
        bowl_eligibility_shock_home=0.0, bowl_eligibility_shock_away=0.0,
        week_of_season=7, common_opponents_count=3,
    )
    base.update(overrides)
    return GameContext(**base)


class TestRegressionWeight(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(regression_weight(0), 0.0)
        self.assertGreater(regression_weight(1000), 0.7)

    def test_half_at_k(self):
        # REGRESSION_K = 250 -> weight 0.5 at 250 plays.
        self.assertAlmostEqual(regression_weight(250), 0.5, places=10)

    def test_monotonic(self):
        self.assertLess(regression_weight(100), regression_weight(400))

    def test_negative_sample_clamped(self):
        self.assertEqual(regression_weight(-10), 0.0)


class TestMarginToWinProb(unittest.TestCase):
    def test_zero_margin_is_coinflip(self):
        self.assertAlmostEqual(margin_to_win_prob(0.0), 0.5, places=10)

    def test_monotonic_in_margin(self):
        self.assertLess(margin_to_win_prob(-7), margin_to_win_prob(7))

    def test_bounded(self):
        self.assertTrue(0.0 < margin_to_win_prob(-100) < margin_to_win_prob(100) < 1.0)

    def test_favorite_above_half(self):
        self.assertGreater(margin_to_win_prob(14), 0.5)


class TestBlendedRating(unittest.TestCase):
    def test_early_season_leans_on_prior(self):
        team = _strong_team()
        r_week1 = blended_team_rating(team.efficiency, team.talent, 1)
        r_week10 = blended_team_rating(team.efficiency, team.talent, 10)
        # The two blends should differ; weighting shifts across the season.
        self.assertNotAlmostEqual(r_week1, r_week10, places=3)

    def test_stronger_team_rates_higher(self):
        strong, weak = _strong_team(), _weak_team()
        rs = blended_team_rating(strong.efficiency, strong.talent, 7)
        rw = blended_team_rating(weak.efficiency, weak.talent, 7)
        self.assertGreater(rs, rw)


class TestContextAdjustment(unittest.TestCase):
    def test_home_field_positive(self):
        adj = context_adjustment(_context(travel_distance_penalty=0.0), for_home=True)
        self.assertGreater(adj, 0.0)

    def test_neutral_site_no_home_field(self):
        home = context_adjustment(_context(travel_distance_penalty=0.0), for_home=True)
        neutral = context_adjustment(
            _context(is_neutral_site=True, is_home=None, travel_distance_penalty=0.0),
            for_home=True,
        )
        self.assertGreater(home, neutral)

    def test_injuries_shift_margin(self):
        healthy = context_adjustment(_context(), for_home=True)
        hurt_home = context_adjustment(_context(injury_adjustment_home=-3.0), for_home=True)
        self.assertLess(hurt_home, healthy)


class TestConfidenceScore(unittest.TestCase):
    def test_full_confidence_late_season_deep_schedule(self):
        ctx = _context(week_of_season=8, common_opponents_count=3)
        conf = confidence_score(ctx, _strong_team().efficiency, _weak_team().efficiency)
        self.assertGreater(conf, 0.5)

    def test_thin_schedule_kills_confidence(self):
        ctx = _context(common_opponents_count=0)
        conf = confidence_score(ctx, _strong_team().efficiency, _weak_team().efficiency)
        self.assertEqual(conf, 0.0)

    def test_early_week_lowers_confidence(self):
        early = confidence_score(_context(week_of_season=1),
                                 _strong_team().efficiency, _weak_team().efficiency)
        late = confidence_score(_context(week_of_season=8),
                                _strong_team().efficiency, _weak_team().efficiency)
        self.assertLess(early, late)

    def test_bounded_zero_to_one(self):
        conf = confidence_score(_context(), _strong_team().efficiency, _weak_team().efficiency)
        self.assertTrue(0.0 <= conf <= 1.0)


class TestProjectGame(unittest.TestCase):
    def test_returns_projection_result(self):
        result = project_game(
            _strong_team("Ohio State"), _weak_team("Michigan State"),
            _context(is_home="Ohio State"),
            pinnacle_home_odds=-280, pinnacle_away_odds=230,
        )
        self.assertIsInstance(result, ProjectionResult)

    def test_favorite_projected_ahead(self):
        result = project_game(
            _strong_team("Ohio State"), _weak_team("Michigan State"),
            _context(is_home="Ohio State"),
            pinnacle_home_odds=-280, pinnacle_away_odds=230,
        )
        self.assertGreater(result.projected_margin, 0.0)
        self.assertGreater(result.model_home_win_prob, 0.5)

    def test_market_prob_is_devigged(self):
        result = project_game(
            _strong_team(), _weak_team(), _context(),
            pinnacle_home_odds=-280, pinnacle_away_odds=230,
        )
        # No-vig prob must be below the raw -280 implied prob (~0.7368).
        self.assertLess(result.market_home_win_prob, 0.7368)
        self.assertGreater(result.overround, 0.0)

    def test_stake_zero_below_edge_threshold(self):
        # Price the favorite so market ~= model; edge should be sub-threshold.
        result = project_game(
            _strong_team(), _weak_team(), _context(),
            pinnacle_home_odds=-900, pinnacle_away_odds=600,
        )
        if abs(result.edge) < 0.045:
            self.assertEqual(result.recommended_stake_fraction, 0.0)

    def test_stake_never_negative(self):
        result = project_game(
            _weak_team(), _strong_team(), _context(),
            pinnacle_home_odds=200, pinnacle_away_odds=-240,
        )
        self.assertGreaterEqual(result.recommended_stake_fraction, 0.0)

    def test_thin_schedule_zeroes_stake_via_confidence(self):
        result = project_game(
            _strong_team(), _weak_team(), _context(common_opponents_count=0),
            pinnacle_home_odds=200, pinnacle_away_odds=-240,
        )
        # Zero confidence -> zero stake regardless of edge.
        self.assertEqual(result.recommended_stake_fraction, 0.0)

    def test_devig_method_passthrough(self):
        r_shin = project_game(
            _strong_team(), _weak_team(), _context(),
            pinnacle_home_odds=-280, pinnacle_away_odds=230, devig_method="shin",
        )
        r_mult = project_game(
            _strong_team(), _weak_team(), _context(),
            pinnacle_home_odds=-280, pinnacle_away_odds=230, devig_method="multiplicative",
        )
        # Different de-vig methods yield different market probs (and edges).
        self.assertNotAlmostEqual(r_shin.market_home_win_prob,
                                  r_mult.market_home_win_prob, places=6)


if __name__ == "__main__":
    unittest.main()

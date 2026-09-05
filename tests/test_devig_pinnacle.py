"""Tests for devig_pinnacle: American-odds conversion and no-vig methods."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devig_pinnacle import (  # noqa: E402
    DEVIG_METHODS,
    DevigResult,
    american_to_implied_prob,
    compute_edge,
    devig,
)


class TestAmericanToImplied(unittest.TestCase):
    def test_negative_odds(self):
        # -150 favorite implies 60%.
        self.assertAlmostEqual(american_to_implied_prob(-150), 0.6, places=10)

    def test_positive_odds(self):
        # +150 underdog implies 40%.
        self.assertAlmostEqual(american_to_implied_prob(150), 0.4, places=10)

    def test_even_money(self):
        self.assertAlmostEqual(american_to_implied_prob(100), 0.5, places=10)
        self.assertAlmostEqual(american_to_implied_prob(-100), 0.5, places=10)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            american_to_implied_prob(0)


class TestDevigMethods(unittest.TestCase):
    HOME, AWAY = -280, 230

    def test_all_methods_sum_to_one(self):
        for m in DEVIG_METHODS:
            r = devig(self.HOME, self.AWAY, method=m)
            self.assertAlmostEqual(r.home_prob + r.away_prob, 1.0, places=9,
                                   msg=f"method {m} did not sum to 1")

    def test_overround_is_positive_and_consistent(self):
        r = devig(self.HOME, self.AWAY, method="shin")
        q_home = american_to_implied_prob(self.HOME)
        q_away = american_to_implied_prob(self.AWAY)
        self.assertAlmostEqual(r.overround, (q_home + q_away) - 1.0, places=10)
        self.assertGreater(r.overround, 0.0)

    def test_devig_lowers_favorite_probability(self):
        # Stripping vig must reduce the favorite's inflated implied prob.
        q_home = american_to_implied_prob(self.HOME)
        r = devig(self.HOME, self.AWAY, method="shin")
        self.assertLess(r.home_prob, q_home)

    def test_shin_z_in_range(self):
        r = devig(self.HOME, self.AWAY, method="shin")
        self.assertGreaterEqual(r.z, 0.0)
        self.assertLess(r.z, 1.0)

    def test_methods_broadly_agree(self):
        probs = [devig(self.HOME, self.AWAY, method=m).home_prob for m in DEVIG_METHODS]
        # Different de-vig methods should land within a few points of each other
        # on a normal two-way market.
        self.assertLess(max(probs) - min(probs), 0.03)

    def test_symmetric_market_is_fifty_fifty(self):
        for m in DEVIG_METHODS:
            r = devig(-110, -110, method=m)
            self.assertAlmostEqual(r.home_prob, 0.5, places=6,
                                   msg=f"method {m} not symmetric")
            self.assertAlmostEqual(r.away_prob, 0.5, places=6)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            devig(-110, -110, method="nope")

    def test_result_validates_sum(self):
        with self.assertRaises(ValueError):
            DevigResult(home_prob=0.6, away_prob=0.6, overround=0.2,
                        method="multiplicative", home_prob_raw=0.6,
                        away_prob_raw=0.6)


class TestComputeEdge(unittest.TestCase):
    def test_positive_edge(self):
        self.assertAlmostEqual(compute_edge(0.60, 0.55), 0.05, places=10)

    def test_negative_edge(self):
        self.assertAlmostEqual(compute_edge(0.50, 0.55), -0.05, places=10)


if __name__ == "__main__":
    unittest.main()

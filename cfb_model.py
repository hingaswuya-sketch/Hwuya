"""
cfb_model.py

Core CFB scoring engine. Ties together everything built so far:

  team stats -> composite scores -> projected margin -> win probability
  -> compare against Pinnacle no-vig probability (devig_pinnacle.py)
  -> edge -> confidence-adjusted EV / stake sizing

Mirrors the subsystem-composite pattern from your MLB model (SP True
Talent, Bullpen Engine, Offense Composite) but built for CFB's specific
holes: talent/roster prior, garbage-time filtering, regression by sample
size, and wider variance/confidence intervals than your MLB/NFL builds.

This is the scoring engine only — fetch_lines_pinnacle.py supplies the
market side, devig_pinnacle.py supplies the no-vig probability conversion.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from devig_pinnacle import devig, compute_edge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cfb_model")


# ---------------------------------------------------------------------------
# Input data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamEfficiency:
    """Opponent-adjusted efficiency inputs, per team."""
    epa_play_off: float
    epa_play_def: float          # allowed, lower is better
    success_rate_off: float
    success_rate_def: float      # allowed
    explosiveness_off: float     # IsoPPP-style
    explosiveness_def: float     # allowed
    havoc_rate_off: float        # havoc allowed (offense's exposure)
    havoc_rate_def: float        # havoc generated (defense's disruption)
    plays_sample_size: int       # total plays this season so far — drives
                                  # regression-to-mean weighting


@dataclass
class TeamTalent:
    """Preseason / roster-continuity prior — the Bayesian anchor that
    matters more in CFB than pro sports due to yearly roster turnover."""
    preseason_sp_rating: float   # SP+/FPI-style prior, points scale
    returning_production_off: float   # 0-1
    returning_production_def: float   # 0-1
    blue_chip_ratio: float             # 0-1, recruiting-based
    qb_starts_experience: int          # career starts of current starter
    transfer_portal_net_impact: float  # signed, points-scale estimate


@dataclass
class TeamSituational:
    """Pace, situational efficiency, special teams."""
    plays_per_game: float
    third_down_conv_off: float
    third_down_conv_def: float   # allowed
    red_zone_td_pct_off: float
    red_zone_td_pct_def: float   # allowed
    special_teams_net_points: float  # composite ST value, points-scale


@dataclass
class GameContext:
    """Situational context for a specific matchup."""
    is_home: Optional[str]         # "home_team_name" or None if neutral site
    is_neutral_site: bool
    travel_distance_penalty: float # points, precomputed
    wind_speed_mph: float
    altitude_flag: bool
    injury_adjustment_home: float  # signed points (OL/QB weighted higher)
    injury_adjustment_away: float
    trap_game_flag_home: bool
    trap_game_flag_away: bool
    bowl_eligibility_shock_home: float  # points, e.g. opt-outs late season
    bowl_eligibility_shock_away: float
    week_of_season: int             # drives talent-prior blend weight
    common_opponents_count: int      # schedule-graph connectivity, drives
                                      # confidence interval width


@dataclass
class TeamInputs:
    name: str
    efficiency: TeamEfficiency
    talent: TeamTalent
    situational: TeamSituational


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

# Regression-to-mean: shrink efficiency numbers toward 0 (average) as a
# function of sample size. Small samples (early season) get heavily
# shrunk; by ~400+ plays, shrinkage approaches zero.
REGRESSION_K = 250  # tuning constant — plays at which shrinkage weight = 0.5


def regression_weight(plays_sample_size: int) -> float:
    """Returns weight in [0,1] for how much to trust raw efficiency vs.
    shrinking toward zero/prior. Higher = trust the raw number more."""
    if plays_sample_size <= 0:
        return 0.0
    return plays_sample_size / (plays_sample_size + REGRESSION_K)


def efficiency_composite(eff: TeamEfficiency) -> float:
    """
    Combines EPA, success rate, explosiveness, and havoc into a single
    points-scale offensive-minus-defensive efficiency rating, shrunk by
    sample size.

    Weights below are a reasonable starting point (EPA-driven, matching
    how most public CFB efficiency models are constructed) — treat as
    tunable, not gospel. Validate/refit against your own backtest.
    """
    raw_off = (
        eff.epa_play_off * 12.0
        + eff.success_rate_off * 6.0
        + eff.explosiveness_off * 4.0
        - eff.havoc_rate_off * 8.0   # havoc allowed hurts offense
    )
    raw_def = (
        -eff.epa_play_def * 12.0     # lower allowed EPA = better defense
        - eff.success_rate_def * 6.0
        - eff.explosiveness_def * 4.0
        + eff.havoc_rate_def * 8.0   # havoc generated helps defense
    )
    raw_composite = raw_off + raw_def

    w = regression_weight(eff.plays_sample_size)
    return raw_composite * w  # shrinks toward 0 as sample size shrinks


def talent_prior(talent: TeamTalent) -> float:
    """
    Preseason/roster-continuity prior, points-scale, on the same scale as
    efficiency_composite so they can be blended directly.
    """
    return (
        talent.preseason_sp_rating
        + talent.returning_production_off * 3.0
        + talent.returning_production_def * 3.0
        + talent.blue_chip_ratio * 5.0
        + min(talent.qb_starts_experience, 30) * 0.1   # diminishing returns cap
        + talent.transfer_portal_net_impact
    )


def blended_team_rating(eff: TeamEfficiency, talent: TeamTalent, week_of_season: int) -> float:
    """
    Bayesian-style blend of in-season efficiency and preseason talent
    prior. Early season -> lean on prior. Late season -> lean on observed
    efficiency. This is the piece flagged earlier as commonly missing:
    a dynamic blend rather than a hard cutover at some game count.
    """
    # Blend weight ramps from mostly-prior (week 1) to mostly-efficiency
    # (week 8+), independent of the sample-size regression already
    # applied inside efficiency_composite — this captures season-long
    # trust shift, not just per-stat noise.
    season_progress = min(week_of_season, 10) / 10.0  # caps ramp at week 10
    efficiency_weight = 0.3 + 0.6 * season_progress    # 0.3 -> 0.9
    prior_weight = 1.0 - efficiency_weight

    return (
        efficiency_composite(eff) * efficiency_weight
        + talent_prior(talent) * prior_weight
    )


def situational_adjustment(team: TeamSituational, opponent: TeamSituational) -> float:
    """
    Converts situational splits into a points-scale adjustment. Pace
    interaction matters: efficiency edge translates into more points for
    a fast team, so scale by combined pace relative to a league-average
    baseline.
    """
    league_avg_pace = 70.0  # plays/game, adjust to current season's actual average

    third_down_edge = (team.third_down_conv_off - opponent.third_down_conv_def) * 2.5
    red_zone_edge = (team.red_zone_td_pct_off - opponent.red_zone_td_pct_def) * 2.0
    st_edge = team.special_teams_net_points - opponent.special_teams_net_points

    pace_factor = (team.plays_per_game + opponent.plays_per_game) / (2 * league_avg_pace)

    return (third_down_edge + red_zone_edge) * pace_factor + st_edge


def context_adjustment(context: GameContext, for_home: bool) -> float:
    """
    Points-scale adjustment for home/away/neutral, weather, injuries,
    trap games, and bowl-eligibility shocks. Returns the net adjustment
    to apply to the home team's projected margin (positive = favors home).
    """
    HOME_FIELD_POINTS = 2.3  # CFB home field is typically smaller than NFL's ~2.5-3

    adj = 0.0
    if context.is_neutral_site:
        adj += 0.0
    elif context.is_home:
        adj += HOME_FIELD_POINTS

    adj -= context.travel_distance_penalty  # penalty applies to away team's trip,
                                              # precomputed as a positive number
                                              # to subtract from home margin...
                                              # NOTE: verify sign convention matches
                                              # your precompute step.

    if context.wind_speed_mph > 20:
        adj -= 0.5  # high wind suppresses passing efficiency, slight edge to
                    # run-heavy/favorite in most model conventions — tune this

    if context.altitude_flag:
        adj += 0.5  # historically favors home team at extreme altitude venues

    adj += context.injury_adjustment_home - context.injury_adjustment_away

    if context.trap_game_flag_home:
        adj -= 1.0
    if context.trap_game_flag_away:
        adj += 1.0

    adj += context.bowl_eligibility_shock_home - context.bowl_eligibility_shock_away

    return adj


# ---------------------------------------------------------------------------
# Margin -> win probability conversion
# ---------------------------------------------------------------------------

# CFB margin distribution has fatter tails than NFL (more blowouts) —
# using a wider logistic scale constant than you'd use for NFL/MLB.
# This constant should be fit against your own historical margin data;
# the value below is a reasonable starting point based on public CFB
# margin-distribution research, NOT a substitute for your own backtest.
CFB_LOGISTIC_SCALE = 13.5


def margin_to_win_prob(projected_margin: float) -> float:
    """
    Converts a projected point margin (positive = home favored) into a
    home win probability using a logistic function calibrated for CFB's
    fatter-tailed margin distribution.
    """
    return 1.0 / (1.0 + math.exp(-projected_margin / CFB_LOGISTIC_SCALE))


# ---------------------------------------------------------------------------
# Confidence scoring (drives stake sizing, independent of edge size)
# ---------------------------------------------------------------------------

def confidence_score(context: GameContext, home_eff: TeamEfficiency, away_eff: TeamEfficiency) -> float:
    """
    Returns a 0-1 confidence multiplier reflecting how much to trust this
    specific projection, independent of how large the computed edge is.
    Low sample size, thin schedule connectivity, or early-season timing
    all reduce confidence -- this is the guardrail flagged earlier as
    commonly missing in CFB models ported from higher-sample-size sports.
    """
    sample_conf = min(
        regression_weight(home_eff.plays_sample_size),
        regression_weight(away_eff.plays_sample_size),
    )

    schedule_conf = min(context.common_opponents_count / 3.0, 1.0)  # 3+ common
                                                                     # opponents = full confidence

    week_conf = min(context.week_of_season / 6.0, 1.0)  # ramps up through week 6

    # Combine conservatively — take the minimum rather than averaging, so
    # one badly thin factor (e.g. zero common opponents) can't be masked
    # by strong values elsewhere.
    return min(sample_conf, schedule_conf, week_conf)


# ---------------------------------------------------------------------------
# Full projection + edge/EV pipeline
# ---------------------------------------------------------------------------

@dataclass
class ProjectionResult:
    home_team: str
    away_team: str
    projected_margin: float       # positive = home favored, in points
    model_home_win_prob: float
    market_home_win_prob: float   # no-vig, from Pinnacle
    edge: float
    confidence: float
    ev_home: float                # per $1 staked, home side
    recommended_stake_fraction: float  # confidence-adjusted fractional Kelly
    overround: float


# Minimum edge threshold before a game is considered signal rather than
# noise -- set wider than MLB given CFB's smaller per-team sample and
# higher variance, per your earlier flag.
MIN_EDGE_THRESHOLD = 0.045  # 4.5%

# Base Kelly fraction cap -- always bet fractional Kelly, never full.
BASE_KELLY_FRACTION = 0.25


def project_game(
    home: TeamInputs,
    away: TeamInputs,
    context: GameContext,
    pinnacle_home_odds: float,
    pinnacle_away_odds: float,
    devig_method: str = "shin",
) -> ProjectionResult:

    home_rating = blended_team_rating(home.efficiency, home.talent, context.week_of_season)
    away_rating = blended_team_rating(away.efficiency, away.talent, context.week_of_season)

    situational = situational_adjustment(home.situational, away.situational)
    ctx_adj = context_adjustment(context, for_home=True)

    projected_margin = (home_rating - away_rating) + situational + ctx_adj

    model_home_win_prob = margin_to_win_prob(projected_margin)

    devig_result = devig(pinnacle_home_odds, pinnacle_away_odds, method=devig_method)
    market_home_win_prob = devig_result.home_prob

    edge = compute_edge(model_home_win_prob, market_home_win_prob)

    conf = confidence_score(context, home.efficiency, away.efficiency)

    # EV per $1 staked at the given American odds, home side
    if pinnacle_home_odds > 0:
        payout_mult = pinnacle_home_odds / 100.0
    else:
        payout_mult = 100.0 / abs(pinnacle_home_odds)

    ev_home = (model_home_win_prob * payout_mult) - (1 - model_home_win_prob)

    # Fractional Kelly, scaled down further by confidence -- a large edge
    # on a low-confidence game (e.g. week 1, thin schedule) should NOT
    # size up like a large edge on a high-confidence game.
    kelly_full = edge / payout_mult if payout_mult > 0 else 0.0
    recommended_stake_fraction = max(0.0, kelly_full * BASE_KELLY_FRACTION * conf)

    if abs(edge) < MIN_EDGE_THRESHOLD:
        recommended_stake_fraction = 0.0
        logger.info(f"{home.name} vs {away.name}: edge {edge:+.3f} below threshold, no play.")

    return ProjectionResult(
        home_team=home.name,
        away_team=away.name,
        projected_margin=projected_margin,
        model_home_win_prob=model_home_win_prob,
        market_home_win_prob=market_home_win_prob,
        edge=edge,
        confidence=conf,
        ev_home=ev_home,
        recommended_stake_fraction=recommended_stake_fraction,
        overround=devig_result.overround,
    )


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    home = TeamInputs(
        name="Ohio State",
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

    away = TeamInputs(
        name="Michigan State",
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

    context = GameContext(
        is_home="Ohio State", is_neutral_site=False, travel_distance_penalty=0.2,
        wind_speed_mph=8, altitude_flag=False,
        injury_adjustment_home=0.0, injury_adjustment_away=-0.5,
        trap_game_flag_home=False, trap_game_flag_away=False,
        bowl_eligibility_shock_home=0.0, bowl_eligibility_shock_away=0.0,
        week_of_season=7, common_opponents_count=2,
    )

    result = project_game(
        home, away, context,
        pinnacle_home_odds=-280, pinnacle_away_odds=230,
    )

    print(f"\n{result.away_team} @ {result.home_team}")
    print(f"Projected margin (home): {result.projected_margin:+.2f}")
    print(f"Model home win prob:     {result.model_home_win_prob:.4f}")
    print(f"Market home win prob:    {result.market_home_win_prob:.4f}  (overround {result.overround:.4f})")
    print(f"Edge:                    {result.edge:+.4f} ({result.edge*100:+.2f}%)")
    print(f"Confidence:              {result.confidence:.2f}")
    print(f"EV (home, per $1):       {result.ev_home:+.4f}")
    print(f"Recommended stake frac:  {result.recommended_stake_fraction:.4f}")

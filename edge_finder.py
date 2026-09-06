#!/usr/bin/env python3
"""Daily college-football spread edge finder.

Reproduces the "raw-overshoot" card pipeline:

  1. Model line   -- from 2026-final SP+ ratings:
                       margin_home = (home_SP+ - away_SP+) / divisor + home_field
                     (or supply ``model_margin`` per game directly).
  2. Market line  -- the book's home spread (e.g. DUKE -7.5) and price.
  3. Raw edge     -- the model's cover-probability advantage over the price:
                       edge = P(model side covers) - break_even(price)
                     expressed as a percentage.
  4. Calibrated   -- raw edge shrunk toward the market (default x0.80).
  5. Gate & rank  -- drop already-played games, keep games whose RAW edge is at
                     or below the overshoot CAP (default 10%), rank strongest
                     -> weakest, cap to the top N (default 5).

The old 3-5% band is gone: the only gate is the overshoot cap.  Anything over
the cap is a FLAG (the model disagrees so hard with the market that it's
probably stale data, not a real edge), and is reported but not played.

Design goals: zero third-party dependencies for the core math, a single data
file in, a ranked slate out, so it can run unattended every morning from cron
or a scheduled trigger.

Examples
--------
    # Score today's slate from a data file and print the ranked cards.
    python edge_finder.py scan --data games.json

    # Same, but as JSON (for piping into a notifier / spreadsheet).
    python edge_finder.py scan --data games.json --json

    # Show every game the model looked at, including sub-threshold and flagged.
    python edge_finder.py scan --data games.json --show-all

    # Explain the math for one matchup.
    python edge_finder.py explain --data games.json --home DUKE
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Tunable configuration.  Defaults reproduce the reference card, except the cap
# which is raised from 9% to 10% per the current spec.
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    divisor: float = 16.2       # SP+ points-to-margin divisor (from the method)
    home_field: float = 3.0     # home-field advantage, points
    shrink: float = 0.80        # calibration: pull the raw edge toward market
    sigma: float = 18.7         # SD of (actual margin - spread), CFB, points
    default_price: int = -110   # assumed American odds when a game omits price
    cap_pct: float = 10.0       # raw-overshoot cap (was 9.0) -- the only gate
    floor_pct: float = 0.0      # ignore edges below this (3-5% band removed)
    top_n: int = 5              # keep at most this many plays

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}  # type: ignore[attr-defined]
        return cls(**known)


@dataclass
class Game:
    """One upcoming matchup with a market line and a model input."""

    away: str
    home: str
    home_spread: float                 # book line for the HOME team, e.g. -7.5
    kickoff: str = ""                  # free-form, e.g. "Sat Sep 5 3:30 ET"
    home_price: Optional[int] = None   # American odds on the home spread
    away_price: Optional[int] = None   # American odds on the away spread
    # Model input: supply EITHER a direct model margin, OR both SP+ ratings.
    model_margin: Optional[float] = None   # expected HOME margin, points (+ = home favored)
    home_rating: Optional[float] = None    # 2026 final SP+ for the home team
    away_rating: Optional[float] = None    # 2026 final SP+ for the away team
    played: bool = False               # already-played games are filtered out

    @classmethod
    def from_dict(cls, data: dict) -> "Game":
        known = {k: data.get(k) for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        known = {k: v for k, v in known.items() if v is not None}
        return cls(**known)


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #
def american_to_prob(odds: int) -> float:
    """Break-even (implied) win probability for American odds, no vig removed."""
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (stdlib only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def model_margin(game: Game, cfg: Config) -> float:
    """Expected HOME margin in points (positive = home favored).

    Uses an explicit ``model_margin`` when given, otherwise the SP+ formula.
    """
    if game.model_margin is not None:
        return game.model_margin
    if game.home_rating is None or game.away_rating is None:
        raise ValueError(
            f"{game.away} @ {game.home}: need model_margin or both SP+ ratings"
        )
    return (game.home_rating - game.away_rating) / cfg.divisor + cfg.home_field


@dataclass
class Pick:
    """The scored result for one game."""

    game: Game
    side: str            # team the model would bet
    line: float          # that side's spread, book convention
    price: int           # that side's American odds
    model_line: float    # model spread for that side (negative = favored)
    cover_prob: float    # model P(that side covers)
    raw_pct: float       # raw edge over break-even, in percent
    cal_pct: float       # calibrated edge (raw * shrink), in percent
    over_cap: bool       # raw edge exceeds the cap -> flagged, not played

    @property
    def flag_reason(self) -> str:
        return f"just over the {int(round(self.raw_pct))}% edge (cap {_fmt_cap()})"


def score_game(game: Game, cfg: Config) -> Pick:
    """Score a single game, picking the side the model favors vs. the market."""
    m = model_margin(game, cfg)
    home_price = game.home_price if game.home_price is not None else cfg.default_price
    away_price = game.away_price if game.away_price is not None else cfg.default_price

    # Home covers when actual_home_margin + home_spread > 0.
    # actual_home_margin ~ Normal(mean=m, sd=sigma).
    home_cover = _norm_cdf((m + game.home_spread) / cfg.sigma)
    away_cover = 1.0 - home_cover

    if home_cover >= away_cover:
        side, line, price = game.home, game.home_spread, home_price
        cover, model_line = home_cover, -m
    else:
        side, line, price = game.away, -game.home_spread, away_price
        cover, model_line = away_cover, m

    raw = (cover - american_to_prob(price)) * 100.0
    return Pick(
        game=game,
        side=side,
        line=line,
        price=price,
        model_line=model_line,
        cover_prob=cover,
        raw_pct=raw,
        cal_pct=raw * cfg.shrink,
        over_cap=raw > cfg.cap_pct,
    )


def scan(games: list[Game], cfg: Config) -> tuple[list[Pick], list[Pick]]:
    """Score a slate.

    Returns ``(plays, flagged)`` where *plays* are the ranked top-N in-range
    picks (floor < raw edge <= cap) and *flagged* are over-cap games.
    Already-played games are dropped entirely.
    """
    scored = [score_game(g, cfg) for g in games if not g.played]
    scored.sort(key=lambda p: p.raw_pct, reverse=True)

    flagged = [p for p in scored if p.over_cap]
    in_range = [p for p in scored if not p.over_cap and p.raw_pct >= cfg.floor_pct]
    return in_range[: cfg.top_n], flagged


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
_CFG_FOR_FMT = Config()


def _fmt_cap() -> str:
    return f"{_CFG_FOR_FMT.cap_pct:g}%"


def _matchup(game: Game) -> str:
    return f"{game.away} @ {game.home}"


def format_pick(pick: Pick, rank: Optional[int], cfg: Config) -> str:
    g = pick.game
    head = _matchup(g)
    if rank is not None:
        head = f"#{rank}  {head}"
    when = f"  ({g.kickoff})" if g.kickoff else ""
    status = "FLAG - NOT PLAYED" if pick.over_cap else "PLAY"
    lines = [
        f"{head}{when}",
        f"    {status}",
        f"    line       {pick.side} {pick.line:+.1f} @ {pick.price:+d}",
        f"    model       {pick.side} {pick.model_line:+.1f}  "
        f"(cover {pick.cover_prob * 100:.1f}%)",
        f"    raw edge    {pick.raw_pct:+.1f}%   "
        f"cal x{cfg.shrink:g} = {pick.cal_pct:+.1f}%   "
        f"cap {cfg.cap_pct:g}%",
    ]
    return "\n".join(lines)


def format_report(plays: list[Pick], flagged: list[Pick], cfg: Config,
                  total_in: int, total_out: int) -> str:
    out: list[str] = []
    out.append(
        f"Slate: {total_in} upcoming -> {total_out} scored | "
        f"cap {cfg.cap_pct:g}% | shrink x{cfg.shrink:g} | sigma {cfg.sigma:g}"
    )
    out.append("")
    if plays:
        out.append(f"TOP {len(plays)} PLAYS (strongest -> weakest)")
        out.append("=" * 48)
        for i, p in enumerate(plays, 1):
            out.append(format_pick(p, i, cfg))
            out.append("")
    else:
        out.append("No in-range plays today.")
        out.append("")
    if flagged:
        out.append(f"FLAGGED - over the {cfg.cap_pct:g}% cap (not played)")
        out.append("-" * 48)
        for p in flagged:
            out.append(format_pick(p, None, cfg))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def pick_to_dict(pick: Pick, cfg: Config, rank: Optional[int] = None) -> dict:
    g = pick.game
    return {
        "rank": rank,
        "matchup": _matchup(g),
        "kickoff": g.kickoff,
        "side": pick.side,
        "line": round(pick.line, 1),
        "price": pick.price,
        "model_line": round(pick.model_line, 1),
        "cover_prob": round(pick.cover_prob, 4),
        "raw_edge_pct": round(pick.raw_pct, 2),
        "cal_edge_pct": round(pick.cal_pct, 2),
        "cap_pct": cfg.cap_pct,
        "status": "flagged" if pick.over_cap else "play",
    }


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> tuple[list[Game], Config]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"error: could not read data file {path}: {exc}")
    cfg = Config.from_dict(raw.get("config", {}))
    games = [Game.from_dict(g) for g in raw.get("games", [])]
    if not games:
        raise SystemExit(f"error: no games found in {path}")
    return games, cfg


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for name in ("cap_pct", "shrink", "sigma", "top_n", "floor_pct"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    return cfg


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_scan(args: argparse.Namespace) -> None:
    games, cfg = load_data(args.data)
    cfg = _apply_overrides(cfg, args)
    global _CFG_FOR_FMT
    _CFG_FOR_FMT = cfg

    total_in = len(games)
    total_out = sum(1 for g in games if not g.played)
    plays, flagged = scan(games, cfg)

    if args.json:
        payload = {
            "config": {
                "cap_pct": cfg.cap_pct,
                "shrink": cfg.shrink,
                "sigma": cfg.sigma,
                "top_n": cfg.top_n,
            },
            "counts": {"upcoming": total_in, "scored": total_out},
            "plays": [pick_to_dict(p, cfg, i) for i, p in enumerate(plays, 1)],
            "flagged": [pick_to_dict(p, cfg) for p in flagged],
        }
        if args.show_all:
            all_scored = sorted(
                (score_game(g, cfg) for g in games if not g.played),
                key=lambda p: p.raw_pct,
                reverse=True,
            )
            payload["all"] = [pick_to_dict(p, cfg) for p in all_scored]
        print(json.dumps(payload, indent=2))
        return

    if args.show_all:
        all_scored = sorted(
            (score_game(g, cfg) for g in games if not g.played),
            key=lambda p: p.raw_pct,
            reverse=True,
        )
        print(f"All {len(all_scored)} scored games (raw edge, high -> low):")
        print("=" * 48)
        for p in all_scored:
            tag = "FLAG" if p.over_cap else ("PLAY" if p.raw_pct >= cfg.floor_pct else "skip")
            print(f"  [{tag:4}] {p.raw_pct:+6.1f}%  {p.side} {p.line:+.1f}  ({_matchup(p.game)})")
        print()

    print(format_report(plays, flagged, cfg, total_in, total_out))


def cmd_explain(args: argparse.Namespace) -> None:
    games, cfg = load_data(args.data)
    cfg = _apply_overrides(cfg, args)
    key = args.home.lower()
    match = next(
        (g for g in games if key in (g.home.lower(), g.away.lower())), None
    )
    if match is None:
        raise SystemExit(f"error: no game with team {args.home!r} in {args.data}")

    m = model_margin(match, cfg)
    pick = score_game(match, cfg)
    fav = match.home if m >= 0 else match.away
    print(f"{_matchup(match)}")
    if match.kickoff:
        print(f"  kickoff:      {match.kickoff}")
    print(f"  market:       {match.home} {match.home_spread:+.1f}")
    print(f"  model margin: {fav} by {abs(m):.1f} "
          f"(home {m:+.1f}; hfa {cfg.home_field:+g})")
    print(f"  edge side:    {pick.side} {pick.line:+.1f} @ {pick.price:+d}")
    print(f"  cover prob:   {pick.cover_prob * 100:.1f}%  "
          f"(break-even {american_to_prob(pick.price) * 100:.1f}%)")
    print(f"  raw edge:     {pick.raw_pct:+.1f}%")
    print(f"  calibrated:   {pick.cal_pct:+.1f}%  (x{cfg.shrink:g} shrink)")
    print(f"  cap:          {cfg.cap_pct:g}%  ->  "
          f"{'OVER CAP (flag, not played)' if pick.over_cap else 'in range (playable)'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge_finder",
        description="Find daily college-football spread edges from SP+ vs. the market.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", type=Path, required=True,
                       help="JSON file with {config?, games:[...]}.")
        p.add_argument("--cap", dest="cap_pct", type=float,
                       help="Override the raw-overshoot cap percent (default 10).")
        p.add_argument("--shrink", type=float, help="Override the calibration shrink factor.")
        p.add_argument("--sigma", type=float, help="Override the margin SD (points).")
        p.add_argument("--floor", dest="floor_pct", type=float,
                       help="Ignore edges below this percent.")
        p.add_argument("--top", dest="top_n", type=int, help="Max number of plays.")

    p_scan = sub.add_parser("scan", help="Score a slate and print the ranked plays.")
    add_common(p_scan)
    p_scan.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    p_scan.add_argument("--show-all", action="store_true",
                        help="Also list every scored game, not just the top plays.")
    p_scan.set_defaults(func=cmd_scan)

    p_exp = sub.add_parser("explain", help="Show the math for one matchup.")
    add_common(p_exp)
    p_exp.add_argument("--home", required=True, help="A team name in the matchup.")
    p_exp.set_defaults(func=cmd_explain)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
